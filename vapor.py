#!/usr/bin/env python
"""
Credit: https://github.com/zalandoresearch/pytorch-vq-vae
"""
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import time
from datetime import datetime

from six.moves import xrange

import adios2 as ad2
from sklearn.model_selection import train_test_split

import logging
import os
import argparse

## (2020/11) pytorch is resetting affinity. We need to check before torch.
_affinity = None
if hasattr(os, 'sched_getaffinity'):
    _affinity = os.sched_getaffinity(0)

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing as mp
import random

# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.autograd.profiler as profiler

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid

import xgc4py
from tqdm import tqdm

import sys
from datetime import datetime
from pathlib import Path

## Global variables
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
xgcexp = None
Z0, zmu, zsig, zmin, zmax = None, None, None, None, None
pidmap = dict()
args = None
comm, size, rank = None, 1, 0

# %%
def log(*args, logtype='debug', sep=' '):
    getattr(logging, logtype)(sep.join(map(str, args)))

# %%
def init(counter):
    global args
    global pidmap
    global comm, size, rank

    pid = os.getpid()
    with counter.get_lock():
        counter.value += 1
        pidmap[pid] = counter.value

    affinity = None
    ## Set affinity when using ProcessPoolExecutor
    if hasattr(os, 'sched_getaffinity'):
        affinity = os.sched_getaffinity(0)
        ## We leave rank-0 core for the main process
        ## No need to set on Summit
        #i = pidmap[os.getpid()]%args.ncorespernode
        #affinity_mask = {cid[i]}
        #os.sched_setaffinity(0, affinity_mask)
        #affinity = os.sched_getaffinity(0)
    logging.info(f"\tWorker: init. rank={rank} pid={pid} ID={pidmap[pid]} affinity={affinity}")
    # time.sleep(random.randint(1, 5))
    return 0

# %%
def dowork(X, inode, num_channels):
    global xgcexp
    global Z0, zmu, zsig, zmin, zmax

    nnodes = len(Z0)//xgcexp.nphi
    nvp0 = xgcexp.f0mesh.f0_nmu+1
    nvp1 = xgcexp.f0mesh.f0_nvp*2+1

    # print ("dowork:", inode, os.getpid())
    mn = zmin[inode:inode+num_channels]
    mx = zmax[inode:inode+num_channels]

    #f0_f = data[i].cpu().data.numpy()
    f0_f = Z0[inode:inode+num_channels,:nvp0,:nvp1]
    #f0_f *= (mx[:,np.newaxis,np.newaxis]-mn[:,np.newaxis,np.newaxis])
    #f0_f += mn[:,np.newaxis,np.newaxis]
    den0, u_para0, T_perp0, T_para0, _, _ = \
        xgcexp.f0_diag(f0_inode1=inode%nnodes, ndata=num_channels, isp=1, f0_f=f0_f, progress=False)

    f0_f = X
    f0_f *= (mx[:,np.newaxis,np.newaxis]-mn[:,np.newaxis,np.newaxis])
    f0_f += mn[:,np.newaxis,np.newaxis]
    den1, u_para1, T_perp1, T_para1, _, _ = \
        xgcexp.f0_diag(f0_inode1=inode%nnodes, ndata=num_channels, isp=1, f0_f=f0_f, progress=False)
    
    _den_err = np.mean((den0-den1)**2)/np.var(den0)
    _u_para_err = np.mean((u_para0-u_para1)**2)/np.var(u_para0)
    _T_perp_err = np.mean((T_perp0-T_perp1)**2)/np.var(T_perp0)
    _T_para_err = np.mean((T_para0-T_para1)**2)/np.var(T_para0)

    # print ("dowork done:", inode, os.getpid())
    return (_den_err, _u_para_err, _T_perp_err, _T_para_err)

def physics_loss_con(data, lb, data_recon, executor=None):
    global args
    global xgcexp
    global Z0, zmu, zsig, zmin, zmax

    batch_size, num_channels = data.shape[:2]
    nnodes = len(Z0)//xgcexp.nphi
    nvp0 = xgcexp.f0mesh.f0_nmu+1
    nvp1 = xgcexp.f0mesh.f0_nvp*2+1

    Xbar = data_recon.cpu().data.numpy()
    
    den_err = 0.
    u_para_err = 0.
    T_perp_err = 0.
    T_para_err = 0.
    
    # counter = mp.Value('i', 0)
    # with ProcessPoolExecutor(max_workers=args.nworkers, initializer=hello, initargs=(counter,)) as executor:
    if executor is not None:
        futures = list()
        for i in range(batch_size):
            futures.append(executor.submit(dowork, Xbar[i,:num_channels,:nvp0,:nvp1], int(lb[i]), num_channels))

        for f in tqdm(futures):
            _den_err, _u_para_err, _T_perp_err, _T_para_err = f.result()
            #_den_err, _u_para_err, _T_perp_err, _T_para_err = dowork(i)
            den_err += _den_err
            u_para_err += _u_para_err
            T_perp_err += _T_perp_err
            T_para_err += _T_para_err

        # inode = int(lb[i])
        # mn = zmin[inode:inode+num_channels]
        # mx = zmax[inode:inode+num_channels]

        # #f0_f = data[i].cpu().data.numpy()
        # f0_f = Z0[inode:inode+num_channels,:nvp0,:nvp1]
        # #f0_f *= (mx[:,np.newaxis,np.newaxis]-mn[:,np.newaxis,np.newaxis])
        # #f0_f += mn[:,np.newaxis,np.newaxis]
        # den0, u_para0, T_perp0, T_para0, _, _ = \
        #     xgcexp.f0_diag(f0_inode1=inode%nnodes, ndata=num_channels, isp=1, f0_f=f0_f, progress=False)

        # f0_f = Xbar[i,...]
        # f0_f *= (mx[:,np.newaxis,np.newaxis]-mn[:,np.newaxis,np.newaxis])
        # f0_f += mn[:,np.newaxis,np.newaxis]
        # den1, u_para1, T_perp1, T_para1, _, _ = \
        #     xgcexp.f0_diag(f0_inode1=inode%nnodes, ndata=num_channels, isp=1, f0_f=f0_f, progress=False)
        
        # den_err += np.mean((den0-den1)**2)/np.var(den0)
        # u_para_err += np.mean((u_para0-u_para1)**2)/np.var(u_para0)
        # T_perp_err += np.mean((T_perp0-T_perp1)**2)/np.var(T_perp0)
        # T_para_err += np.mean((T_para0-T_para1)**2)/np.var(T_para0)

        return (den_err/batch_size, u_para_err/batch_size, T_perp_err/batch_size, T_para_err/batch_size)

# %%
def physics_loss(data, lb, data_recon):
    global device
    global xgcexp
    global Z0, zmu, zsig, zmin, zmax

    batch_size, num_channels = data.shape[:2]
    nnodes = len(Z0)//xgcexp.nphi
    nvp0 = xgcexp.f0mesh.f0_nmu+1
    nvp1 = xgcexp.f0mesh.f0_nvp*2+1

    Xbar = data_recon.cpu().data.numpy() ## shape: (nbatch, nchannel, nvp0, nvp1)
    
    den_err = 0.
    u_para_err = 0.
    T_perp_err = 0.
    T_para_err = 0.
    
    for i in tqdm(range(batch_size)):
        inode = int(lb[i])
        mn = zmin[inode:inode+num_channels]
        mx = zmax[inode:inode+num_channels]

        #f0_f = data[i].cpu().data.numpy()
        f0_f = Z0[inode:inode+num_channels,:nvp0,:nvp1]
        #f0_f *= (mx[:,np.newaxis,np.newaxis]-mn[:,np.newaxis,np.newaxis])
        #f0_f += mn[:,np.newaxis,np.newaxis]
        den0, u_para0, T_perp0, T_para0, _, _ = \
            xgcexp.f0_diag(f0_inode1=inode%nnodes, ndata=num_channels, isp=1, f0_f=f0_f, progress=False)

        f0_f = Xbar[i,:num_channels,:nvp0,:nvp1]
        f0_f *= (mx[:,np.newaxis,np.newaxis]-mn[:,np.newaxis,np.newaxis])
        f0_f += mn[:,np.newaxis,np.newaxis]
        den1, u_para1, T_perp1, T_para1, _, _ = \
            xgcexp.f0_diag(f0_inode1=inode%nnodes, ndata=num_channels, isp=1, f0_f=f0_f, progress=False)
        
        den_err += np.mean((den0-den1)**2)/np.var(den0)
        u_para_err += np.mean((u_para0-u_para1)**2)/np.var(u_para0)
        T_perp_err += np.mean((T_perp0-T_perp1)**2)/np.var(T_perp0)
        T_para_err += np.mean((T_para0-T_para1)**2)/np.var(T_para0)

    return (den_err/batch_size, u_para_err/batch_size, T_perp_err/batch_size, T_para_err/batch_size)

# %%
def read_f0(istep, expdir=None, iphi=None, inode=0, nnodes=None, average=False, randomread=0.0, nchunk=16, fieldline=False):
    def adios2_get_shape(f, varname):
        nstep = int(f.available_variables()[varname]['AvailableStepsCount'])
        shape = f.available_variables()[varname]['Shape']
        lshape = None
        if shape == '':
            ## Accessing Adios1 file
            ## Read data and figure out
            v = f.read(varname)
            lshape = v.shape
        else:
            lshape = tuple([ int(x.strip(',')) for x in shape.strip().split() ])
        return (nstep, lshape)

    fname = os.path.join(expdir, 'restart_dir/xgc.f0.%05d.bp'%istep)
    if randomread > 0.0:
        ## prefetch to get metadata
        with ad2.open(fname, 'r') as f:
            nstep, nsize = adios2_get_shape(f, 'i_f')
            ndim = len(nsize)
            nphi = nsize[0]
            _nnodes = nsize[2] if nnodes is None else nnodes
            nmu = nsize[1]
            nvp = nsize[3]
        assert _nnodes%nchunk == 0
        _lnodes = list(range(inode, inode+_nnodes, nchunk))
        lnodes = random.sample(_lnodes, k=int(len(_lnodes)*randomread))
        lnodes = np.sort(lnodes)

        lf = list()
        li = list()
        for i in tqdm(lnodes):
            li.append(np.array(range(i,i+nchunk), dtype=np.int32))
            with ad2.open(fname, 'r') as f:
                nphi = nsize[0] if iphi is None else 1
                iphi = 0 if iphi is None else iphi
                start = (iphi,0,i,0)
                count = (nphi,nmu,nchunk,nvp)
                _f = f.read('i_f', start=start, count=count).astype('float64')
                lf.append(_f)
        i_f = np.concatenate(lf, axis=2)
        lb = np.concatenate(li)
    elif fieldline is True:
        import networkx as nx

        fname2 = os.path.join(expdir, 'xgc.mesh.bp')
        with ad2.open(fname2, 'r') as f:
            _nnodes = int(f.read('n_n', ))
            nextnode = f.read('nextnode')
        
        G = nx.Graph()
        for i in range(_nnodes):
            G.add_node(i)
        for i in range(_nnodes):
            G.add_edge(i, nextnode[i])
            G.add_edge(nextnode[i], i)
        cc = [x for x in list(nx.connected_components(G)) if len(x) >= nchunk]

        li = list()
        for k, components in enumerate(cc):
            DG = nx.DiGraph()
            for i in components:
                DG.add_node(i)
            for i in components:
                DG.add_edge(i, nextnode[i])
            
            cycle = list(nx.find_cycle(DG))
            DG.remove_edge(*cycle[-1])
            
            path = nx.dag_longest_path(DG)
            #print (k, len(components), path[0])
            for i in path[:len(path)-len(path)%nchunk]:
                li.append(i)

        with ad2.open(fname, 'r') as f:
            nstep, nsize = adios2_get_shape(f, 'i_f')
            ndim = len(nsize)
            nphi = nsize[0] if iphi is None else 1
            iphi = 0 if iphi is None else iphi
            _nnodes = nsize[2]
            nmu = nsize[1]
            nvp = nsize[3]
            start = (iphi,0,0,0)
            count = (nphi,nmu,_nnodes,nvp)
            logging.info (f"Reading: {start} {count}")
            i_f = f.read('i_f', start=start, count=count).astype('float64')
        
        _nnodes = len(li)-inode if nnodes is None else nnodes
        lb = np.array(li[inode:inode+_nnodes], dtype=np.int32)
        logging.info (f"Fieldline: {len(lb)}")
        logging.info (f"{lb}")
        i_f = i_f[:,:,lb,:]
    else:
        with ad2.open(fname, 'r') as f:
            nstep, nsize = adios2_get_shape(f, 'i_f')
            ndim = len(nsize)
            nphi = nsize[0] if iphi is None else 1
            iphi = 0 if iphi is None else iphi
            _nnodes = nsize[2] if nnodes is None else nnodes
            nmu = nsize[1]
            nvp = nsize[3]
            start = (iphi,0,inode,0)
            count = (nphi,nmu,_nnodes,nvp)
            logging.info (f"Reading: {start} {count}")
            i_f = f.read('i_f', start=start, count=count).astype('float64')
            #e_f = f.read('e_f')
        li = list(range(inode, inode+_nnodes))
        lb = np.array(li, dtype=np.int32)

    if i_f.shape[3] == 31:
        i_f = np.append(i_f, i_f[...,30:31], axis=3)
        # e_f = np.append(e_f, e_f[...,30:31], axis=3)
    if i_f.shape[3] == 39:
        i_f = np.append(i_f, i_f[...,38:39], axis=3)
        i_f = np.append(i_f, i_f[:,38:39,:,:], axis=1)

    Z0 = np.moveaxis(i_f, 1, 2)

    if average:
        Z0 = np.mean(Z0, axis=0)
        zlb = lb
    else:
        Z0 = Z0.reshape((-1,Z0.shape[2],Z0.shape[3]))
        _lb = list()
        for i in range(nphi):
            _lb.append( i*100_000_000 + lb)
        zlb = np.concatenate(_lb)
    
    #zlb = np.concatenate(li)
    zmu = np.mean(Z0, axis=(1,2))
    zsig = np.std(Z0, axis=(1,2))
    zmin = np.min(Z0, axis=(1,2))
    zmax = np.max(Z0, axis=(1,2))
    Zif = (Z0 - zmin[:,np.newaxis,np.newaxis])/(zmax-zmin)[:,np.newaxis,np.newaxis]

    return (Z0, Zif, zmu, zsig, zmin, zmax, zlb)

# %%
""" Gradient averaging. """
def average_gradients(model):
    try:
        MPI
    except NameError:
        return
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    #import pdb; pdb.set_trace()
    for i, (name, param) in enumerate(model.named_parameters()):
        if param.grad is not None:
            #print ('[%d] %d: %s (b): %f '%(rank, i, name, param.grad.data.sum()))
            x = comm.allreduce(param.grad.data, op=MPI.SUM)
            param.grad.data = x/size
            #print ('[%d] %d: %s (a): %f'%(rank, i, name, param.grad.data.sum()))
        else:
            #print ('[%d] %d: %s (b): %f '%(rank, i, name, param.data.sum()))
            x = comm.allreduce(param.data, op=MPI.SUM)
            param.data = x/size
            #print ('[%d] %d: %s (a): %f'%(rank, i, name, param.data.sum()))

# %%
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.uniform_(-1/self._num_embeddings, 1/self._num_embeddings)
        self._commitment_cost = commitment_cost

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        #import pdb; pdb.set_trace()
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        #import pdb; pdb.set_trace()
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        q_latent_loss = torch.mean((quantized - inputs.detach())**2)
        loss = q_latent_loss + self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

# %%
class VectorQuantizerEMA(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, decay, epsilon=1e-5):
        super(VectorQuantizerEMA, self).__init__()
        
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings
        
        self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self._embedding.weight.data.normal_()
        self._commitment_cost = commitment_cost
        
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
        self._ema_w.data.normal_()
        
        self._decay = decay
        self._epsilon = epsilon

    def forward(self, inputs):
        # convert inputs from BCHW -> BHWC
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape
        
        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)
        
        # Calculate distances
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self._embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self._embedding.weight.t()))
            
        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings).to(device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size = self._ema_cluster_size * self._decay + \
                                     (1 - self._decay) * torch.sum(encodings, 0)
            
            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size = (
                (self._ema_cluster_size + self._epsilon)
                / (n + self._num_embeddings * self._epsilon) * n)
            
            dw = torch.matmul(encodings.t(), flat_input)
            self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)
            
            self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))
        
        # Quantize and unflatten
        quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
        
        # Loss
        e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
        loss = self._commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        # convert quantized from BHWC -> BCHW
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, encodings

# %%
class Residual(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(Residual, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hiddens,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hiddens,
                      out_channels=num_hiddens,
                      kernel_size=1, stride=1, bias=False)
        )
    
    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

# %%
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, rescale=None):
        super(Encoder, self).__init__()

        self._rescale=rescale
        log ("Model rescale:", self._rescale)
        log ("in_channels, num_hiddens:", in_channels, num_hiddens)

        self._conv_0 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=4,
                                 stride=2, padding=0)
        # (2020/11) possible kernel size
        # kernel_size=4, stride=2, padding=1
        # kernel_size=3, stride=2, padding=1
        # kernel_size=2, stride=2, padding=0
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=3, stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=3, stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3, stride=2, padding=1)

        # kernel_size=3, stride=1, padding=1
        # kernel_size=2, stride=2, padding=0
        self._conv_4 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3, stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        # import pdb; pdb.set_trace()
        # (2020/11) Testing with resize
        x = inputs
        if self._rescale is not None:
            x = F.interpolate(inputs, size=x.shape[-1]*self._rescale)
            x = self._conv_0(x)
            x = F.relu(x)

        x = self._conv_1(x)
        x = F.relu(x)
        
        x = self._conv_2(x)
        x = F.relu(x)
        
        x = self._conv_3(x)
        x = F.relu(x)

        x = self._conv_4(x)
        
        x = self._residual_stack(x)
        return x

# %%
class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, num_channels):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        # (2020/11) possible kernel size
        # kernel_size=4, stride=2, padding=1, output_padding=0
        # kernel_size=3, stride=2, padding=1, output_padding=1
        # kernel_size=2, stride=2, padding=0, output_padding=0
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=3, stride=2, padding=1, output_padding=1)
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=3, stride=2, padding=1, output_padding=1)
        self._conv_trans_3 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=num_channels,
                                                kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, inputs):
        # import pdb; pdb.set_trace()
        x = inputs
        x = self._conv_1(x)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        x = self._conv_trans_2(x)
        x = F.relu(x)

        x = self._conv_trans_3(x)
        return x

# %%
"""
Credit: https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
"""
class AE(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=128
        )
        self.encoder_output_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=128
        )
        self.decoder_output_layer = nn.Linear(
            in_features=128, out_features=kwargs["input_shape"]
        )

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.tanh(activation)
        code = self.encoder_output_layer(activation)
        code = torch.tanh(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.tanh(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.tanh(activation)
        return reconstructed

# %%
"""
Credit: https://medium.com/@vaibhaw.vipul/building-autoencoder-in-pytorch-34052d1d280c
"""
class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 6, kernel_size=5),
            nn.ReLU(True),
            nn.Conv2d(6,16,kernel_size=5),
            nn.ReLU(True))
        self.decoder = nn.Sequential(             
            nn.ConvTranspose2d(16,6,kernel_size=5),
            nn.ReLU(True),
            nn.ConvTranspose2d(6,3,kernel_size=5),
            nn.ReLU(True))
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# %%
class Model(nn.Module):
    def __init__(self, num_channels, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost, decay=0, rescale=None, learndiff=False, input_shape=None):
        super(Model, self).__init__()
        
        self._encoder = Encoder(num_channels, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens, rescale=rescale)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hiddens, 
                                      out_channels=embedding_dim,
                                      kernel_size=1, 
                                      stride=1)
        if decay > 0.0:
            self._vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                              commitment_cost, decay)
        else:
            self._vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                           commitment_cost)
        self._decoder = Decoder(embedding_dim,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens, num_channels)

        """
        Learn diff
        """
        self._learndiff = learndiff
        log ("Model learndiff: %s"%self._learndiff)
        if self._learndiff:
            """
            self._encoder2 = Encoder(num_channels, num_hiddens,
                                    num_residual_layers, 
                                    num_residual_hiddens, rescale=rescale)
            self._pre_vq_conv2 = nn.Conv2d(in_channels=num_hiddens, 
                                        out_channels=embedding_dim,
                                        kernel_size=1, 
                                        stride=1)
            if decay > 0.0:
                self._vq_vae2 = VectorQuantizerEMA(num_embeddings, embedding_dim, 
                                                commitment_cost, decay)
            else:
                self._vq_vae2 = VectorQuantizer(num_embeddings, embedding_dim,
                                            commitment_cost)
            self._decoder2 = Decoder(embedding_dim,
                                    num_hiddens, 
                                    num_residual_layers, 
                                    num_residual_hiddens, num_channels)
            """

            self._input_shape = input_shape
            nbatch, nchannel, dim1, dim2 = self._input_shape
            self._dmodel = AE(input_shape=nchannel*dim1*dim2)
            #self._doptimizer = optim.Adam(self._dmodel.parameters(), lr=1e-3)
            self._dcriterion = nn.MSELoss()
        
    def forward(self, x):
        #import pdb; pdb.set_trace()
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)
        drecon = 0
        dloss = 0

        if self._learndiff:
            # z2 = self._encoder2(x-x_recon)
            # z2 = self._pre_vq_conv2(z2)
            # loss2, quantized2, perplexity2, _ = self._vq_vae2(z2)
            # x_recon2 = self._decoder2(quantized2)
            # return loss+loss2, x_recon+x_recon2, perplexity+perplexity2

            nbatch, nchannel, dim1, dim2 = x.shape
            dx = (x-x_recon).view(-1, nchannel*dim1*dim2)
            outputs = self._dmodel(dx)
            dloss = self._dcriterion(outputs, dx)
            drecon = outputs.view(-1, nchannel, dim1, dim2)

        return loss, x_recon+drecon, perplexity, dloss


# %%
def load_checkpoint(DIR, prefix, model):
    import hashlib
    import sys

    ## (2020/06) no use anymore
    # hcode = hashlib.md5(str(model).encode()).hexdigest()
    # print ('hash:', hcode)
    _prefix = '%s/%s'%(DIR, prefix)
    _istart = None
    _model = None
    _dmodel = None
    _err = None
    try:
        with open('%s/checkpoint.txt'%(_prefix), 'r') as f:
            _istart = int(f.readline())
        fname = '%s/checkpoint.%d.pytorch'%(_prefix, _istart)
        log ('Checkpoint:', fname)
        _model = torch.load(fname)
        _model.eval()
    except:
        log ("Error:", sys.exc_info()[0])
        log ("No restart info")
        pass

    try:
        fname = '%s/checkpoint-dmodel.%d.pytorch'%(_prefix, _istart)
        log ('Checkpoint:', fname)
        _dmodel = torch.load(fname)
    except:
        pass

    return (_istart, _model, _dmodel)

# %%
def save_checkpoint(DIR, prefix, model, err, epoch, dmodel=None):
    import hashlib
    from pathlib import Path

    ## (2020/06) no use anymore
    # hcode = hashlib.md5(str(model).encode()).hexdigest()
    _prefix = '%s/%s'%(DIR, prefix)
    Path(_prefix).mkdir(parents=True, exist_ok=True)
    with open("%s/model.txt"%(_prefix), 'w+') as f:
        f.write(str(model))
    fname = '%s/err.%d.npz'%(_prefix, epoch)
    np.savez(fname, err=err)
    fname = '%s/checkpoint.%d.pytorch'%(_prefix, epoch)
    torch.save(model, fname)
    if dmodel is not None:
        fname = '%s/checkpoint-dmodel.%d.pytorch'%(_prefix, epoch)
        torch.save(dmodel, fname)
    with open('%s/checkpoint.txt'%(_prefix), 'w+') as f:
        f.write(str(epoch))
    log ("Saved checkpoint: %s"%(fname))

def main():
    global device
    global xgcexp
    global Z0, zmu, zsig, zmin, zmax
    global pidmap
    global args
    global comm, size, rank

    # %%
    # Main start
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', help='exp', default='m1')
    parser.add_argument('-n', '--num_training_updates', help='num_training_updates (default: %(default)s)', type=int, default=10_000)
    parser.add_argument('-E', '--embedding_dim', help='embedding_dim (default: %(default)s)', type=int, default=64)
    parser.add_argument('-H', '--num_hiddens', help='num_hidden (default: %(default)s)', type=int, default=128)
    parser.add_argument('-R', '--num_residual_hiddens', help='num_residual_hiddens (default: %(default)s)', type=int, default=32)
    parser.add_argument('-L', '--num_residual_layers', help='num_residual_layers (default: %(default)s)', type=int, default=2)
    parser.add_argument('-C', '--num_channels', help='num_channels', type=int, default=16)
    parser.add_argument('-B', '--batch_size', help='batch_size (default: %(default)s)', type=int, default=256)
    parser.add_argument('-d', '--device_id', help='device_id (default: %(default)s)', type=int, default=0)
    parser.add_argument('--wdir', help='working directory (default: current)', default=os.getcwd())
    parser.add_argument('--datadir', help='data directory (default: %(default)s)', default='data')
    parser.add_argument('--timesteps', help='timesteps', nargs='+', type=int)
    parser.add_argument('--average_interval', help='average_interval (default: %(default)s)', type=int)
    parser.add_argument('--log_interval', help='log_interval (default: %(default)s)', type=int, default=1_000)
    parser.add_argument('--checkpoint_interval', help='checkpoint_interval (default: %(default)s)', type=int, default=10_000)
    parser.add_argument('--nompi', help='nompi', action='store_true')
    parser.add_argument('--seed', help='seed (default: %(default)s)', type=int)
    parser.add_argument('--nworkers', help='nworkers (default: %(default)s)', type=int)
    parser.add_argument('--physicsloss', help='physicsloss', action='store_true')
    parser.add_argument('--physicsloss_interval', help='physicsloss_interval (default: %(default)s)', type=int, default=1)
    parser.add_argument('--randomread', help='randomread', type=float, default=0.0)
    parser.add_argument('--iphi', help='iphi', type=int)
    parser.add_argument('--splitfiles', help='splitfiles', action='store_true')
    parser.add_argument('--overwrap', help='overwrap', type=int, default=1)
    parser.add_argument('--inode', help='inode', type=int, default=0)
    parser.add_argument('--nnodes', help='nnodes', type=int, default=None)
    parser.add_argument('--rescale', help='rescale', type=int, default=None)
    parser.add_argument('--learndiff', help='learndiff', action='store_true')
    parser.add_argument('--learndiff2', help='learndiff2', action='store_true')
    parser.add_argument('--fieldline', help='fieldline', action='store_true')
    parser.add_argument('--overwrite', help='overwrite', action='store_true')
    parser.add_argument('--log', help='log', action='store_true')
    args = parser.parse_args()

    DIR=args.wdir
    prefix='exp-B%d-C%d-H%d-R%d-L%d-E%d-%s'%\
        (args.batch_size, args.num_channels, args.num_hiddens, args.num_residual_hiddens, args.num_residual_layers, args.embedding_dim, args.exp)

    if not args.nompi:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        size = comm.Get_size()
        rank = comm.Get_rank()
    else:
        comm = None
        size = 1
        rank = 0

    fmt = '[%d:%%(levelname)s] %%(message)s'%(rank)
    handlers = [logging.StreamHandler()]
    if args.log:
        _prefix = '%s/%s'%(DIR, prefix)
        Path(_prefix).mkdir(parents=True, exist_ok=True)
        suffix = datetime.now().strftime("%Y%m%d-%H%M%S")
        pid = os.getpid()
        fname = "%s/%s-%s-%d.log"%(_prefix, prefix, suffix, pid)
        handlers.append(logging.FileHandler(fname))
    logging.basicConfig(level=logging.DEBUG, format=fmt, handlers=handlers)

    logging.info("Command: {0}\n".format(" ".join([x for x in sys.argv]))) 
    logging.debug("All settings used:") 
    for k,v in sorted(vars(args).items()): 
        logging.debug("\t{0}: {1}".format(k,v))

    logging.info ('EXP: %s' % args.exp)
    logging.info ('embedding_dim: %d' % args.embedding_dim)
    logging.info ('DIR: %s' % args.wdir)

    # %%
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info ('device: %s' % device)

    if args.seed is not None:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        if device.type == 'cuda':
            torch.cuda.manual_seed(args.seed)

    # %%
    num_channels = args.num_channels

    ## This is the original setting
        # batch_size = 256
        # num_hiddens = 128
        # num_residual_hiddens = 32
        # num_residual_layers = 2
        # embedding_dim = 64
        # num_embeddings = 512

    ## This is for small size
        # batch_size = 256
        # num_channels = 4
        # num_hiddens = 8
        # num_residual_hiddens = 8
        # num_residual_layers = 2
        # embedding_dim = 4
        # num_embeddings = 512

    ## This is the setting for quality?
        # batch_size = 256
        # num_training_updates = 15000
        # num_channels = 16
        # num_hiddens = 128
        # num_residual_hiddens = 32
        # num_residual_layers = 2
        # embedding_dim = 64
        # num_embeddings = 512

    batch_size = args.batch_size

    num_hiddens = args.num_hiddens
    num_residual_hiddens = args.num_residual_hiddens
    num_residual_layers = args.num_residual_layers

    embedding_dim = args.embedding_dim
    num_embeddings = 512

    commitment_cost = 0.25
    decay = 0.99
    learning_rate = 1e-3

    #prefix='xgc-%s-batch%d-edim%d-nhidden%d-nchannel%d-nresidual_hidden%d'%(args.exp, args.batch_size, args.embedding_dim, args.num_hiddens, args.num_channels, args.num_residual_hiddens)
    logging.info ('prefix: %s' % prefix)

    # %%
    xgcexp = xgc4py.XGC(args.datadir)
    nnodes = xgcexp.mesh.nnodes if args.nnodes is None else args.nnodes
    
    timesteps = args.timesteps
    if args.splitfiles:
        timesteps = np.array_split(np.array(timesteps), size)[rank]
    f0_data_list = list()
    logging.info (f'Data dir: {args.datadir}')
    for istep in timesteps:
        logging.info (f'Reading: {istep}')
        f0_data_list.append(read_f0(istep, expdir=args.datadir, iphi=args.iphi, inode=args.inode, nnodes=nnodes, \
            randomread=args.randomread, nchunk=num_channels, fieldline=args.fieldline))

    lst = list(zip(*f0_data_list))

    global Z0, zmu, zsig, zmin, zmax, zlb
    Z0 = np.r_[(lst[0])]
    Zif = np.r_[(lst[1])]
    zmu = np.r_[(lst[2])]
    zsig = np.r_[(lst[3])]
    zmin = np.r_[(lst[4])]
    zmax = np.r_[(lst[5])]
    zlb = np.r_[(lst[6])]
    ## z-score normalization
    #Zif = (Zif - zmu[:,np.newaxis,np.newaxis])/zsig[:,np.newaxis,np.newaxis]
    ## min-max normalization
    #Zif = (Zif - np.min(Zif))/(np.max(Zif)-np.min(Zif))
    #Zif = (Zif - zmin[:,np.newaxis,np.newaxis])/(zmax-zmin)[:,np.newaxis,np.newaxis]

    log ('Zif bytes,shape:', Zif.size * Zif.itemsize, Zif.shape, zmu.shape, zsig.shape)
    log ('Minimum training epoch:', Zif.shape[0]/batch_size)

    lx = list()
    ly = list()
    for i in range(0,len(Zif)-num_channels+1,num_channels//args.overwrap):
        X = Zif[i:i+num_channels,:,:]
        mu = zmu[i:i+num_channels]
        sig = zsig[i:i+num_channels]
        N = X.astype(np.float32)
        #N = rescale(N, 2.0, anti_aliasing=False, multichannel=True)
        ## z-score normalization
        #N = (X - mu[:,np.newaxis,np.newaxis])/sig[:,np.newaxis,np.newaxis]
        lx.append(N)
        ly.append(zlb[i:i+num_channels])

    data_variance = np.var(lx, dtype=np.float64)
    log ('data_variance', data_variance)

    # %% 
    # Loadding
    # X_train, X_test, y_train, y_test = train_test_split(lx, ly, test_size=0.10, random_state=42)
    # training_data = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    # validation_data = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))    
    # (2020/11) Temporary. Use all data for training
    training_data = torch.utils.data.TensorDataset(torch.Tensor(lx), torch.Tensor(ly))
    validation_data = torch.utils.data.TensorDataset(torch.Tensor(lx), torch.Tensor(ly))    

    training_loader = DataLoader(training_data, 
                                batch_size=batch_size, 
                                shuffle=True,
                                pin_memory=True)
    validation_loader = DataLoader(validation_data,
                                batch_size=batch_size,
                                shuffle=True,
                                pin_memory=True)

    # %% 
    # Model
    if args.learndiff:
        model = Model(num_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                    num_embeddings, embedding_dim, 
                    commitment_cost, decay, rescale=args.rescale, learndiff=args.learndiff, 
                    input_shape=[batch_size, num_channels, Zif.shape[-2], Zif.shape[-1]]).to(device)
    else:
        model = Model(num_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                    num_embeddings, embedding_dim, 
                    commitment_cost, decay, rescale=args.rescale, learndiff=args.learndiff).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    dmodel = None
    if args.learndiff2:
        dim1, dim2 = Zif.shape[-2], Zif.shape[-1]
        dmodel = AE(input_shape=num_channels*dim1*dim2).to(device)
        doptimizer = optim.Adam(dmodel.parameters(), lr=1e-3)
        dcriterion = nn.MSELoss()

    model.train()
    train_res_recon_error = []
    train_res_perplexity = []
    train_res_physics_error = []
    istart = 1

    # %%
    # Load checkpoint
    _istart, _model, _dmodel = 0, None, None
    if not args.overwrite: _istart, _model, _dmodel = load_checkpoint(DIR, prefix, model)
    if _model is not None:
        istart = _istart + 1
        model = _model
    log (istart)

    # %%
    nworkers = args.nworkers if args.nworkers is not None else 8
    if args.nworkers is None and hasattr(os, 'sched_getaffinity'):
        nworkers = len(os.sched_getaffinity(0))-1 
    logging.info('Nworkers: %d'%nworkers)

    counter = mp.Value('i', 0)
    executor = ProcessPoolExecutor(max_workers=nworkers, initializer=init, initargs=(counter,))
    num_training_updates=args.num_training_updates
    logging.info ('Training: %d' % num_training_updates)
    model.train()
    t0 = time.time()
    for i in xrange(istart, istart+num_training_updates):
        (data, lb) = next(iter(training_loader))
        # print ("Training:", lb)
        ns = torch.Tensor(np.random.normal(0.0, data.numpy()*0.1, size=data.numpy().shape)).to(device)
        data = data.to(device)
        optimizer.zero_grad() # clear previous gradients
        
        vq_loss, data_recon, perplexity, dloss = model(data+ns)
        ## mean squared error: torch.mean((data_recon - data)**2)
        ## relative variance
        recon_error = F.mse_loss(data_recon, data) / data_variance
        physics_error = 0.0
        if args.physicsloss and (i % args.physicsloss_interval == 0):
            den_err, u_para_err, T_perp_err, T_para_err = physics_loss_con(data, lb, data_recon, executor=executor)
            ds = np.mean(data_recon.cpu().data.numpy()**2)
            # print (lb[0], recon_error.data, vq_loss.data, den_err, u_para_err, T_perp_err, T_para_err, ds)
            # physics_error += den_err/ds * torch.mean(data_recon)
            physics_error += den_err
        loss = recon_error + vq_loss + physics_error + dloss
        loss.backward()

        if args.learndiff2:
            doptimizer.zero_grad() # clear previous gradients
            dim1, dim2 = Zif.shape[-2], Zif.shape[-1]
            dx = (data-data_recon).view(-1,num_channels*dim1*dim2).detach()
            drecon = dmodel(dx)
            dloss = dcriterion(drecon, dx)
            dloss.backward()
            doptimizer.step()

        if (args.average_interval is not None) and (i%args.average_interval == 0):
            ## Gradient averaging
            logging.info('iteration %d: gradient averaging' % (i))
            average_gradients(model)
        optimizer.step()

        #print ('AFTER', model._vq_vae._embedding.weight.data.numpy().sum())
        
        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())
        train_res_physics_error.append(physics_error)

        if i % args.log_interval == 0:
            logging.info(f'{i} time: {time.time()-t0:.3f}')
            logging.info(f'{i} Avg: {np.mean(train_res_recon_error[-args.log_interval:]):g} {np.mean(train_res_perplexity[-args.log_interval:]):g} {np.mean(train_res_physics_error[-args.log_interval:]):g}')
            logging.info(f'{i} Loss: {recon_error.item():g} {vq_loss.data.item():g} {perplexity.item():g} {physics_error:g} {dloss:g} {len(training_loader.dataset)} {len(data)}')
            if args.learndiff2:
                logging.info(f'{i} dloss: {dloss.item():g}')
            # logging.info('')
        
        if (i % args.checkpoint_interval == 0) and (rank == 0):
            save_checkpoint(DIR, prefix, model, train_res_recon_error, i, dmodel=dmodel)
    istart=istart+num_training_updates

    # %%
    model.eval()
    (valid_originals, valid_labels) = next(iter(validation_loader))
    valid_originals = valid_originals.to(device)

    vq_encoded = model._encoder(valid_originals)
    vq_output_eval = model._pre_vq_conv(vq_encoded)
    _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
    valid_reconstructions = model._decoder(valid_quantize)

    logging.info ('Original: %s' % (valid_originals.cpu().numpy().shape,))
    logging.info ('Encoded: %s' % (vq_encoded.detach().cpu().numpy().shape,))
    logging.info ('Quantized: %s' % (valid_quantize.detach().cpu().numpy().shape,))
    logging.info ('Reconstructed: %s' % (valid_reconstructions.detach().cpu().numpy().shape,))
    logging.info ('compression ratio: %.2fx'%(valid_originals.cpu().numpy().size/vq_encoded.detach().cpu().numpy().size))
    logging.info ('compression ratio: %.2fx'%(valid_originals.cpu().numpy().size/valid_quantize.detach().cpu().numpy().size))

if __name__ == "__main__":
    # torch.set_default_tensor_type(torch.DoubleTensor)

    ## (2020/11) Temporary fix. pytorch reset affinity. This is to rollback.
    if hasattr(os, 'sched_getaffinity'):
        os.sched_setaffinity(0, _affinity)
    
    main()
    #with profiler.profile() as prof:
    # from pytracing import TraceProfiler
    # tp = TraceProfiler(output=open('trace.out', 'wb'))
    # with tp.traced():
    #     main()
    #prof.export_chrome_trace("trace.json")
    
