#!/usr/bin/env python
from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
import time
from datetime import datetime

from six.moves import xrange

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

import adios2 as ad2
from sklearn.model_selection import train_test_split

import logging
import os
import argparse

# %%
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

# %%
def f0_diag(isp, f0_inode1, ndata, f0_nmu, f0_nvp, f0_T_ev, f0_grid_vol_vonly, f0_f, f0_dvp):
    """
    isp: electron(=0) or ion(=1)
    (nphi: int)
    (nnodes: int)
    f0_inode1: int
    ndata: int (f0_inode2=f0_inode1+ndata)
    f0_nmu: int
    f0_nvp: int
    f0_T_ev: (nsp, nnodes)
    f0_grid_vol_vonly: (nsp, nnodes)
    f0_f: (ndata, f0_nmu, f0_nvp)
    f0_dvp: double
    uparr: (1, 1, f0_nmu, f0_inode2-f0_inode1+1, f0_nvp)
    """
    if f0_f.ndim == 2:
        f0_f = f0_f[np.newaxis,:]
    #print (f0_f.shape, (ndata, f0_nmu+1, f0_nvp*2+1))
    assert(f0_f.shape[0] == ndata)
    assert(f0_f.shape[1] == f0_nmu+1)
    assert(f0_f.shape[2] >= f0_nvp*2+1)
    
    sml_e_charge=1.6022E-19  ## electron charge (MKS)
    sml_ev2j=sml_e_charge

    ptl_e_mass_au=2E-2
    ptl_mass_au=2E0
    sml_prot_mass=1.6720E-27 ## proton mass (MKS)
    ptl_mass = [ptl_e_mass_au*sml_prot_mass, ptl_mass_au*sml_prot_mass]

    ## index: imu, range: [0, f0_nmu]
    mu_vol = np.ones(f0_nmu+1)
    mu_vol[0] = 0.5
    mu_vol[-1] = 0.5

    ## index: ivp, range: [-f0_nvp, f0_nvp]
    vp_vol = np.ones(f0_nvp*2+1)
    vp_vol[0] = 0.5
    vp_vol[-1] = 0.5
    
    f0_smu_max = 3.0
    f0_dsmu = f0_smu_max/f0_nmu
    mu = (np.arange(f0_nmu+1, dtype=np.float64)*f0_dsmu)**2

    # out
    den = np.zeros((ndata, f0_nmu+1, 2*f0_nvp+1))
    u_para = np.zeros((ndata, f0_nmu+1, 2*f0_nvp+1))
    T_perp = np.zeros((ndata, f0_nmu+1, 2*f0_nvp+1))
    T_para = np.zeros((ndata, f0_nmu+1, 2*f0_nvp+1))


    for inode in range(0, ndata):
        ## Mesh properties
        en_th = f0_T_ev[isp,f0_inode1+inode]*sml_ev2j
        vth = np.sqrt(en_th/ptl_mass[isp])
        f0_grid_vol = f0_grid_vol_vonly[isp,f0_inode1+inode]

        for imu in range(0, f0_nmu+1):
            for ivp in range(0, f0_nvp*2+1):
                ## Vspace properties
                f = f0_f[inode, imu, ivp] #f0_f(ivp,inode,imu,isp)
                vol = f0_grid_vol * mu_vol[imu] * vp_vol[ivp]
                vp = (ivp - f0_nvp) * f0_dvp
                en = 0.5 * mu[imu]

                den[inode, imu, ivp] = f * vol
                u_para[inode, imu, ivp] = f * vol * vp * vth
                T_perp[inode, imu, ivp] = f * vol * en * vth**2 * ptl_mass[isp]
                
                en = 0.5 * (vp - u_para[inode, imu, ivp] / vth)**2
                T_para[inode, imu, ivp] = f * vol * en * vth**2 * ptl_mass[isp]

    for inode in range(0, ndata):
        u_para[inode,:] = u_para[inode,:]/np.sum(den[inode,:])
        T_perp[inode,:] = T_perp[inode,:]/np.sum(den[inode,:])/sml_e_charge
        T_para[inode,:] = 2.0*T_para[inode,:]/np.sum(den[inode,:])/sml_e_charge
                                        
    return (den, u_para, T_perp, T_para)

# %%
def physics_loss(data, lb, zmu, zsig, data_recon):
    batch_size, num_channels = data.shape[:2]
    
    den_err = 0.
    u_para_err = 0.
    T_perp_err = 0.
    T_para_err = 0.
    
    for i in range(batch_size):
        inode = int(lb[i])
        mu = zmu[inode:inode+num_channels]
        sig = zsig[inode:inode+num_channels]

        f0_f = data[i].cpu().data.numpy()
        f0_f *= sig[:,np.newaxis,np.newaxis]
        f0_f += mu[:,np.newaxis,np.newaxis]
        den0, u_para0, T_perp0, T_para0 = f0_diag(1, i, num_channels, f0_nmu, f0_nvp, 
                                              f0_T_ev, f0_grid_vol_vonly, 
                                              f0_f, f0_dvp)

        f0_f = data_recon[i].cpu().data.numpy()
        f0_f *= sig[:,np.newaxis,np.newaxis]
        f0_f += mu[:,np.newaxis,np.newaxis]
        den1, u_para1, T_perp1, T_para1 = f0_diag(1, i, num_channels, f0_nmu, f0_nvp, 
                                              f0_T_ev, f0_grid_vol_vonly, 
                                              f0_f, f0_dvp)
        
        den_err += np.mean((den0-den1)**2)/np.var(den0)
        u_para_err += np.mean((u_para0-u_para1)**2)/np.var(u_para0)
        T_perp_err += np.mean((T_perp0-T_perp1)**2)/np.var(T_perp0)
        T_para_err += np.mean((T_para0-T_para1)**2)/np.var(T_para0)

    return (den_err/batch_size, u_para_err/batch_size, T_perp_err/batch_size, T_para_err/batch_size)

# %%
def read_f0(istep, dir='data', full=False, iphi=None, inode=0, nnodes=None):
    fname = '%s/restart_dir/xgc.f0.%05d.bp'%(dir,istep)
    with ad2.open(fname, 'r') as f:
        nstep, nsize = adios2_get_shape(f, 'i_f')
        ndim = len(nsize)
        nphi = nsize[0]
        nnodes = nsize[2] if nnodes is None else nnodes
        nmu = nsize[1]
        nvp = nsize[3]
        start = (0,0,inode,0)
        count = (nphi,nmu,nnodes,nvp)
        print ("Reading: ", start, count)
        i_f = f.read('i_f', start=start, count=count).astype('float32')
        #e_f = f.read('e_f')

    if i_f.shape[3] == 31:
        i_f = np.append(i_f, i_f[...,30:31], axis=3)
        # e_f = np.append(e_f, e_f[...,30:31], axis=3)
    if i_f.shape[3] == 39:
        i_f = np.append(i_f, i_f[...,38:39], axis=3)
        i_f = np.append(i_f, i_f[:,38:39,:,:], axis=1)

    if full:
        Zif = np.moveaxis(i_f, 1, 2).reshape((-1,i_f.shape[1],i_f.shape[3]))
    else:
        if iphi is not None:
            Zif = np.moveaxis(i_f, 1, 2)
            Zif = Zif[iphi,:]
        else:
            Zif = np.einsum('ijkl->kjl', i_f)/i_f.shape[0]
            # Zef = np.einsum('ijkl->kjl', e_f)/sml_nphi
    
    zmu = np.mean(Zif, axis=(1,2))
    zsig = np.std(Zif, axis=(1,2))
    zmin = np.min(Zif, axis=(1,2))
    zmax = np.max(Zif, axis=(1,2))
    #Zif.shape, zmu.shape, zsig.shape

    return (Zif, zmu, zsig, zmin, zmax)

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
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens//2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hiddens//2,
                                 out_channels=num_hiddens,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=2, padding=1)
        self._conv_4 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs):
        x = self._conv_1(inputs)
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
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Decoder, self).__init__()
        
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hiddens,
                                 kernel_size=3, 
                                 stride=1, padding=1)
        
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)
        
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hiddens, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=4, 
                                                stride=2, padding=1)
        
        self._conv_trans_3 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=num_channels,
                                                kernel_size=4, 
                                                stride=2, padding=1)

    def forward(self, inputs):
        #import pdb; pdb.set_trace()
        x = self._conv_1(inputs)
        
        x = self._residual_stack(x)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        
        x = self._conv_trans_2(x)
        x = F.relu(x)

        x = self._conv_trans_3(x)
        return x

# %%
class Model(nn.Module):
    def __init__(self, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost, decay=0):
        super(Model, self).__init__()
        
        self._encoder = Encoder(num_channels, num_hiddens,
                                num_residual_layers, 
                                num_residual_hiddens)
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
                                num_residual_hiddens)

    def forward(self, x):
        #import pdb; pdb.set_trace()
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        x_recon = self._decoder(quantized)

        return loss, x_recon, perplexity

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
    _err = None
    try:
        with open('%s/checkpoint.txt'%(_prefix), 'r') as f:
            _istart = int(f.readline())
        fname = '%s/checkpoint.%d.pytorch'%(_prefix, _istart)
        print ('Checkpoint:', fname)
        _model = torch.load(fname)
        _model.eval()
    except:
        print ("Error:", sys.exc_info()[0])
        print ("No restart info")
        pass
    
    return (_istart, _model)

# %%
def save_checkpoint(DIR, prefix, model, err, epoch):
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
    with open('%s/checkpoint.txt'%(_prefix), 'w+') as f:
        f.write(str(epoch))
    print ("Saved checkpoint: %s"%(fname))

def main():
    global num_channels
    global device

    # %%
    # Main start
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', help='exp', default='m1')
    parser.add_argument('-n', '--num_training_updates', help='num_training_updates (default: %(default)s)', type=int, default=10_000)
    parser.add_argument('-e', '--embedding_dim', help='embedding_dim (default: %(default)s)', type=int, default=64)
    parser.add_argument('-H', '--num_hiddens', help='num_hidden (default: %(default)s)', type=int, default=128)
    parser.add_argument('-b', '--batch_size', help='batch_size (default: %(default)s)', type=int, default=256)
    parser.add_argument('-d', '--device_id', help='device_id (default: %(default)s)', type=int, default=0)
    parser.add_argument('--wdir', help='working directory (default: current)', default=os.getcwd())
    parser.add_argument('--datadir', help='data directory (default: %(default)s)', default='data')
    parser.add_argument('--timesteps', help='timesteps', nargs='+', type=int)
    parser.add_argument('--average_interval', help='average_interval (default: %(default)s)', type=int, default=1_000)
    parser.add_argument('--log_interval', help='log_interval (default: %(default)s)', type=int, default=1_000)
    parser.add_argument('--checkpoint_interval', help='checkpoint_interval (default: %(default)s)', type=int, default=10_000)
    parser.add_argument('--nompi', help='nompi', action='store_true')
    args = parser.parse_args()

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
    print (fmt)
    #logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')
    logging.basicConfig(level=logging.DEBUG, format=fmt)
    logging.info ('EXP: %s' % args.exp)
    logging.info ('embedding_dim: %d' % args.embedding_dim)
    logging.info ('DIR: %s' % args.wdir)

    # %%
    DIR=args.wdir
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info ('device: %s' % device)

    # %%
    num_channels = 16

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
    num_residual_hiddens = 32
    num_residual_layers = 2

    embedding_dim = args.embedding_dim
    num_embeddings = 512

    commitment_cost = 0.25
    decay = 0.99
    learning_rate = 1e-3

    prefix='xgc-%s-batch%d-edim%d-nhidden%d'%(args.exp, args.batch_size, args.embedding_dim, args.num_hiddens)
    logging.info ('prefix: %s' % prefix)

    # %%
    with ad2.open('%s/xgc.f0.mesh.bp'%(args.datadir), 'r') as f:
        f0_dvp = f.read('f0_dvp')
        f0_nmu = f.read('f0_nmu')
        f0_nvp = f.read('f0_nvp')
        f0_T_ev = f.read('f0_T_ev')
        f0_grid_vol_vonly = f.read('f0_grid_vol_vonly')
        nnodes = f.read('n_n')

    #f0_filenames = (13_000, 10_000)
    #f0_filenames = (13_000, 13_100, 13_200, 13_300, 13_400)
    #f0_filenames = ('data/xgc.f0.13000.bp', 'data/xgc.f0.13100.bp', 'data/xgc.f0.13200.bp', 'data/xgc.f0.13300.bp', 'data/xgc.f0.13400.bp')
    f0_filenames = args.timesteps
    if len(f0_filenames) == 0:
        print ()
    f0_filenames = np.array_split(np.array(f0_filenames), size)[rank]
    f0_data_list = list()
    logging.info (f'Data dir: {args.datadir}')
    for fname in f0_filenames:
        logging.info (f'Reading: {fname}')
        f0_data_list.append(read_f0(fname, dir=args.datadir, full=True, inode=0, nnodes=nnodes%batch_size))

    lst = list(zip(*f0_data_list))
    Zif = np.r_[(lst[0])]
    zmu = np.r_[(lst[1])]
    zsig = np.r_[(lst[2])]
    zmin = np.r_[(lst[3])]
    zmax = np.r_[(lst[4])]
    ## z-score normalization
    #Zif = (Zif - zmu[:,np.newaxis,np.newaxis])/zsig[:,np.newaxis,np.newaxis]
    ## min-max normalization
    #Zif = (Zif - np.min(Zif))/(np.max(Zif)-np.min(Zif))
    Zif = (Zif - zmin[:,np.newaxis,np.newaxis])/(zmax-zmin)[:,np.newaxis,np.newaxis]

    print ('Zif bytes,shape:', Zif.size * Zif.itemsize, Zif.shape, zmu.shape, zsig.shape)
    print ('Minimum training epoch:', Zif.shape[0]/batch_size)

    lx = list()
    ly = list()
    for i in range(0,len(Zif)-num_channels,num_channels):
        X = Zif[i:i+num_channels,:,:]
        mu = zmu[i:i+num_channels]
        sig = zsig[i:i+num_channels]
        N = X
        ## z-score normalization
        #N = (X - mu[:,np.newaxis,np.newaxis])/sig[:,np.newaxis,np.newaxis]
        lx.append(N)
        ly.append(np.array([i,], dtype=int))

    data_variance = np.var(lx, dtype=np.float64)
    print ('data_variance', data_variance)

    # %% 
    # Loadding
    X_train, X_test, y_train, y_test = train_test_split(lx, ly, test_size=0.10, random_state=42)
    training_data = torch.utils.data.TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train))
    validation_data = torch.utils.data.TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))    

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
    model = Model(num_hiddens, num_residual_layers, num_residual_hiddens,
                num_embeddings, embedding_dim, 
                commitment_cost, decay).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)

    model.train()
    train_res_recon_error = []
    train_res_perplexity = []
    istart = 1

    # %%
    # Load checkpoint
    _istart, _model = load_checkpoint(DIR, prefix, model)
    if _model is not None:
        istart = _istart + 1
        model = _model
    print (istart)

    # %%
    num_training_updates=args.num_training_updates
    logging.info ('Training: %d' % num_training_updates)
    model.train()
    t0 = time.time()
    for i in xrange(istart, istart+num_training_updates):
        (data, lb) = next(iter(training_loader))
        data = data.to(device)
        optimizer.zero_grad() # clear previous gradients
        
        vq_loss, data_recon, perplexity = model(data)
        #recon_error = torch.mean((data_recon - data)**2) / data_variance
        recon_error = torch.mean((data_recon - data)**2) 
        loss = recon_error + vq_loss
        #print (recon_error, vq_loss)
    #     with torch.no_grad():
    #         den_err, u_para_err, T_perp_err, T_para_err = physics_loss(data, lb, zmu, zsig, data_recon)
    #         ds = np.mean(data_recon.cpu().data.numpy()**2)
    #         print (recon_error.data, vq_loss.data, den_err, u_para_err, T_perp_err, T_para_err, ds)

        # Here is to add physics information:
        # # loss += den_err/ds * torch.sum(data_recon)
    #     loss += u_para_err/ds * torch.mean(data_recon**2)
        # # loss += T_perp_err/ds * torch.sum(data_recon)
        # # loss += T_para_err/ds * torch.sum(data_recon)

        loss.backward()
        if i % args.average_interval == 0:
            ## Gradient averaging
            logging.info('iteration %d: gradient averaging' % (i))
            average_gradients(model)
        optimizer.step()
        #print ('AFTER', model._vq_vae._embedding.weight.data.numpy().sum())
        
        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())

        if i % args.log_interval == 0:
            # logging.info('%d iterations' % (i))
            # logging.info('recon_error: %.3g' % np.mean(train_res_recon_error[-1000:]))
            # logging.info('perplexity: %.3g' % np.mean(train_res_perplexity[-1000:]))
            # logging.info('time: %.3f' % (time.time()-t0))
            # logging.info('last recon_error, vq_loss: %.3g %.3g'%(recon_error.data.item(), vq_loss.data.item()))
            logging.info(f'{i} Avg: {np.mean(train_res_recon_error[-args.log_interval:])} {np.mean(train_res_perplexity[-args.log_interval:])}')
            logging.info(f'{i} Loss: {recon_error.item()} {vq_loss.data.item()} {perplexity.item()} {len(training_loader.dataset)} {len(data)}')
            # logging.info('')
        
        if (i % args.checkpoint_interval == 0) and (rank == 0):
            save_checkpoint(DIR, prefix, model, train_res_recon_error, i)
    istart=istart+num_training_updates

    # %%
    model.eval()
    (valid_originals, valid_labels) = next(iter(validation_loader))
    valid_originals = valid_originals.to(device)

    x = model._encoder(valid_originals)
    logging.info ('Original: %s' % (valid_originals.shape,))
    logging.info ('Encoding: %s' % (x.shape,))
    logging.info ('compression ratio: %.3f'%(x.detach().cpu().numpy().size/valid_originals.cpu().numpy().size))

if __name__ == "__main__":
    main()
    #with profiler.profile() as prof:
    # from pytracing import TraceProfiler
    # tp = TraceProfiler(output=open('trace.out', 'wb'))
    # with tp.traced():
    #     main()
    #prof.export_chrome_trace("trace.json")
    
