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
#torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.autograd.profiler as profiler
from torch.backends import cudnn

import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid, save_image
from torch.utils.tensorboard import SummaryWriter

import xgc4py
from tqdm import tqdm

import sys
from datetime import datetime
from pathlib import Path

import hashlib
from skimage.transform import resize

from models import *

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

def info(*args, logtype='info', sep=' '):
    getattr(logging, logtype)(sep.join(map(str, args)))

def debug(*args, logtype='debug', sep=' '):
    getattr(logging, logtype)(sep.join(map(str, args)))

# %%
def relative_linf_error(x, y):
    """
    relative L-inf error: max(|x_i - y_i|)/max(|x_i|)
    """
    assert(x.shape == y.shape)
    linf = np.max(np.abs(x-y))
    maxv = np.max(np.abs(x))
    return (linf/maxv)
 
def rmse_error(x, y):
    """
    root mean square error: square-root of sum of all (x_i-y_i)**2
    """
    assert(x.shape == y.shape)
    mse = np.mean((x-y)**2)
    return np.sqrt(mse)

# %%
def conv_hash(X):
    d1, d2 = X.shape
    digest_size = hashlib.sha3_512().digest_size

    ds = list()
    msg = X.tobytes()
    for i in range(d1*d2//digest_size+1):
        f = hashlib.sha3_512()
        f.update(msg)
        msg = f.digest()
        ds.extend([float(d)/255.0 for d in msg])
    ds = ds[:d1*d2]
    return np.array(ds).reshape([d1,d2])

def conv_hash_torch(X):
        Y = torch.zeros_like(X)
        b, c, d1, d2 = Y.shape
        for i in range(b):
            for j in range(c):
                _X = X[i,j,:].cpu().numpy()
                _Y = conv_hash(_X)
                Y[i,j,:] = torch.tensor(_Y).to(X.device)
        return Y

def parse_rangestr(rangestr):
    _rangestr = rangestr.replace(' ', '')
    # Convert String ranges to list 
    # Using sum() + list comprehension + enumerate() + split() 
    res = sum(((list(range(*[int(b) + c  
            for c, b in enumerate(a.split('-'))])) 
            if '-' in a else [int(a)]) for a in _rangestr.split(',')), [])
    return res

# %%
def init(counter):
    """
    Init processes for ProcessPoolExecutor
    """
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
    """
    Process work for ProcessPoolExecutor
    """
    global xgcexp
    global args
    global Z0, zmu, zsig, zmin, zmax

    # hard coding inode: 100_000_000+iphi + inode
    nnodes = 100_000_000
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

    ## Unnormalizing
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

def physics_loss_con(data, lb, data_recon, executor, progress=False):
    """
    Calculate phyiscs loss in parallel by using ProcessPoolExecutor
    """
    global args
    global xgcexp
    global Z0, zmu, zsig, zmin, zmax

    batch_size, num_channels = data.shape[:2]
    # hard coding inode: 100_000_000+iphi + inode
    # nnodes = len(Z0)//xgcexp.nphi
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
            futures.append(executor.submit(dowork, Xbar[i,:num_channels,:nvp0,:nvp1], int(lb[i])-args.inode, num_channels))

        for f in tqdm(futures, disable=not progress):
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

        return (den_err, u_para_err, T_perp_err, T_para_err)

# %%
def physics_loss(data, lb, data_recon, progress=False):
    """
    Calculate phyiscs loss
    """
    global device
    global xgcexp
    global args
    global Z0, zmu, zsig, zmin, zmax

    batch_size, num_channels = data.shape[:2]
    # hard coding inode: 100_000_000+iphi + inode
    nnodes = 100_000_000
    nvp0 = xgcexp.f0mesh.f0_nmu+1
    nvp1 = xgcexp.f0mesh.f0_nvp*2+1

    # Xbar = data_recon.cpu().data.numpy() ## shape: (nbatch, nchannel, nvp0, nvp1)
    Xbar = data_recon
    device = data_recon.device
    
    den_err = torch.tensor(0.).to(device)
    u_para_err = torch.tensor(0.).to(device)
    T_perp_err = torch.tensor(0.).to(device)
    T_para_err = torch.tensor(0.).to(device)
    
    for i in tqdm(range(batch_size), disable=not progress):
        if num_channels==1:
            inode = int(lb[i]) - args.inode
        else:
            inode = int(lb[i,0]) - args.inode
        mn = zmin[inode:inode+num_channels]
        mx = zmax[inode:inode+num_channels]

        #f0_f = data[i].cpu().data.numpy()
        f0_f = torch.from_numpy(Z0[inode:inode+num_channels,:nvp0,:nvp1]).to(device)
        #f0_f *= (mx[:,np.newaxis,np.newaxis]-mn[:,np.newaxis,np.newaxis])
        #f0_f += mn[:,np.newaxis,np.newaxis]
        den0, u_para0, T_perp0, T_para0, _, _ = \
            xgcexp.f0_diag_torch(f0_inode1=inode%nnodes, ndata=num_channels, isp=1, f0_f=f0_f, progress=False)

        ## Re-scale first
        f0_f1 = Xbar[i,:num_channels,:nvp0,:nvp1]
        f0_f2 = f0_f1 * torch.from_numpy(mx[:,np.newaxis,np.newaxis]-mn[:,np.newaxis,np.newaxis]).to(device)
        f0_f = f0_f2 + torch.from_numpy(mn[:,np.newaxis,np.newaxis]).to(device)
        den1, u_para1, T_perp1, T_para1, _, _ = \
            xgcexp.f0_diag_torch(f0_inode1=inode%nnodes, ndata=num_channels, isp=1, f0_f=f0_f, progress=False)
        
        den_err += torch.mean((den0-den1)**2)/torch.var(den0)
        u_para_err += torch.mean((u_para0-u_para1)**2)/torch.var(u_para0)
        T_perp_err += torch.mean((T_perp0-T_perp1)**2)/torch.var(T_perp0)
        T_para_err += torch.mean((T_para0-T_para1)**2)/torch.var(T_para0)

    return (den_err, u_para_err, T_perp_err, T_para_err)

# %%
def read_f0_nodes(istep, inodes, expdir=None, iphi=None, nextnode_arr=None, rescale=None):
    """
    Read XGC f0 data
    """
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
    with ad2.open(fname, 'r') as f:
        nstep, nsize = adios2_get_shape(f, 'i_f')
        ndim = len(nsize)
        nphi = nsize[0] if iphi is None else 1
        iphi = 0 if iphi is None else iphi
        nnodes = nsize[2]
        nmu = nsize[1]
        nvp = nsize[3]
        start = (iphi,0,0,0)
        count = (nphi,nmu,nnodes,nvp)
        logging.info (f"Reading: {fname} {start} {count}")
        i_f = f.read('i_f', start=start, count=count).astype('float64')

    # if i_f.shape[3] == 31:
    #     i_f = np.append(i_f, i_f[...,30:31], axis=3)
    #     # e_f = np.append(e_f, e_f[...,30:31], axis=3)
    # if i_f.shape[3] == 39:
    #     i_f = np.append(i_f, i_f[...,38:39], axis=3)
    #     i_f = np.append(i_f, i_f[:,38:39,:,:], axis=1)

    i_f = np.moveaxis(i_f, 1, 2)

    if nextnode_arr is not None:
        logging.info (f"Reading: untwist is on")
        f_new = np.zeros_like(i_f)
        for i in range(len(f_new)):
            od = nextnode_arr[i+iphi,:]
            f_new[i,:,:,:] = i_f[i+iphi,od,:,:]
        i_f = f_new

    da_list = list()
    lb_list = list()
    ## i_f is already subset
    ## (2021/02) group by inter-planes first
    for j in inodes:
        for i in range(nphi):
            da_list.append(i_f[i,j,:,:])
            k = j
            if nextnode_arr is not None:
                k = nextnode_arr[i+iphi,j]
            lb_list.append((istep,i+iphi,k))
    
    Z0 = np.array(da_list)
    zlb = np.array(lb_list)

    if rescale is not None:
        log ("Input rescale:", rescale)
        nn, nx, ny = Z0.shape
        _Z0 = np.zeros((nn,nx*rescale,ny*rescale), dtype=Z0.dtype)
    
        for i in range(len(_Z0)):
            X = Z0[i,:]
            img = Image.fromarray(X)
            img = img.resize((_Z0.shape[-2],_Z0.shape[-1]))
            _Z0[i,:] = np.array(img)
        Z0 = _Z0

    zmu = np.mean(Z0, axis=(1,2))
    zsig = np.std(Z0, axis=(1,2))
    zmin = np.min(Z0, axis=(1,2))
    zmax = np.max(Z0, axis=(1,2))
    Zif = (Z0 - zmin[:,np.newaxis,np.newaxis])/(zmax-zmin)[:,np.newaxis,np.newaxis]

    return (Z0, Zif, zmu, zsig, zmin, zmax, zlb)

# %%
def read_f0(istep, expdir=None, iphi=None, inode=0, nnodes=None, average=False, randomread=0.0, nchunk=16, fieldline=False):
    """
    Read XGC f0 data
    """
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
        cc = [x for x in list(nx.connected_components(G)) if len(x) >= 16]

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
            for i in path[:len(path)-len(path)%16]:
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
            _nnodes = nsize[2]-inode if nnodes is None else nnodes
            nmu = nsize[1]
            nvp = nsize[3]
            start = (iphi,0,inode,0)
            count = (nphi,nmu,_nnodes,nvp)
            logging.info (f"Reading: {start} {count}")
            i_f = f.read('i_f', start=start, count=count).astype('float64')
            #e_f = f.read('e_f')
        li = list(range(inode, inode+_nnodes))
        lb = np.array(li, dtype=np.int32)

    # if i_f.shape[3] == 31:
    #     i_f = np.append(i_f, i_f[...,30:31], axis=3)
    #     # e_f = np.append(e_f, e_f[...,30:31], axis=3)
    # if i_f.shape[3] == 39:
    #     i_f = np.append(i_f, i_f[...,38:39], axis=3)
    #     i_f = np.append(i_f, i_f[:,38:39,:,:], axis=1)

    Z0 = np.moveaxis(i_f, 1, 2)

    if average:
        Z0 = np.mean(Z0, axis=0)
        _lb = list()
        for k in lb:
            _lb.append((istep, -1, k))
        zlb = np.array(_lb)
    else:
        Z0 = Z0.reshape((-1,Z0.shape[2],Z0.shape[3]))
        _lb = list()
        for i in range(nphi):
            for k in lb:
                _lb.append((istep, i, k))
        zlb = np.array(_lb)
    
    #zlb = np.concatenate(li)
    zmu = np.mean(Z0, axis=(1,2))
    zsig = np.std(Z0, axis=(1,2))
    zmin = np.min(Z0, axis=(1,2))
    zmax = np.max(Z0, axis=(1,2))
    Zif = (Z0 - zmin[:,np.newaxis,np.newaxis])/(zmax-zmin)[:,np.newaxis,np.newaxis]

    return (Z0, Zif, zmu, zsig, zmin, zmax, zlb)

# %%
def read_nstx(expdir=None, offset=159065, nframes=16384, average=False, randomread=0.0, nchunk=16, gaussian=False):
    """
    Read NSTX data
    """
    start = offset
    count = nframes - nframes%nchunk
    logging.info (f"Reading: nstx_data_ornl_demo_v2.bp {start} {count}")
    fname = os.path.join(expdir, 'nstx_data_ornl_demo_v2.bp')
    with ad2.open(fname,'r') as f:
        gpiData = f.read('gpiData')[start:start+count,:,:]
    
    Z0 = gpiData.astype('float32')
    zlb = np.array(range(start, start+count), dtype=np.int32)
    if gaussian:
        from scipy.ndimage import gaussian_filter
        for i in range(len(Z0)):
            Z0[i,:] = gaussian_filter(Z0[i,:], sigma=2)

    #zlb = np.concatenate(li)
    zmu = np.mean(Z0, axis=(1,2))
    zsig = np.std(Z0, axis=(1,2))
    zmin = np.min(Z0, axis=(1,2))
    zmax = np.max(Z0, axis=(1,2))
    Zif = (Z0 - zmin[:,np.newaxis,np.newaxis])/(zmax-zmin)[:,np.newaxis,np.newaxis]

    return (Z0, Zif, zmu, zsig, zmin, zmax, zlb)

# %%
def average_gradients(model):
    """
    Gradient averaging
    """
    try:
        MPI
    except NameError:
        return
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
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
def recon(model, Xif, zmin, zmax, zlb, num_channels=16, dmodel=None, modelname='vqvae', return_encode=False, conditional=False):
    """
    Reconstructing data based on a trained model
    """
    global da
    mode = model.training
    model.eval()
    with torch.no_grad():
        lx = list()
        ly = list()
        for i in range(0,len(Xif),num_channels):
            X = Xif[i:i+num_channels,:,:]
            N = X.astype(np.float32)
            lx.append(N)
            # ly.append(zlb[i])

        lz = list()
        nbatch = 1
        encode_list = list()
        for i in range(0, len(lx), nbatch):
            valid_originals = torch.tensor(lx[i:i+nbatch]).to(device)
            nbatch, nchannel, dim1, dim2 = valid_originals.shape
            if modelname in ('vae','cvae'):
                _da = None
                if args.model == 'cvae':
                    _da = da[zlb[i:i+nbatch,-1],]

                valid_reconstructions, mu, logvar = model(valid_originals, _da)
                valid_reconstructions = valid_reconstructions.view(-1, model.nc, model.ny, model.nx)
                lz.append((valid_reconstructions).cpu().data.numpy())
            elif modelname in ('ae','cae','ae2d'):
                _da = None
                if args.model == 'cae':
                    _da = da[zlb[i:i+nbatch,-1],]

                valid_encode = model.encode(valid_originals, _da)
                if args.conditional:
                    x = valid_originals
                    _x = torch.cat((x,x,x), axis=1)
                    feature = model.feature_extractor(_x)
                    flat = torch.mean(feature, dim=(2,3))
                    y = model.feature_encoder(flat)
                    valid_encode = torch.cat((valid_encode,y), axis=1)

                valid_reconstructions = model.decode(valid_encode, _da)
                
                lz.append((valid_reconstructions).cpu().data.numpy())
                encode_list.append(valid_encode)
            elif modelname == 'ae-vqvae':
                valid_encode = model.encode(valid_originals)
                valid_reconstructions = model.decode(valid_encode)

                model2 = dmodel
                vq_loss, data_recon, perplexity, dloss = model2(valid_reconstructions)

                lz.append((data_recon).cpu().data.numpy())
                encode_list.append(valid_encode)
            else:
                if model._grid is not None:
                    x = valid_originals
                    x = torch.cat([x, model._grid.repeat(nbatch,1,1,1)], dim=1)
                    x = x.permute(0, 2, 3, 1)
                    x = model.fc0(x)
                    x = x.permute(0, 3, 1, 2)
                    valid_originals = x

                _da = None
                if modelname == 'cvqvae':
                    _da = da[zlb[i:i+nbatch,-1],]

                vq_encoded = model._encoder(valid_originals, _da)
                vq_output_eval = model._pre_vq_conv(vq_encoded)
                _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
                if conditional:
                    x = valid_originals
                    x = torch.cat((x,x,x), axis=1)
                    feature = model.feature_extractor(x)
                    cond = F.avg_pool2d(feature, kernel_size=3, stride=2, padding=1)
                    # feature = model.feature_extractor(valid_originals)
                    # cond = F.avg_pool1d(feature.view(nbatch,nchannel,1090), kernel_size=11)
                    # p1d = (0, 1)
                    # cond = F.pad(cond, p1d, "constant", 0)
                    # cond = torch.reshape(cond, (nbatch, 4, 5, 5))
                    valid_quantize = torch.cat((valid_quantize, cond), dim=1)
                valid_reconstructions = model._decoder(valid_quantize, _da)
                encode_list.append(valid_quantize)
                #print (vq_encoded.shape, vq_output_eval.shape, valid_quantize.shape, valid_reconstructions.shape)

                if model._grid is not None:
                    x = valid_reconstructions
                    x = x.permute(0, 2, 3, 1)
                    x = model.fc1(x)
                    x = F.relu(x)
                    x = model.fc2(x)
                    x = x.permute(0, 3, 1, 2)
                    valid_reconstructions = x

                #print (valid_originals.sum().item(), valid_reconstructions.shape, valid_reconstructions.sum().item())

                drecon = 0
                if dmodel is not None:
                    dx = (valid_originals-valid_reconstructions).view(-1, nchannel*dim1*dim2)
                    drecon = dmodel(dx).view(-1, nchannel, dim1, dim2)
                lz.append((valid_reconstructions+drecon).cpu().data.numpy())

        Xbar = np.array(lz).reshape([-1,dim1,dim2])
        Xenc = None
        if return_encode:
            Xenc = torch.cat(encode_list)
            Xenc = Xenc.detach().cpu().numpy()

        ## Normalize
        xmin = np.min(Xbar, axis=(1,2))
        xmax = np.max(Xbar, axis=(1,2))
        Xbar = (Xbar-xmin[:,np.newaxis,np.newaxis])/(xmax-xmin)[:,np.newaxis,np.newaxis]

        ## Un-normalize
        X0 = Xbar*((zmax-zmin)[:,np.newaxis,np.newaxis])+zmin[:,np.newaxis,np.newaxis]
    model.train(mode)
    
    if not return_encode:
        return (X0, Xbar, np.mean(X0, axis=(1,2)))
    else:
        return (X0, Xbar, np.mean(X0, axis=(1,2)), Xenc)

def estimate_error(model, Xif, Zif, zmin, zmax, zlb, num_channels, modelname, fname=None, conditional=False, nosort=False):
    """
    Error calculation
    """
    X0, Xbar, xmu = recon(model, Xif, zmin, zmax, zlb, num_channels=num_channels, modelname=modelname, conditional=conditional)

    rmse_list = list()
    abs_list = list()
    for i in range(len(Xbar)):
        Z = Zif[i,:,:]
        X = Xbar[i,:,:]
        ## RMSE
        rmse = np.sqrt(np.sum((Z-X)**2)/Z.size)
        rmse_list.append(rmse)
        ## ABS error
        abserr = np.max(np.abs(Z-X))
        abs_list.append(abserr)
    
    if fname is not None:
        od = np.argsort(abs_list)[::-1]
        if nosort:
            od = np.arange(len(abs_list))
        idx = od[:8]
        dat = torch.cat((torch.tensor(Zif[idx,np.newaxis,:,:]),torch.tensor(Xbar[idx,np.newaxis,:,:])))
        #grid_img = make_grid(dat)
        #plt.imshow(grid_img.permute(1, 2, 0))
        save_image(dat, fname)

    # max_rmse_list = [max(rmse_list[i: i+num_channels]) for i in xrange(0, len(rmse_list), num_channels)]
    # max_abs_list = [max(abs_list[i: i+num_channels]) for i in xrange(0, len(abs_list), num_channels)]
    # max_abs_list = np.array(max_abs_list)
    # max_abs_list = max_abs_list/sum(max_abs_list)

    return (rmse_list, abs_list)

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
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self._block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        )

    def forward(self, x):
        return self._block(x)

class ConvTBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvTBlock, self).__init__()
        self._block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            #nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self._block(x)

# %%
class Encoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(Encoder, self).__init__()

        log ("in_channels, num_hiddens:", in_channels, num_hiddens)

        self._conv_0 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=in_channels,
                                 kernel_size=4,
                                 stride=2, padding=1)
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

        # self._block = nn.Sequential(
        #     ConvBlock(in_channels, num_hiddens//2),
        #     ConvBlock(num_hiddens//2, num_hiddens),
        #     ConvBlock(num_hiddens, num_hiddens),
        # )

        # kernel_size=3, stride=1, padding=1
        # kernel_size=2, stride=2, padding=0
        self._conv_4 = nn.Conv2d(in_channels=num_hiddens,
                                 out_channels=num_hiddens,
                                 kernel_size=3, stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hiddens,
                                             num_hiddens=num_hiddens,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hiddens=num_residual_hiddens)

    def forward(self, inputs, da=None):
        # (2020/11) Testing with resize
        x = inputs
        if da is not None:
            _d = torch.zeros_like(x)
            _a = torch.zeros_like(x)
            _d[:,:,:,:] = da[:,0,np.newaxis,np.newaxis,np.newaxis]
            _a[:,:,:,:] = da[:,1,np.newaxis,np.newaxis,np.newaxis]
            _da = torch.cat((_d, _a), dim=1)
            x = torch.cat((x, _d, _a), dim=1)
        # print ('ENC #1:', x.shape)
        # if self._rescale is not None:
        #     x = F.interpolate(inputs, size=x.shape[-1]*self._rescale)
        #     x = self._conv_0(x)
        #     x = F.relu(x)

        x = self._conv_1(x)
        x = F.relu(x)
        # print ('ENC #2:', x.shape)
        
        x = self._conv_2(x)
        x = F.relu(x)
        # print ('ENC #3:', x.shape)
        
        x = self._conv_3(x)
        x = F.relu(x)
        # x = self._block(x)
        # print ('ENC #4:', x.shape)

        x = self._conv_4(x)
        x = self._residual_stack(x)
        # print ('ENC #5:', x.shape)
        return x

# %%
class Decoder(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens, num_channels, padding=[1,1,1], layer_sizes=[]):
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
                                                kernel_size=3, stride=2, padding=1, output_padding=padding[0])
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=num_hiddens//2,
                                                kernel_size=3, stride=2, padding=1, output_padding=padding[1])
        self._conv_trans_3 = nn.ConvTranspose2d(in_channels=num_hiddens//2, 
                                                out_channels=num_channels,
                                                kernel_size=3, stride=2, padding=1, output_padding=padding[2])

        self.MLP = None
        if len(layer_sizes)>1:
            self.MLP = nn.Sequential()
            for i, (in_size, out_size) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
                self.MLP.add_module(name="L{:d}".format(i), module=nn.Linear(in_size, out_size))
                self.MLP.add_module(name="A{:d}".format(i), module=nn.ReLU())

        # # (2021/03)
        # self._block = nn.Sequential(
        #     ConvTBlock(num_hiddens, num_hiddens//2, 4, 2, 1),
        #     ConvTBlock(num_hiddens//2, num_hiddens//4, 4, 2, 1),
        #     ConvTBlock(num_hiddens//4, num_channels, 4, 2, 1)
        # )

        # # (2021/03)
        # self._block = nn.Sequential(
        #     ## 128->64, 5->10
        #     ConvTBlock(128, 64, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(64, 64, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(64, 64, kernel_size=4, stride=1, padding=1),
        #     ## 64->32, 10->20
        #     ConvTBlock(64, 32, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(32, 32, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(32, 32, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(32, 32, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(32, 32, kernel_size=3, stride=1, padding=0),
        #     ## 32->1, 20->40
        #     ConvTBlock(32, 16, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(16, 16, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(16, 16, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(16, 16, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(16, 4, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(4, 4, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(4, 4, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(4, 4, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(4, 1, kernel_size=3, stride=1, padding=0),
        #     ConvTBlock(1, 1, kernel_size=3, stride=1, padding=0),
        # )

    #0: torch.Size([1, 16, 5, 5])
    #1: torch.Size([1, 128, 5, 5])
    #2: torch.Size([1, 128, 5, 5])
    #3: torch.Size([1, 64, 10, 10])
    #4: torch.Size([1, 64, 20, 20])
    #5: torch.Size([1, 1, 40, 40])
    def forward(self, inputs, da=None):
        x = inputs
        if da is not None:
            nb, nc, nx, ny = x.shape
            _d = torch.zeros_like(x)
            _a = torch.zeros_like(x)
            _d[:,:,:,:] = da[:,0,np.newaxis,np.newaxis,np.newaxis]
            _a[:,:,:,:] = da[:,1,np.newaxis,np.newaxis,np.newaxis]
            x = torch.cat((x, _d[:,-1:,:,:], _a[:,-1:,:,:]), dim=1)
        # print ('DEC #1:', x.shape)

        x = self._conv_1(x)
        x = self._residual_stack(x)
        # print ('DEC #2:', x.shape)
        
        x = self._conv_trans_1(x)
        x = F.relu(x)
        # print ('DEC #3:', x.shape)
        
        x = self._conv_trans_2(x)
        x = F.relu(x)
        # print ('DEC #4:', x.shape)
        
        x = self._conv_trans_3(x)
        # x = torch.sigmoid(x)
        # x = self._block(x)
        # print ('DEC #5:', x.shape)

        if self.MLP is not None:
            nb, nc, nx, ny = x.shape
            x = self.MLP(x.view(-1, nc*nx*ny))
            x = x.view(-1, nc, nx, ny)

        return x

# %%
class Model(nn.Module):
    def __init__(self, num_channels, num_hiddens, num_residual_layers, num_residual_hiddens, 
                 num_embeddings, embedding_dim, commitment_cost, decay=0, rescale=None, learndiff=False, 
                 input_shape=None, shaconv=False, grid=None, conditional=False, decoder_padding=[1,1,1], 
                 da_conditional=False, decoder_layer_sizes=[]):
        super(Model, self).__init__()
        
        self._grid = grid
        self.width = num_channels
        self.rescale = rescale
        self.conditional = conditional
        log ("Model rescale:", self.rescale)
        
        if grid is not None:
            self.width = 32
            self.fc0 = nn.Linear(3, self.width)
            self.fc1 = nn.Linear(self.width, 128)
            self.fc2 = nn.Linear(128, num_channels)

        self.ncond = 0
        if da_conditional:
            self.ncond = 2

        self._encoder = Encoder(self.width+self.ncond, num_hiddens,
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
        _embedding_dim = embedding_dim
        if self.conditional:
            _embedding_dim = embedding_dim + 256
        self._decoder = Decoder(_embedding_dim+self.ncond,
                                num_hiddens, 
                                num_residual_layers, 
                                num_residual_hiddens, self.width, padding=decoder_padding, 
                                layer_sizes=decoder_layer_sizes)

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
        
        self._shaconv = shaconv

        if self.conditional:
            # XGC-VGG19
            modelfile = 'xgc-vgg19-ch3-N1000.torch'
            self.feature_extractor = XGCFeatureExtractor(modelfile)
            # VGG
            # self.feature_extractor = FeatureExtractor()
            self.feature_extractor = self.feature_extractor.to(device)
            self.feature_extractor.eval()

    def forward(self, x, da=None):
        nbatch, nchannel, dim1, dim2 = x.shape
        if self._grid is not None:
            x = torch.cat([x, self._grid.repeat(nbatch,1,1,1)], dim=1)
            x = x.permute(0, 2, 3, 1)
            x = self.fc0(x)
            x = x.permute(0, 3, 1, 2)

        ## sha conv
        if self._shaconv:
            x = conv_hash_torch(x)
        
        if self.rescale is not None:
            b, c, nx, ny = x.shape
            x = F.interpolate(x, size=(nx*self.rescale, ny*self.rescale))
            print ('scale-up', x.shape)

        # print ('#0:', x.shape)
        z = self._encoder(x, da)
        # print ('#1:', z.shape)
        z = self._pre_vq_conv(z)
        # print ('#2:', z.shape)
        loss, quantized, perplexity, _ = self._vq_vae(z)
        # print ('#3:', quantized.shape)
        if self.conditional:
            _x = torch.cat((x,x,x), axis=1)
            feature = self.feature_extractor(_x)
            cond = F.avg_pool2d(feature, kernel_size=3, stride=2, padding=1)
            # cond = F.avg_pool1d(feature.view(nbatch,nchannel,1090), kernel_size=11)
            # p1d = (0, 1)
            # cond = F.pad(cond, p1d, "constant", 0)
            # cond = torch.reshape(cond, (nbatch, 4, 5, 5))
            quantized = torch.cat((quantized, cond), dim=1)

        x_recon = self._decoder(quantized, da)
        # print ('#4:', x_recon.shape)

        if self.rescale is not None:
            x_recon = F.interpolate(x_recon, size=(nx, ny))
            print ('scale-down', x_recon.shape)

        if self._grid is not None:
            x = x_recon
            x = x.permute(0, 2, 3, 1)
            x = self.fc1(x)
            x = F.relu(x)
            x = self.fc2(x)
            x = x.permute(0, 3, 1, 2)
            x_recon = x.contiguous()

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
        model.load_state_dict(_model.state_dict())
        _model = model
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

class ResidualLinear(nn.Module):
    """
    in_channels, num_hiddens, num_residual_hiddens: in, out, hidden
    """
    def __init__(self, in_channels, num_hiddens, num_residual_hiddens):
        super(ResidualLinear, self).__init__()
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Linear(in_channels, num_residual_hiddens),
            nn.ReLU(True),
            nn.Linear(num_residual_hiddens, num_hiddens)
        )
    
    def forward(self, x):
        return x + self._block(x)

class ResidualLinearStack(nn.Module):
    def __init__(self, in_channels, num_hiddens, num_residual_layers, num_residual_hiddens):
        super(ResidualLinearStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([ResidualLinear(in_channels, num_hiddens, num_residual_hiddens)
                             for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)

"""
Credit: 
VAE: https://github.com/pytorch/examples/tree/master/vae
CVAE: https://github.com/timbmg/VAE-CVAE-MNIST/blob/master/models.py
"""
class VAE(nn.Module):
    ## (2021/07) Add conditional for CVAE. We use distance and angle for conditional values
    def __init__(self, nc, nx, ny, nh, nz, num_residual_hiddens, num_residual_layers, shaconv=False, da_conditional=False):
        super(VAE, self).__init__()

        self.nc = nc
        self.nx = nx
        self.ny = ny
        self.nh = nh
        self.nz = nz
        self.num_residual_hiddens = num_residual_hiddens
        self.num_residual_layers = num_residual_layers
        info ("VAE (nc,nx,ny,nh,nz):", nc, nx, ny, nh, nz)

        self.ncond = 0
        if da_conditional:
            self.ncond = 2
        self.fc1 = nn.Linear(self.nx*self.ny + self.ncond, self.nh)
        self.fc21 = nn.Linear(self.nh, self.nz)
        self.fc22 = nn.Linear(self.nh, self.nz)
        self.fc3 = nn.Linear(self.nz + self.ncond, self.nh)
        self.fc4 = nn.Linear(self.nh, self.nx*self.ny)

        self.rs = ResidualLinearStack(in_channels=self.nh,
                                    num_hiddens=self.nh,
                                    num_residual_layers=num_residual_layers,
                                    num_residual_hiddens=num_residual_hiddens)
        
        self._shaconv = shaconv

    def encode(self, x, da=None):
        if self.ncond > 0:
            assert da is not None
            x = torch.cat((x, da), dim=-1)

        x = F.relu(self.fc1(x))
        x = self.rs(x)
        return self.fc21(x), self.fc22(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, da=None):
        if self.ncond > 0:
            assert da is not None
            z = torch.cat((z, da), dim=-1)

        x = F.relu(self.fc3(z))
        x = self.rs(x)
        x = torch.sigmoid(self.fc4(x))
        # x = F.relu(self.fc4(x))
        # x = self.fc4(x)
        return x

    def forward(self, x, da=None):
        ## sha conv
        if self._shaconv:
            x = conv_hash_torch(x)

        mu, logvar = self.encode(x.view(-1, self.nx*self.ny), da)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, da), mu, logvar

    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(self, recon_x, x, mu, logvar):
        BCE = F.binary_cross_entropy(recon_x, x.view(-1, self.nx*self.ny), reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return BCE + KLD

# %%
"""
Credit: https://medium.com/pytorch/implementing-an-autoencoder-in-pytorch-19baa22647d1
"""
class AE(nn.Module):
    def __init__(self, input_dim, embedding_dim, conditional=False, da_conditional=False):
        super().__init__()

        self.conditional = conditional

        _embedding_dim = embedding_dim
        if self.conditional:
            # modelfile = 'xgc-vgg19-ch1-N1000.torch'
            # self.feature_extractor = XGCFeatureExtractor(modelfile)
            ## VGG pre-built model: 3-channel -> 256 feature vector
            self.feature_extractor = FeatureExtractor() 
            self.feature_extractor = self.feature_extractor.to(device)
            self.feature_extractor.eval()
            self.feature_encoder = nn.Sequential(
                nn.Linear(256, embedding_dim),
                nn.LeakyReLU(True),
                )
            _embedding_dim = embedding_dim*2

        self.ncond = 0
        if da_conditional:
            self.ncond = 2

        ## (2021/03) 400 = 16x5x5 to match with VQ-VAE
        self._encoder = nn.Sequential(
            nn.Linear(input_dim + self.ncond, embedding_dim),
            nn.LeakyReLU(True),
            nn.Linear(embedding_dim, embedding_dim),
            nn.LeakyReLU(True),
            )

        self._decoder = nn.Sequential(
            nn.Linear(_embedding_dim + self.ncond, embedding_dim),
            nn.LeakyReLU(True),
            nn.Linear(embedding_dim, input_dim),
            nn.LeakyReLU(True),
            )
    
    def encode(self, x, da=None):
        self.nbatch, self.nc, self.nx, self.ny = x.shape
        x = x.view(self.nbatch, -1)

        if self.ncond > 0:
            x = torch.cat((x, da), dim=-1)

        x = self._encoder(x)
        return x

    def decode(self, x, da=None):
        if self.ncond > 0:
            x = torch.cat((x, da), dim=-1)

        x = self._decoder(x)

        x = x.view(self.nbatch, self.nc, self.nx, self.ny)
        return x

    def forward(self, x, da=None):
        if self.conditional:
            ## x.shape: [32, 1, 39, 39]
            ## feature.shape: [32, 256, 9, 9]
            ## flat.shape: [32, 256]
            ## y.shape: [32, 16]
            _x = torch.cat((x,x,x), axis=1)
            feature = self.feature_extractor(_x)
            flat = torch.mean(feature, dim=(2,3))
            y = self.feature_encoder(flat)
            # cond = F.avg_pool2d(feature, kernel_size=3, stride=2, padding=1)
            # cond = F.avg_pool1d(feature.view(nbatch,nchannel,1090), kernel_size=11)
            # p1d = (0, 1)
            # cond = F.pad(cond, p1d, "constant", 0)
            # cond = torch.reshape(cond, (nbatch, 4, 5, 5))
            # quantized = torch.cat((quantized, cond), dim=1)

        x = self.encode(x, da)
        if self.conditional:
            x = torch.cat((x,y), axis=1)
        x = self.decode(x, da)

        # nbatch, nc, nx, ny = x.shape
        # x = x.view(nbatch, -1)

        # x = self.encoder(x)
        # x = self.decoder(x)

        # x = x.view(nbatch, nc, nx, ny)
        return x

# %%
"""
Credit: https://medium.com/@vaibhaw.vipul/building-autoencoder-in-pytorch-34052d1d280c
"""
class FlatView(nn.Module):
    def __init__(self):
        super(FlatView, self).__init__()

    def forward(self, x):
        return x.view(x.size()[0], -1)

class ConvView(nn.Module):
    def __init__(self, nc, nh, nw):
        super(ConvView, self).__init__()
        self.nc = nc
        self.nh = nh
        self.nw = nw

    def forward(self, x):
        return x.view(x.size()[0], self.nc, self.nh, self.nw)

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder,self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(True),
            FlatView(),
            nn.Linear(400, 400),
            nn.LeakyReLU(True),
            #nn.Conv2d(3, 6, kernel_size=5),
            #nn.ReLU(True),
            #nn.Conv2d(6,16,kernel_size=5),
            #nn.ReLU(True)
            )

        self.decoder = nn.Sequential(             
            nn.Linear(400, 400),
            nn.LeakyReLU(True),
            ConvView(1, 20, 20),
            nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.LeakyReLU(True),
            #nn.ConvTranspose2d(16,6,kernel_size=5),
            #nn.ReLU(True),
            #nn.ConvTranspose2d(6,3,kernel_size=5),
            #nn.ReLU(True)
            )

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# %%
"""
Credit: https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/gan/gan.py
"""
class Discriminator(nn.Module):
    def __init__(self, nc, nx, ny):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(int(np.prod([nc, nx, ny])), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)

        return validity

# %%
"""
Credit: https://github.com/zongyi-li/fourier_neural_operator
"""

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *

from PIL import Image

#Complex multiplication
def compl_mul2d(a, b):
    # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
    op = partial(torch.einsum, "bixy,ioxy->boxy")
    return torch.stack([
        op(a[..., 0], b[..., 0]) - op(a[..., 1], b[..., 1]),
        op(a[..., 1], b[..., 0]) + op(a[..., 0], b[..., 1])
    ], dim=-1)


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        Nx*Ny array requires Nx*(Ny/2+1) complex elements
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, 2))

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        # print ('SpectralConv2d', x.shape)
        x_ft = torch.rfft(x, 2, normalized=True, onesided=True)
        # print ('rfft', x_ft.shape)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.in_channels,  x.size(-2), x.size(-1)//2 + 1, 2, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        if False:
            ## (2021/02) Save lower-freq
            m = x_ft.shape[2]//2
            out_ft[:, :, m-self.modes1:m, -self.modes2:] = compl_mul2d(x_ft[:, :, m-self.modes1:m, -self.modes2:], self.weights1)
            out_ft[:, :, m:m+self.modes1, -self.modes2:] = compl_mul2d(x_ft[:, :, m:m+self.modes1, -self.modes2:], self.weights2)

        #Return to physical space
        x = torch.irfft(out_ft, 2, normalized=True, onesided=True, signal_sizes=( x.size(-2), x.size(-1)))
        # print ('irfft', x.shape)
        return x

class SpectralStack(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(SpectralStack, self).__init__()

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv1d(self.width, self.width, 1)
        self.bn0 = torch.nn.BatchNorm2d(self.width)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[-2], x.shape[-1]

        x1 = self.conv0(x)
        x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn0(x1 + x2)
        x = F.relu(x)
        return x

class SimpleBlock2d(nn.Module):
    def __init__(self, modes1, modes2, width, nlayers=3):
        super(SimpleBlock2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.nlayers = nlayers
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        # self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.w0 = nn.Conv1d(self.width, self.width, 1)
        # self.w1 = nn.Conv1d(self.width, self.width, 1)
        # self.w2 = nn.Conv1d(self.width, self.width, 1)
        self.w3 = nn.Conv1d(self.width, self.width, 1)
        # self.bn0 = torch.nn.BatchNorm2d(self.width)
        # self.bn1 = torch.nn.BatchNorm2d(self.width)
        # self.bn2 = torch.nn.BatchNorm2d(self.width)
        self.bn3 = torch.nn.BatchNorm2d(self.width)

        layer_list = list()
        for i in range(nlayers):
            layer_list.append(SpectralStack(modes1, modes2, width))
        self._layers = nn.ModuleList(layer_list)
        
        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        batchsize = x.shape[0]
        size_x, size_y = x.shape[1], x.shape[2]

        # print ('#0:', x.shape)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        
        # x1 = self.conv0(x)
        # x2 = self.w0(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        # x = self.bn0(x1 + x2)
        # x = F.relu(x)
        
        # # print ('#2:', x.shape)
        # x1 = self.conv1(x)
        # x2 = self.w1(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        # x = self.bn1(x1 + x2)
        # x = F.relu(x)
        
        # # print ('#3:', x.shape)
        # x1 = self.conv2(x)
        # x2 = self.w2(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        # x = self.bn2(x1 + x2)
        # x = F.relu(x)

        for i in range(len(self._layers)):
            x = self._layers[i](x)

        # print ('#4:', x.shape)
        x1 = self.conv3(x)
        x2 = self.w3(x.view(batchsize, self.width, -1)).view(batchsize, self.width, size_x, size_y)
        x = self.bn3(x1 + x2)


        # print ('#5:', x.shape)
        x = x.permute(0, 2, 3, 1)
        
        # print ('#6:', x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        
        # print ('#7:', x.shape)
        x = self.fc2(x)
        
        # print ('#8:', x.shape)
        return x

class Net2d(nn.Module):
    def __init__(self, modes, width, nlayers=3):
        super(Net2d, self).__init__()

        """
        A wrapper function
        """

        self.conv1 = SimpleBlock2d(modes, modes, width, nlayers=nlayers)


    def forward(self, x):
        x = self.conv1(x)
        return x.squeeze()


    def count_params(self):
        c = 0
        for p in self.parameters():
            c += reduce(operator.mul, list(p.size()))

        return c

def rescale(lx, grid):
    ntrain = len(lx)
    _nx, _ny = lx[0].shape
    nx, ny, _ = grid.shape
    x = torch.tensor(lx).view(ntrain,1,_nx,_ny)
    x = nn.functional.interpolate(x, size=(nx, ny))
    x = x.permute(0, 2, 3, 1)
    x = torch.cat([x, grid.repeat(ntrain,1,1,1)], dim=3)
    return (x)

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
    parser.add_argument('--exp', help='exp')
    parser.add_argument('-n', '--num_training_updates', help='num_training_updates (default: %(default)s)', type=int, default=10_000)
    parser.add_argument('-E', '--embedding_dim', help='embedding_dim (default: %(default)s)', type=int, default=16)
    parser.add_argument('-e', '--num_embeddings', help='num_embeddings (default: %(default)s)', type=int, default=512)
    parser.add_argument('-H', '--num_hiddens', help='num_hidden (default: %(default)s)', type=int, default=128)
    parser.add_argument('-R', '--num_residual_hiddens', help='num_residual_hiddens (default: %(default)s)', type=int, default=32)
    parser.add_argument('-L', '--num_residual_layers', help='num_residual_layers (default: %(default)s)', type=int, default=2)
    parser.add_argument('-C', '--num_channels', help='num_channels', type=int, default=1)
    parser.add_argument('-B', '--batch_size', help='batch_size (default: %(default)s)', type=int, default=32)
    parser.add_argument('-d', '--device_id', help='device_id (default: %(default)s)', type=int, default=0)
    parser.add_argument('--wdir', help='working directory (default: current)', default=os.getcwd())
    parser.add_argument('--datadir', help='data directory (default: %(default)s)', default='d3d_coarse_v2')
    parser.add_argument('--hr_datadir', help='HR data directory (default: %(default)s)')
    parser.add_argument('--timesteps', help='timesteps', nargs='+', type=int, default=[420,])
    # parser.add_argument('--surfid', help='flux surface index', nargs='+', type=int)
    parser.add_argument('--surfid', help='flux surface index')
    parser.add_argument('--untwist', help='untwist', action='store_true')
    parser.add_argument('--average_interval', help='average_interval (default: %(default)s)', type=int)
    parser.add_argument('--log_interval', help='log_interval (default: %(default)s)', type=int, default=1_000)
    parser.add_argument('--checkpoint_interval', help='checkpoint_interval (default: %(default)s)', type=int, default=10_000)
    parser.add_argument('--nompi', help='nompi', action='store_true')
    parser.add_argument('--seed', help='seed (default: %(default)s)', type=int)
    parser.add_argument('--nworkers', help='nworkers (default: %(default)s)', type=int)
    parser.add_argument('--log', help='log', action='store_true')
    parser.add_argument('--noise', help='noise value in (0,1)', type=float, default=None)
    parser.add_argument('--resampling', help='resampling', action='store_true')
    parser.add_argument('--resampling_interval', help='resampling_interval', type=int, default=None)
    parser.add_argument('--overwrite', help='overwrite', action='store_true')
    parser.add_argument('--learning_rate', '--lr', help='learning_rate', type=float, default=1e-3)
    parser.add_argument('--shaconv', help='shaconv', action='store_true')
    parser.add_argument('--meshgrid', help='meshgrid', action='store_true')
    parser.add_argument('--fno_nmodes', help='fno num. of modes', type=int, default=12)
    parser.add_argument('--fno_width', help='fno width', type=int, default=32)
    parser.add_argument('--fno_nlayers', help='fno num. of layers', type=int, default=3)
    parser.add_argument('--fno_rescale', help='rescale fno', type=int, default=1)
    parser.add_argument('--vgg', help='vgg', action='store_true')
    parser.add_argument('--conditional', help='conditional', action='store_true')
    parser.add_argument('--hr', help='high resolution', action='store_true')
    parser.add_argument('--milestones', help='scheduler milestones', nargs='*', type=int, default=[20_000,30_000,40_000])

    parser.add_argument('--c_alpha', help='c_alpha', type=float, default=1.0)
    parser.add_argument('--c_beta', help='c_beta', type=float, default=1.0)
    parser.add_argument('--c_gamma', help='c_gamma', type=float, default=1.0)
    parser.add_argument('--c_delta', help='c_delta', type=float, default=1.0)
    parser.add_argument('--c_zeta', help='c_zeta', type=float, default=1.0)
    parser.add_argument("--decoder_layer_sizes", type=int, nargs='*', default=[])

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--xgc', help='XGC dataset', action='store_const', dest='dataset', const='xgc')
    group.add_argument('--nstx', help='NSTX dataset', action='store_const', dest='dataset', const='nstx')
    parser.set_defaults(dataset='xgc')

    group1 = parser.add_argument_group('XGC', 'XGC processing options')
    group1.add_argument('--physicsloss', help='physicsloss', action='store_true')
    group1.add_argument('--physicsloss_interval', help='physicsloss_interval (default: %(default)s)', type=int, default=1)
    group1.add_argument('--randomread', help='randomread', type=float, default=0.0)
    group1.add_argument('--iphi', help='iphi', type=int, default=None)
    group1.add_argument('--nodestride', help='nodestride', type=int, default=1)
    group1.add_argument('--splitfiles', help='splitfiles', action='store_true')
    group1.add_argument('--overwrap', help='overwrap', type=int, default=1)
    group1.add_argument('--inode', help='inode', type=int, default=0)
    group1.add_argument('--nnodes', help='nnodes', type=int, default=None)
    group1.add_argument('--rescale', help='rescale', type=int, default=None)
    group1.add_argument('--rescaleinput', help='rescaleinput', type=int, default=None)
    group1.add_argument('--learndiff', help='learndiff', action='store_true')
    group1.add_argument('--learndiff2', help='learndiff2', action='store_true')
    group1.add_argument('--fieldline', help='fieldline', action='store_true')
    group1.add_argument('--saverecon', help='save recon', action='store_true')

    group2 = parser.add_argument_group('NSTX', 'NSTX processing options')
    ## 159065, 172585, 186106, 199626, 213146, 226667, 240187, 253708, 267228, 280749
    group2.add_argument('--offset', help='offset', type=int, default=159065)
    group2.add_argument('--nframes', help='nframes', type=int, default=16_384)
    group2.add_argument('--gaussian', help='apply gaussian filter', action='store_true')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--vae', help='vae model', action='store_const', dest='model', const='vae')
    group.add_argument('--cvae', help='cvae model', action='store_const', dest='model', const='cvae')
    group.add_argument('--vqvae', help='vqvae model', action='store_const', dest='model', const='vqvae')
    group.add_argument('--cvqvae', help='cvqvae model', action='store_const', dest='model', const='cvqvae')
    group.add_argument('--gan', help='gan model', action='store_const', dest='model', const='gan')
    group.add_argument('--fno', help='fno model', action='store_const', dest='model', const='fno')
    group.add_argument('--ae', help='ae model', action='store_const', dest='model', const='ae')
    group.add_argument('--cae', help='cae model', action='store_const', dest='model', const='cae')
    group.add_argument('--ae-vqvae', help='ae-vqvae model', action='store_const', dest='model', const='ae-vqvae')
    group.add_argument('--ae2d', help='ae conv2d model', action='store_const', dest='model', const='ae2d')
    parser.set_defaults(model='vqvae')
    args = parser.parse_args()

    DIR=args.wdir
    prefix='exp-%s-%s-B%d-C%d-H%d-R%d-L%d-E%d-e%d'%\
        (args.dataset, args.model, args.batch_size, args.num_channels, args.num_hiddens, args.num_residual_hiddens, args.num_residual_layers, args.embedding_dim, args.num_embeddings)
    if args.exp is not None:
        prefix = '%s-%s'%(prefix, args.exp)

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
        fname = "%s/run-%s-%d.log"%(_prefix, suffix, pid)
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
    
    # torch.set_deterministic(True)
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        # torch.set_deterministic(True)
        cudnn.deterministic = True # type: ignore
        cudnn.benchmark = False # type: ignore

    ## to support noise is given by "--noise=10**0"
    if args.noise == 0.0 or args.noise == 1.0:
        args.noise = None

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
    num_embeddings = args.num_embeddings

    commitment_cost = 0.25
    decay = 0.99
    learning_rate = args.learning_rate

    alpha, beta, gamma, delta, zeta = args.c_alpha, args.c_beta, args.c_gamma, args.c_delta, args.c_zeta
    #prefix='xgc-%s-batch%d-edim%d-nhidden%d-nchannel%d-nresidual_hidden%d'%(args.exp, args.batch_size, args.embedding_dim, args.num_hiddens, args.num_channels, args.num_residual_hiddens)
    logging.info ('prefix: %s' % prefix)
    writer = SummaryWriter('runs/%s'%prefix)

    # %%
    ## Reading data
    global Z0, zmu, zsig, zmin, zmax, zlb
    info ('Dataset:', args.dataset)
    assert(args.timesteps is not None)
    if args.dataset == 'xgc':
        ## (2020/12) single timestep. temporary.
        xgcexp = xgc4py.XGC(args.datadir, step=args.timesteps[0], device=device)
        #nnodes = xgcexp.mesh.nnodes if args.nnodes is None else args.nnodes
        
        timesteps = args.timesteps
        if args.splitfiles:
            timesteps = np.array_split(np.array(timesteps), size)[rank]
        f0_data_list = list()
        hr_data_list = list()
        logging.info (f'Data dir: {args.datadir}')
        for istep in timesteps:
            logging.info (f'Reading: {istep}')
            if args.surfid is not None:
                surfid_list = parse_rangestr(args.surfid)
                node_list = list()
                for i in surfid_list:
                    _nodes = xgcexp.mesh.surf_nodes(i)[::args.nodestride]
                    logging.info (f'Surf idx, len: {i} {len(_nodes)}')
                    node_list.extend(_nodes)
                if args.nnodes is not None:
                    node_list = node_list[:args.nnodes]
                nextnode_arr = xgcexp.nextnode_arr if args.untwist else None
                _out = read_f0_nodes(istep, node_list, expdir=args.datadir, iphi=args.iphi, nextnode_arr=nextnode_arr, rescale=args.rescaleinput)
                f0_data_list.append(_out)
                if args.hr:
                    assert(args.hr_datadir is not None)
                    _out2 = read_f0_nodes(istep, node_list, expdir=args.hr_datadir, iphi=args.iphi, nextnode_arr=nextnode_arr, rescale=args.rescaleinput)
                    hr_data_list.append(_out2)
            else:
                _out = read_f0(istep, expdir=args.datadir, iphi=args.iphi, inode=args.inode, nnodes=args.nnodes, \
                            randomread=args.randomread, nchunk=num_channels, fieldline=args.fieldline)
                f0_data_list.append(_out)

        lst = list(zip(*f0_data_list))

        Z0 = np.r_[(lst[0])]
        Xif = np.r_[(lst[1])]
        Zif = np.r_[(lst[1])]
        zmu = np.r_[(lst[2])]
        zsig = np.r_[(lst[3])]
        zmin = np.r_[(lst[4])]
        zmax = np.r_[(lst[5])]
        zlb = np.r_[(lst[6])] ## array of (idx, step, iphi, inode)
        if zlb.ndim == 1:
            zlb = zlb[:,np.newaxis]
        zlb = np.hstack([np.arange(len(zlb))[:,np.newaxis], zlb])
        ## z-score normalization
        #Zif = (Zif - zmu[:,np.newaxis,np.newaxis])/zsig[:,np.newaxis,np.newaxis]
        ## min-max normalization
        #Zif = (Zif - np.min(Zif))/(np.max(Zif)-np.min(Zif))
        #Zif = (Zif - zmin[:,np.newaxis,np.newaxis])/(zmax-zmin)[:,np.newaxis,np.newaxis]

        if args.hr:
            lst = list(zip(*hr_data_list))
            Zif = np.r_[(lst[1])]
            zmu = np.r_[(lst[2])]
            zsig = np.r_[(lst[3])]
            zmin = np.r_[(lst[4])]
            zmax = np.r_[(lst[5])]

    if args.dataset == 'nstx':
        Z0, Zif, zmu, zsig, zmin, zmax, zlb = read_nstx(args.datadir, args.offset, args.nframes, gaussian=args.gaussian)
        Xif = Zif

    log ('Zif bytes,shape:', Zif.size * Zif.itemsize, Zif.shape, zmu.shape, zsig.shape)
    log ('Minimum training epoch:', Zif.shape[0]/batch_size)

    if args.model == 'fno':
        with ad2.open('d3d_coarse_v2-recon.bp', 'r') as f:
            Xrec = f.read('i_f_recon').astype(np.float32)
            # if Xrec.shape[3] == 39:
            #     Xrec = np.append(Xrec, Xrec[...,38:39], axis=3)
            #     Xrec = np.append(Xrec, Xrec[:,:,38:39,:], axis=2)

        _, nx, ny = Z0.shape
        nx = nx * args.fno_rescale
        ny = ny * args.fno_rescale
        x = np.linspace(0, 1, nx, dtype=np.float32)
        y = np.linspace(0, 1, ny, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        # grid = np.stack([xv, yv], axis=2)
        # grid = torch.tensor(grid, dtype=torch.float)

        lx = list()
        ly = list()
        for i in range(len(zlb)):
            iphi, inode = zlb[i,2:]
            X = Xrec[iphi,inode,:]
            # X = (X-np.min(X))/(np.max(X)-np.min(X))
            if args.fno_rescale>1:
                X = resize(X, (nx,ny), order=0, anti_aliasing=False)
            # img = Image.fromarray(X)
            # img = img.resize((Z0.shape[-2],Z0.shape[-1]))
            # X = np.array(img)
            lx.append(np.stack([X, xv, yv], axis=2))
            # lx.append(Xrec[i,:])
            Z = Zif[i,:]
            if args.fno_rescale>1:
                Z = resize(Z, (nx,ny), order=0, anti_aliasing=False)
            ly.append(Z)
        
        X_train, X_test, y_train, y_test = train_test_split(lx, ly, test_size=0.3)
        print (lx[0].shape, ly[0].shape, len(X_train), len(X_test))

        # X_train, y_train = rescale(X_train, grid), torch.tensor(y_train)
        # X_test, y_test = rescale(X_test, grid), torch.tensor(y_test)
        # X_full, y_full = rescale(lx, grid), torch.tensor(ly)
        X_train, y_train = torch.tensor(X_train), torch.tensor(y_train)
        X_test, y_test = torch.tensor(X_test), torch.tensor(y_test)
        X_full, y_full = torch.tensor(lx), torch.tensor(ly)

        training_data = torch.utils.data.TensorDataset(X_train, y_train)
        validation_data = torch.utils.data.TensorDataset(X_test, y_test)
        full_data = torch.utils.data.TensorDataset(X_full, y_full)

        train_loader = torch.utils.data.DataLoader(full_data, batch_size=batch_size, shuffle=True)
        test_loader = torch.utils.data.DataLoader(full_data, batch_size=batch_size, shuffle=True)
        full_loader = torch.utils.data.DataLoader(full_data, batch_size=1, shuffle=False)

        y_normalizer = UnitGaussianNormalizer(torch.tensor(ly))

        ################################################################
        # configs
        ################################################################
        ntrain = len(train_loader)
        ntest = len(test_loader)
        # modes = 12
        # width = 32
        step_size = 100
        gamma = 0.5
        modes = args.fno_nmodes
        width = args.fno_width
        nlayers = args.fno_nlayers

        model = Net2d(modes, width, nlayers=nlayers)
        print(model.count_params())
        istart = 1

        _istart, _model, _dmodel = 0, None, None
        if not args.overwrite: _istart, _model, _dmodel = load_checkpoint(DIR, prefix, model)
        if _model is not None:
            istart = _istart + 1
            model = _model
        log ('istart:', istart)
        model = model.to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

        myloss = LpLoss(size_average=False)
        y_normalizer.to(device)
        for ep in range(istart, istart+args.num_training_updates):
            model.train()
            t1 = default_timer()
            train_err = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                # print ('x,y:', x.shape, y.shape)

                optimizer.zero_grad()
                # loss = F.mse_loss(model(x).view(-1), y.view(-1), reduction='mean')
                out = model(x)
                # out = y_normalizer.decode(out)
                # y = y_normalizer.decode(y)
                loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
                loss.backward()

                optimizer.step()
                train_err += loss.item()

            scheduler.step()

            model.eval()
            abs_err = 0.0
            test_err = 0.0
            with torch.no_grad():
                x = out.detach().cpu().numpy()
                y = y.detach().cpu().numpy()
                #import pdb; pdb.set_trace()
                #np.max(np.abs(x,y), dim=(2,3))

                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    out = model(x)
                    # out = y_normalizer.decode(out)
                    
                    abs_err = max(abs_err, torch.max(nn.L1Loss(reduction='none')(y, out)).item())
                    test_err += myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()

            train_err /= ntrain
            test_err /= ntest

            t2 = default_timer()
            log(ep, t2-t1, train_err, test_err, abs_err)

            if (ep % args.checkpoint_interval == 0) and (rank == 0):
                save_checkpoint(DIR, prefix, model, train_err, ep)

                out_list = list()
                out1_list = list()
                y_list = list()
                abs_err = 0.0
                for x, y in full_loader:
                    x, y = x.to(device), y.to(device)

                    out = model(x)
                    out1 = y_normalizer.decode(out)
                    out_list.append(out.detach().cpu().numpy())
                    out1_list.append(out1.detach().cpu().numpy().copy())
                    y_list.append(torch.squeeze(y).detach().cpu().numpy().copy())

                    ## Need squeeze for batch size 1
                    abs_err = max(abs_err, torch.max(nn.L1Loss(reduction='none')(y.squeeze(), out)).item())

                log('ABS:', abs_err)
                
                print (np.array(out_list).shape, np.array(out1_list).shape)
                with ad2.open('d3d_coarse_v2-fno.bp', 'w') as fw:
                    out = np.array(out_list)
                    out1 = np.array(out1_list)
                    shape, start, count = out.shape, [0,]*out.ndim, out.shape
                    fw.write('recon', out.copy(), shape, start, count)
                    fw.write('norm', out1.copy(), shape, start, count)
                    tmp = np.array(lx)
                    tmp = tmp[:,:,:,0].copy()
                    fw.write('enc', tmp, shape, start, count)
                    fw.write('dec', np.array(y_list).copy(), shape, start, count)

                    shape, start, count = zlb.shape, [0,]*zlb.ndim, zlb.shape
                    fw.write('zlb', zlb.copy(), shape, start, count)

        return 0
    ## end of fno

    grid = None
    if args.meshgrid:
        assert(num_channels==1)
        _, nx, ny = Z0.shape
        x = np.linspace(0, 1, nx, dtype=np.float32)
        y = np.linspace(0, 1, ny, dtype=np.float32)
        xv, yv = np.meshgrid(x, y)
        grid = np.stack([xv, yv])
        grid = torch.tensor(grid, dtype=torch.float).to(device)

    ## Preparing training and validation set
    lx = list()
    ly = list()
    lh = list()
    for i in range(0,len(Xif)-num_channels+1,num_channels//args.overwrap):
        X = Xif[i:i+num_channels,:,:]
        mu = zmu[i:i+num_channels]
        sig = zsig[i:i+num_channels]
        N = X.astype(np.float32)
        # N = np.vstack([N, xv[np.newaxis,:,:], yv[np.newaxis,:,:]])
        #N = rescale(N, 2.0, anti_aliasing=False, multichannel=True)
        ## z-score normalization
        #N = (X - mu[:,np.newaxis,np.newaxis])/sig[:,np.newaxis,np.newaxis]
        lx.append(N)
        ly.append(zlb[i:i+num_channels])
        if args.hr:
            H = Zif[i:i+num_channels,:,:]
            H = H.astype(np.float32)
            lh.append(H)
        else:
            lh.append(0)

    _lx = [ x[0,:] for x in lx ]
    data_variance = np.var(_lx, dtype=np.float64)
    log ('data_variance', data_variance)
    if args.hr:
        _lh = [ x[0,:] for x in lh ]
        hr_data_variance = np.var(_lh, dtype=np.float64)

    # %% 
    # Loadding
    # X_train, X_test, y_train, y_test = train_test_split(lx, ly, test_size=0.10, random_state=42)
    # training_data = torch.utils.data.TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    # validation_data = torch.utils.data.TensorDataset(torch.tensor(X_test), torch.tensor(y_test))    
    # (2020/11) Temporary. Use all data for training
    training_data = torch.utils.data.TensorDataset(torch.tensor(lx), torch.tensor(ly), torch.tensor(lh))
    validation_data = torch.utils.data.TensorDataset(torch.tensor(lx), torch.tensor(ly), torch.tensor(lh))

    training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    validation_loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True, pin_memory=True)

    # %% 
    # Model
    # if args.learndiff:
    #     model = Model(num_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
    #                 num_embeddings, embedding_dim, 
    #                 commitment_cost, decay, rescale=args.rescale, learndiff=args.learndiff, 
    #                 input_shape=[batch_size, num_channels, Zif.shape[-2], Zif.shape[-1]]).to(device)
    # else:
    #     model = Model(num_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
    #                 num_embeddings, embedding_dim, 
    #                 commitment_cost, decay, rescale=args.rescale, learndiff=args.learndiff).to(device)
    
    def setup_da():
        fname2 = os.path.join(args.datadir, 'xgc.mesh.bp')
        with ad2.open(fname2, 'r') as f:
            rz = f.read('rz')

        _rz = np.array(rz[:,0], dtype=complex)
        _rz.imag = rz[:,1]
        
        global da
        da = np.zeros((len(zlb), 2), dtype=np.float32) ## list of distance and angle pair
        for inode in  zlb[:,3]:
            dist = np.linalg.norm(_rz[inode] - _rz[0])
            angle = np.angle(_rz[inode] - _rz[0])
            da[inode,0] = dist
            da[inode,1] = angle
        da = (da - np.min(da, axis=0))/(np.max(da, axis=0) - np.min(da, axis=0))
        log ('Condition information (distance and angle) normalized:', np.min(da, axis=0), np.max(da, axis=0))
        da = torch.tensor(da, dtype=torch.float).to(device)

    _, nx, ny = Z0.shape
    padding = [1,1,1]
    if nx == 39: padding = [1,1,0]
    if nx == 45: padding = [1,0,0]
    if args.model in ('vqvae','cvqvae'):
        da_conditional = False
        if args.model == 'cvqvae':
            setup_da()
            da_conditional = True
        model = Model(num_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                    num_embeddings, embedding_dim, 
                    commitment_cost, decay, rescale=args.rescale, learndiff=args.learndiff, 
                    shaconv=args.shaconv, grid=grid, conditional=args.conditional, decoder_padding=padding, da_conditional=da_conditional,
                    decoder_layer_sizes=args.decoder_layer_sizes).to(device)

    if args.model == 'vae':
        model = VAE(args.num_channels, nx, ny, nx*ny//4, nx*ny//4//4, num_residual_hiddens, num_residual_layers, shaconv=args.shaconv).to(device)

    if args.model == 'cvae':
        setup_da()
        model = VAE(args.num_channels, nx, ny, nx*ny//4, nx*ny//4//4, num_residual_hiddens, num_residual_layers, da_conditional=True).to(device)
    
    if args.model == 'gan':
        model = Model(num_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                    num_embeddings, embedding_dim, 
                    commitment_cost, decay, rescale=args.rescale, learndiff=args.learndiff, shaconv=args.shaconv, decoder_padding=padding).to(device)
        discriminator = Discriminator(args.num_channels, nx, ny).to(device)
        adversarial_loss = torch.nn.BCELoss().to(device)
        optimizer_D = optim.Adam(discriminator.parameters(), lr=learning_rate, amsgrad=False)

    if args.model == 'ae':
        log ('AE model: input_dim: %d, embedding_dim= %d'%(num_channels*ny*nx, args.embedding_dim))
        model = AE(input_dim=num_channels*ny*nx, embedding_dim=args.embedding_dim, conditional=args.conditional).to(device)

    if args.model == 'cae':
        setup_da()
        log ('CAE model: input_dim: %d, embedding_dim= %d'%(num_channels*ny*nx, args.embedding_dim))
        model = AE(input_dim=num_channels*ny*nx, embedding_dim=args.embedding_dim, da_conditional=True).to(device)

    if args.model == 'ae-vqvae':
        model = AE(input_dim=num_channels*ny*nx, embedding_dim=400).to(device)

        model2 = Model(num_channels, num_hiddens, num_residual_layers, num_residual_hiddens,
                    num_embeddings, embedding_dim, 
                    commitment_cost, decay, rescale=args.rescale, learndiff=args.learndiff, 
                    shaconv=args.shaconv, grid=grid, conditional=args.conditional, decoder_padding=padding).to(device)

        optimizer2 = optim.Adam(model2.parameters(), lr=learning_rate, amsgrad=False)

    if args.model == 'ae2d':
        model = Autoencoder().to(device)

    # %%
    train_res_recon_error = []
    train_res_perplexity = []
    train_res_physics_error = []
    istart = 1

    # Load checkpoint
    _istart, _model, _dmodel = 0, None, None
    if not args.overwrite: 
        _istart, _model, _dmodel = load_checkpoint(DIR, prefix, model)
        
    if _model is not None:
        istart = _istart + 1
        model = _model
    model.train()
    log ('istart:', istart)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)

    dmodel = None
    if args.learndiff2:
        dim1, dim2 = Zif.shape[-2], Zif.shape[-1]
        dmodel = AE(input_shape=num_channels*dim1*dim2).to(device)
        doptimizer = optim.Adam(dmodel.parameters(), lr=1e-3)
        dcriterion = nn.MSELoss()

    if args.vgg:
        modelfile = 'xgc-vgg19-ch1-N1000.torch'
        feature_extractor = XGCFeatureExtractor(modelfile)
        feature_extractor = feature_extractor.to(device)
        feature_extractor.eval()

        # criterion_content = torch.nn.MSELoss().to(device)
        criterion_content = torch.nn.L1Loss().to(device)

    # %%
    nworkers = args.nworkers if args.nworkers is not None else 8
    if args.nworkers is None and hasattr(os, 'sched_getaffinity'):
        nworkers = len(os.sched_getaffinity(0))-1 
    logging.info('Nworkers: %d'%nworkers)

    counter = mp.Value('i', 0)
    executor = ProcessPoolExecutor(max_workers=nworkers, initializer=init, initargs=(counter,))
    num_training_updates = args.num_training_updates
    resampling_interval = len(lx)//batch_size*10 if args.resampling_interval is None else args.resampling_interval
    logging.info (f'Rsampling, resampling interval: {args.resampling} {resampling_interval}')
    total_trained = np.ones(len(lx), dtype=np.int32)
    logging.info ('Training: %d' % num_training_updates)
    model.train()
    ns = 0.0 
    t0 = time.time()
    for i in xrange(istart, istart+num_training_updates):
        (data, lb, hr_data) = next(iter(training_loader))
        # print ("Training:", lb)
        data = data.to(device)
        if args.hr:
            hr_data = hr_data.to(device)
            hr_data_variance = hr_data_variance
        else:
            hr_data = data
            hr_data_variance = data_variance

        if args.noise is not None:
            ns = torch.normal(mean=0.0, std=data.detach()*args.noise)

        optimizer.zero_grad() # clear previous gradients
        
        if args.model in ('vqvae','cvqvae'):
            _da = None
            if args.model == 'cvqvae':
                _da = da[lb[:,0,-1]]

            vq_loss, data_recon, perplexity, dloss = model(data+ns, _da)
            ## mean squared error: torch.mean((data_recon - data)**2)
            ## relative variance
            # import pdb; pdb.set_trace()
            recon_error = F.mse_loss(data_recon, hr_data) / hr_data_variance
            physics_error = torch.tensor(0.0).to(data_recon.device)
            if args.physicsloss and (i % args.physicsloss_interval == 0):
                # den_err, u_para_err, T_perp_err, T_para_err = physics_loss_con(data, lb, data_recon, executor=executor)
                den_err, u_para_err, T_perp_err, T_para_err = physics_loss(hr_data, lb, data_recon)
                # ds = torch.mean(data_recon.cpu().data.numpy()**2)
                if i % args.log_interval == 0:
                    print ('Physics loss:', den_err, u_para_err, T_perp_err, T_para_err)
                # physics_error += den_err/ds * torch.mean(data_recon)
                physics_error += den_err + u_para_err + T_perp_err + T_para_err

            feature_loss = torch.tensor(0.0).to(data_recon.device)
            if args.vgg:
                recon_features = feature_extractor(data_recon)
                data_features = feature_extractor(data+ns)
                feature_loss = criterion_content(recon_features, data_features)

            loss = alpha*recon_error + beta*vq_loss + gamma*physics_error + delta*dloss + zeta*feature_loss
            loss.backward()
            writer.add_scalar('Loss/train', loss, i)
            if (args.average_interval is not None) and (i%args.average_interval == 0):
                ## Gradient averaging
                logging.info('iteration %d: gradient averaging' % (i))
                average_gradients(model)
            optimizer.step()

        if args.model in ('vae','cvae'):
            _da = None
            if args.model == 'cvae':
                _da = da[lb[:,0,-1],]
            recon_batch, mu, logvar = model(data, _da)

            loss = model.loss_function(recon_batch, hr_data, mu, logvar)
            recon_error = F.mse_loss(recon_batch, hr_data.view(-1, nx*ny)) / hr_data_variance
            perplexity = torch.tensor(0)
            physics_error = torch.tensor(0.0)
            loss.backward()
            optimizer.step()

        if args.model == 'gan':
            valid = torch.ones(args.batch_size, 1, requires_grad=False).to(device)
            fake = torch.zeros(args.batch_size, 1, requires_grad=False).to(device)
            #z = torch.tensor(np.random.normal(0, 1, data.shape)).to(device)

            # vq_loss, data_recon, perplexity, dloss = model(data+ns)
            vq_loss, data_recon, perplexity, dloss = model(data+ns)
            recon_error = F.mse_loss(data_recon, hr_data) / hr_data_variance
            physics_error = torch.tensor(0.0).to(data_recon.device)

            dout = discriminator(data_recon)
            g_loss = adversarial_loss(dout, valid)

            loss = recon_error + vq_loss + physics_error + dloss + g_loss

            loss.backward()
            optimizer.step()

            #  Train Discriminator
            optimizer_D.zero_grad()
            real_dout = discriminator(hr_data)
            fake_dout = discriminator(data_recon.detach())
            real_loss = adversarial_loss(real_dout, valid)
            fake_loss = adversarial_loss(fake_dout, fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

        if args.model in ('ae', 'cae'):
            _da = None
            if args.model == 'cae':
                _da = da[lb[:,0,-1],]
            recon_batch = model(data, _da)

            recon_error = F.mse_loss(recon_batch, hr_data) / hr_data_variance
            #l1loss = nn.L1Loss()
            #recon_error = l1loss(recon_batch, data) 
            perplexity = torch.tensor(0)
            physics_error = torch.tensor(0.0)
            loss = recon_error
            loss.backward()
            optimizer.step()

        if args.model == 'ae2d':
            recon_batch = model(data)
            recon_error = F.mse_loss(recon_batch, hr_data) / hr_data_variance
            perplexity = torch.tensor(0)
            physics_error = torch.tensor(0.0)
            loss = recon_error
            loss.backward()
            optimizer.step()

        if args.model == 'ae-vqvae':
            recon_batch = model(data+ns)
            recon_error = F.mse_loss(recon_batch, hr_data.detach()) / hr_data_variance
            #l1loss = nn.L1Loss()
            #recon_error = l1loss(recon_batch, data) 
            perplexity = torch.tensor(0)
            physics_error = torch.tensor(0.0)
            loss = recon_error
            loss.backward()
            optimizer.step()

            optimizer2.zero_grad()
            vq_loss, data_recon, perplexity, dloss = model2(recon_batch.detach())
            recon2_error = F.mse_loss(data_recon, hr_data.detach()) / hr_data_variance
            physics2_error = torch.tensor(0.0).to(data_recon.device)
            feature_loss = torch.tensor(0.0).to(data_recon.device)
            loss2 = alpha*recon2_error + beta*vq_loss + gamma*physics2_error + delta*dloss + zeta*feature_loss
            loss2.backward()
            optimizer2.step()

        train_res_recon_error.append(recon_error.item())
        train_res_perplexity.append(perplexity.item())
        train_res_physics_error.append(physics_error.item())
        scheduler.step()

        if args.resampling and (i % resampling_interval == 0):
            err_list, _ = estimate_error(model, Xif, Zif, zmin, zmax, zlb, num_channels, modelname=args.model, conditional=args.conditional)
            err = np.array([max(err_list[i: i+num_channels]) for i in xrange(0, len(err_list), num_channels)])
            idx = np.random.choice(range(len(lx)), len(lx), p=err/sum(err))
            lxx = [ lx[i] for i in idx ]
            lyy = [ ly[i] for i in idx ]
            lhh = [ lh[i] for i in idx ]

            training_data = torch.utils.data.TensorDataset(torch.tensor(lxx), torch.tensor(lyy), torch.tensor(lhh))
            training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)
            total_trained[idx] += 1

        if i % args.log_interval == 0:
            logging.info(f'{i} time: {time.time()-t0:.3f}')
            logging.info(f'{i} Avg: {np.mean(train_res_recon_error[-args.log_interval:]):g} {np.mean(train_res_perplexity[-args.log_interval:]):g} {np.mean(train_res_physics_error[-args.log_interval:]):g}')
            if args.model in ('vqvae','cvqvae'):
                logging.info(f'{i} Loss: {recon_error.item():g} {vq_loss.data.item():g} {perplexity.item():g} {physics_error:g} {dloss:g} {feature_loss.item():g} {len(training_loader.dataset)} {len(data)}')
                
                if args.learndiff2:
                    logging.info(f'{i} dloss: {dloss.item():g}')
        
            if args.model in ('vae','cvae','ae','cae','ae2d'):
                logging.info(f'{i} Loss: {recon_error.item():g}')

            if args.model == 'gan':
                logging.info(f'{i} Loss: {recon_error.item():g} {vq_loss.data.item():g} {perplexity.item():g} {physics_error:g} {dloss:g} {len(training_loader.dataset)} {len(data)}')
                logging.info(f'{i} G-D loss: {g_loss.item():g} {d_loss.item():g}')

            fname = None
            if (i % args.checkpoint_interval == 0) and (rank == 0):
                fname='%s/%s/img-%d.jpg'%(DIR,prefix,i)
            rmse_list, abserr_list = estimate_error(model, Xif, Zif, zmin, zmax, zlb, num_channels, modelname=args.model, fname=fname, conditional=args.conditional)
            logging.info(f'{i} Error: {np.max(rmse_list):g} {np.max(abserr_list):g} Next LR: {optimizer.param_groups[0]["lr"]:g}')

        if (i % args.checkpoint_interval == 0) and (rank == 0):
            save_checkpoint(DIR, prefix, model, train_res_recon_error, i, dmodel=dmodel)
            writer.flush()
    istart=istart+num_training_updates

    # %%
    model.eval()
    with torch.no_grad():
        (valid_originals, valid_labels, hr_originals) = next(iter(validation_loader))
        valid_originals = valid_originals.to(device)
        nbatch, nchannel, dim1, dim2 = valid_originals.shape
        if args.hr:
            hr_originals = hr_originals.to(device)
        else:
            hr_originals = valid_originals

        if args.model in ('vqvae','cvqvae'):
            if model._grid is not None:
                x = valid_originals
                x = torch.cat([x, model._grid.repeat(nbatch,1,1,1)], dim=1)
                x = x.permute(0, 2, 3, 1)
                x = model.fc0(x)
                x = x.permute(0, 3, 1, 2)
                valid_originals = x

            _da = None
            if args.model == 'cvqvae':
                _da = da[valid_labels[:,0,-1],]

            vq_encoded = model._encoder(valid_originals, _da)
            vq_output_eval = model._pre_vq_conv(vq_encoded)
            _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
            if args.conditional:
                x = valid_originals
                x = torch.cat((x,x,x), axis=1)
                feature = model.feature_extractor(x)
                cond = F.avg_pool2d(feature, kernel_size=3, stride=2, padding=1)
                # feature = model.feature_extractor(valid_originals)
                # cond = F.avg_pool1d(feature.view(nbatch,nchannel,1090), kernel_size=11)
                # p1d = (0, 1)
                # cond = F.pad(cond, p1d, "constant", 0)
                # cond = torch.reshape(cond, (nbatch, 4, 5, 5))
                valid_quantize = torch.cat((valid_quantize, cond), dim=1)
            valid_reconstructions = model._decoder(valid_quantize, _da)
            if model._grid is not None:
                x = valid_reconstructions
                x = x.permute(0, 2, 3, 1)
                x = model.fc1(x)
                x = F.relu(x)
                x = model.fc2(x)
                x = x.permute(0, 3, 1, 2)
                valid_reconstructions = x

            logging.info ('Original: %s' % (valid_originals.cpu().numpy().shape,))
            logging.info ('Encoded: %s' % (vq_encoded.detach().cpu().numpy().shape,))
            logging.info ('Quantized: %s' % (valid_quantize.detach().cpu().numpy().shape,))
            logging.info ('Reconstructed: %s' % (valid_reconstructions.detach().cpu().numpy().shape,))
            logging.info ('compression ratio: %.2fx'%(valid_originals.cpu().numpy().size/vq_encoded.detach().cpu().numpy().size))
            logging.info ('compression ratio: %.2fx'%(valid_originals.cpu().numpy().size/valid_quantize.detach().cpu().numpy().size))

        if args.model in ('vae','cvae'):
            _da = None
            if args.model == 'cvae':
                _da = da[valid_labels[:,0,-1],]
            valid_reconstructions, mu, logvar = model(valid_originals, _da)

            logging.info ('Original: %s' % (valid_originals.cpu().numpy().shape,))
            # logging.info ('Encoded: %s' % (vq_encoded.detach().cpu().numpy().shape,))
            # logging.info ('Quantized: %s' % (valid_quantize.detach().cpu().numpy().shape,))
            logging.info ('Reconstructed: %s' % (valid_reconstructions.detach().cpu().numpy().shape,))
            logging.info ('compression ratio: %.2fx'%(model.nx*model.ny/model.nz))
            # logging.info ('compression ratio: %.2fx'%(valid_originals.cpu().numpy().size/vq_encoded.detach().cpu().numpy().size))
            # logging.info ('compression ratio: %.2fx'%(valid_originals.cpu().numpy().size/valid_quantize.detach().cpu().numpy().size))

        logging.info ('Reconstructing ...')
        if args.model == 'ae-vqvae':
            dmodel = model2
        X0, Xbar, xmu = recon(model, Xif, zmin, zmax, zlb, num_channels=num_channels, dmodel=dmodel, modelname=args.model, conditional=args.conditional)
        log (Xif.shape, Xbar.shape)

        if args.saverecon:
            fname = "%s/recon.bp"%(prefix)
            with ad2.open(fname, 'w') as fw:
                shape = Xbar.shape
                start = [0,]*len(shape)
                count = shape
                fw.write('Xbar', Xbar.copy(), shape, start, count)
                fw.write('X0', X0.copy(), shape, start, count)
                fw.write('Zif', Zif.copy(), shape, start, count)
                shape = zlb.shape
                start = [0,]*len(shape)
                count = shape
                fw.write('zlb', zlb.copy(), shape, start, count)
            logging.info('Recon saved: %s'%fname)
            logging.info('Done.')

        rmse_list = list()
        abs_list = list()
        psnr_list = list()
        ssim_list = list()
        for i in range(len(Xbar)):
            Z = Zif[i,:,:]
            X = Xbar[i,:,:]
            ## RMSE
            rmse = np.sqrt(np.sum((Z-X)**2)/Z.size)
            rmse_list.append(rmse)
            ## ABS error
            abserr = np.max(np.abs(Z-X))
            abs_list.append(abserr)
            ## PSNR
            psnr = 20*np.log10(1.0/np.sqrt(np.sum((Z-X)**2)/Z.size))
            psnr_list.append(psnr)
            ## SSIM
            #_ssim = ssim(Z, X, data_range=X.max()-X.min())
            #ssim_list.append(_ssim)

        info ('RMSE error: %g %g %g'%(np.min(rmse_list), np.mean(rmse_list), np.max(rmse_list)))
        info ('ABS error: %g %g %g'%(np.min(abs_list), np.mean(abs_list), np.max(abs_list)))
        info ('total_trained:')
        info (total_trained)

        info ('')
        info ('Model params')
        info ('-'*50)
        num_params = 0
        for k, v in model.state_dict().items():
            info ('%50s\t%20s\t%10d'%(k, list(v.shape), v.numel()))
            num_params += v.numel()
        info ('-'*50)
        info ('%50s\t%20s\t%10d'%('Total', '', num_params))
        info ('All (total, MB, ratio): %d %g %g'%(num_params, num_params*4/1024/1024, num_params/nx/ny))

        if args.model in ('vqvae','cvqvae'):
            num_params = 0
            for k, v in model._decoder.state_dict().items():
                num_params += v.numel()
            info ('Decoder (total, MB, ratio): %d %g %g'%(num_params, num_params*4/1024/1024, num_params/nx/ny))

        if args.model in ('ae','cae'):
            num_params = 0
            for k, v in model._decoder.state_dict().items():
                num_params += v.numel()
            info ('Decoder (total, MB, ratio): %d %g %g'%(num_params, num_params*4/1024/1024, num_params/nx/ny))

        if args.model == 'ae-vqvae':
            info ('-'*50)
            num_params = 0
            for k, v in model2.state_dict().items():
                info ('%50s\t%20s\t%10d'%(k, list(v.shape), v.numel()))
                num_params += v.numel()
            info ('-'*50)
            info ('%50s\t%20s\t%10d'%('Total', '', num_params))
            info ('VQVAE (total, MB, ratio): %d %g %g'%(num_params, num_params*4/1024/1024, num_params/nx/ny))

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
    
