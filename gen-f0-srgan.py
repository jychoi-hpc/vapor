import numpy as np
import adios2 as ad2
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from skimage.transform import resize

## https://github.com/jychoi-hpc/xgc4py
import xgc4py
from vapor import *

import argparse

def recon(model, Zif, zmin, zmax, num_channels=16, dmodel=None):
    lx = list()
    ly = list()
    for i in range(0,len(Zif),num_channels):
        X = Zif[i:i+num_channels,:,:]
        N = X.astype(np.float32)
        lx.append(N)
        ly.append(zlb[i])

    lz = list()
    nbatch = 1
    for i in range(0, len(lx), nbatch):
        valid_originals = torch.tensor(lx[i:i+nbatch]).to(device)
        vq_encoded = model._encoder(valid_originals)
        vq_output_eval = model._pre_vq_conv(vq_encoded)
        _, valid_quantize, _, _ = model._vq_vae(vq_output_eval)
        valid_reconstructions = model._decoder(valid_quantize)
        #print (valid_originals.sum().item(), valid_reconstructions.shape, valid_reconstructions.sum().item())

        drecon = 0
        _, nchannel, dim1, dim2 = valid_originals.shape
        if dmodel is not None:
            dx = (valid_originals-valid_reconstructions).view(-1, nchannel*dim1*dim2)
            drecon = dmodel(dx).view(-1, nchannel, dim1, dim2)   

        lz.append((valid_reconstructions+drecon).cpu().data.numpy())

    Xbar = np.array(lz).reshape([-1,dim1,dim2])

    ## Normalize
    xmin = np.min(Xbar, axis=(1,2))
    xmax = np.max(Xbar, axis=(1,2))
    Xbar = (Xbar-xmin[:,np.newaxis,np.newaxis])/(xmax-xmin)[:,np.newaxis,np.newaxis]

    ## Un-normalize
    X0 = Xbar*((zmax-zmin)[:,np.newaxis,np.newaxis])+zmin[:,np.newaxis,np.newaxis]
    
    return (X0, Xbar, np.mean(X0, axis=(1,2)))

def saveimg(X, outdir, seq, nx=None):
    os.makedirs(outdir, exist_ok=True)
    if nx is not None:
        X = resize(X, (nx,nx), order=3, anti_aliasing=False)
    X = (X-X.min())/(X.max()-X.min())*255
    X = X.astype(np.float32).copy()

    im = Image.fromarray(np.uint8(X))
    #im = im.resize((n, n), Image.BICUBIC)
    #fname = '%s/%d-%d-%05d.jpg'%(outdir,420,iphi,inode)
    fname = '%s/%06d.jpg'%(outdir,seq)
    print (fname)
    im.save(fname)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=int, default=64, help="low res. image width (default: %(default)s)")
    parser.add_argument("--hr", type=int, default=256, help="high res. image height (default: %(default)s)")
    opt = parser.parse_args()
    
    step=420
    xgcexp = xgc4py.XGC('d3d_coarse_v2', step=step)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    istep = 50_000
    mdir='exp-xgc-vqvae-B32-C1-H128-R256-L2-E16-s76-80-t1' #ABS error: 0.00654279 0.0340131 0.268001

    mfile = os.path.join(mdir, 'checkpoint.%d.pytorch'%istep)
    model = torch.load(mfile, map_location=torch.device(device))
    model.eval();

    dmodel = None
    mfile = os.path.join(mdir, 'checkpoint-dmodel.%d.pytorch'%istep)
    if os.path.exists(mfile):
        dmodel = torch.load(mfile, map_location=torch.device(device))
        dmodel.eval();

    hr_list = list()
    lr_list = list()
    for surfid in range(76,81):
        nodes = xgcexp.mesh.surf_nodes(surfid)
        Z0, Zif, zmu, zsig, zmin, zmax, zlb = read_f0_nodes(step, nodes, expdir='d3d_coarse_v2')
        X0, Xbar, xmu = recon(model, Zif, zmin, zmax, num_channels=1, dmodel=dmodel)
        hr_list.append(Zif)
        lr_list.append(Xbar)

    HR = np.vstack(hr_list)
    LR = np.vstack(lr_list)

    for i in range(len(LR)):
        X = LR[i,:,:]
        outdir='xgc_images-d3d_coarse_v2-lr'
        saveimg(X, outdir, i)

        Z = HR[i,:,:]
        outdir='xgc_images-d3d_coarse_v2-hr'
        saveimg(Z, outdir, i, nx=opt.hr)

