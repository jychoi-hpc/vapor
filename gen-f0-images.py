import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.axes_grid1 import make_axes_locatable

import numpy as np
import adios2 as ad2

from PIL import Image
import os

from matplotlib.image import imread
from tempfile import NamedTemporaryFile

import concurrent.futures
from tqdm import tqdm
import queue
import argparse

def get_size(fig, dpi=100):
    with NamedTemporaryFile(suffix='.png') as f:
        fig.savefig(f.name, bbox_inches='tight', dpi=dpi)
        height, width, _channels = imread(f.name).shape
        return width / dpi, height / dpi

def set_size(fig, size, dpi=100, eps=1e-2, give_up=2, min_size_px=10):
    target_width, target_height = size
    set_width, set_height = target_width, target_height # reasonable starting point
    deltas = [] # how far we have
    while True:
        fig.set_size_inches([set_width, set_height])
        actual_width, actual_height = get_size(fig, dpi=dpi)
        set_width *= target_width / actual_width
        set_height *= target_height / actual_height
        deltas.append(abs(actual_width - target_width) + abs(actual_height - target_height))
        if deltas[-1] < eps:
            return True
        if len(deltas) > give_up and sorted(deltas[-give_up:]) == deltas[-give_up:]:
            return False
        if set_width * dpi < min_size_px or set_height * dpi < min_size_px:
            return False

def colorbar(mappable):
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    last_axes = plt.gca()
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="5%", pad=0.05)
    #cbar = fig.colorbar(mappable, cax=cax)
    cax = divider.append_axes("bottom", size="7%", pad="2%")
    cbar = fig.colorbar(mappable, cax=cax, orientation="horizontal")
    plt.sca(last_axes)
    return cbar

def dowork(Z, Zbar, trimesh, r, z, fsum, inode, title, outdir, seq):
    fig = plt.figure(figsize=[12,6])
    
    ax = plt.subplot(1,3,1)
    plt.tricontourf(trimesh, fsum)
    clb = plt.colorbar()
    clb.ax.set_title('log10(max)', fontsize=10)
    plt.axis('scaled')
    plt.axis('off')
    #plt.triplot(trimesh, alpha=0.3, c='0.3')
    plt.scatter(r[inode], z[inode], c='r', marker='x', s=80)
    
    nx, ny = Z.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)

    ax = plt.subplot(1,3,2)
    im = ax.imshow(Z, origin='lower')
    plt.colorbar(im, orientation='horizontal', pad=0.01)
    #colorbar(im)
    plt.axvline(x=nx/2, c='w', alpha=0.3, ls='dashed')
    plt.contour(X, Y, Z, levels=5, colors='w', alpha=0.3, origin='lower')
    plt.title(title)
    plt.axis('scaled')
    plt.axis('off')
    #plt.tight_layout(h_pad=1)

    ax = plt.subplot(1,3,3)
    im = ax.imshow(Zbar, origin='lower')
    plt.colorbar(im, orientation='horizontal', pad=0.01)
    #colorbar(im)
    plt.axvline(x=nx/2, c='w', alpha=0.3, ls='dashed')
    plt.contour(X, Y, Zbar, levels=5, colors='w', alpha=0.3, origin='lower')
    #plt.title(title)
    plt.axis('scaled')
    plt.axis('off')
    
    fname = '%s/%06d.jpg'%(outdir,seq)
    #if inode%100 == 0: print (fname)
    #set_size(fig, (8, 6))
    #plt.savefig(fname, bbox_inches='tight')
    plt.savefig(fname)
    plt.close()
    
    return fname

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('exp', help='exp')
    parser.add_argument('--prefix', help='prefix', default='xgc_images_fluxsurf')
    parser.add_argument('--grey', help='grey', action='store_true')
    parser.add_argument('--onlyn', type=int, help='onlyn', default=10000000)
    parser.add_argument('--nofuture', help='nofuture', action='store_true')
    args = parser.parse_args()

    exp = args.exp #'d3d_coarse_v2_4x'
    print (exp)
    with ad2.open('%s/xgc.mesh.bp'%exp, 'r') as f:
        nnodes = int(f.read('n_n', ))
        #ncells = int(f.read('n_t', ))
        rz = f.read('rz')
        conn = f.read('nd_connect_list')
        #psi = f.read('psi')
        #nextnode = f.read('nextnode')
        #epsilon = f.read('epsilon')
        #node_vol = f.read('node_vol')
        #node_vol_nearest = f.read('node_vol_nearest')
        #psi_surf = f.read('psi_surf')
        surf_idx = f.read('surf_idx')
        surf_len = f.read('surf_len')

    r = rz[:,0]
    z = rz[:,1]
    print (nnodes)

    with ad2.open('%s/restart_dir/xgc.f0.00420.bp'%exp,'r') as f:
        i_f = f.read('i_f')
    i_f = np.moveaxis(i_f,1,2)
    print (i_f.shape)

    with ad2.open('recon.bp','r') as f:
        X0 = f.read('i_f_recon')
    print (X0.shape)

    outdir = '%s-%s'%(args.prefix,exp)
    print ('outdir:', outdir)
    os.makedirs(outdir, exist_ok=True)

    trimesh = tri.Triangulation(r, z, conn)

    if args.grey:    
        for iphi in range(1): #,i_f.shape[0]):
            for inode in range(i_f.shape[1]):
                #print (iphi, inode)
                X  = i_f[iphi,inode,:,:]
                X = (X-X.min())/(X.max()-X.min())*255
                X = X.astype(np.float32).copy()

                im = Image.fromarray(np.uint8(X))
                im = im.resize((256, 256))
                #fname = '%s/%d-%d-%05d.jpg'%(outdir,420,iphi,inode)
                fname = '%s/%06d.jpg'%(outdir,inode)
                im.save(fname)

                if inode%1000 == 0: print (fname)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
            future_list = list()
            seq = 0
            for iphi in range(1): #,i_f.shape[0]):
                #fsum = np.mean(i_f[iphi,:], axis=(1,2))
                fsum = np.log10(np.max(i_f[iphi,:], axis=(1,2)))
                for i in tqdm(range(min(len(surf_idx), args.onlyn))):
                    for j in range(surf_len[i]):
                        inode = surf_idx[i,j]-1
                        Z  = i_f[iphi,inode,:,:]
                        Zbar = X0[iphi,inode,:,:]
                        title = 'node: %d (surfid: %d)'%(inode,i)
                        if not args.nofuture: 
                            future = executor.submit(dowork, Z, Zbar, trimesh, r, z, fsum, inode, title, outdir, seq)
                            future_list.append(future)
                        else:
                            dowork(Z, Zbar, trimesh, r, z, fsum, inode, title, outdir, seq)
                        seq += 1

            if not args.nofuture:
                for future in tqdm(future_list):
                    _ = future.result()
                