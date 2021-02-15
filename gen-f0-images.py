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

def xdowork(Z, Zbar, trimesh, r, z, fsum, inode, title, outdir, seq):
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
    ## (2021/02) not working
    # plt.savefig(fname, bbox_inches='tight', pad_inches=1.0) 
    plt.savefig(fname)
    plt.close()
    
    return fname

def dowork(Z0, Z1, N0, N1, T0, T1, trimesh, r, z, fsum, inode, title, outdir, seq, save=True):
    fig = plt.figure(figsize=[12,8], constrained_layout=False)
    fig.suptitle(title, fontsize=14, y=0.97)
    gs = fig.add_gridspec(2, 4, width_ratios=[1,1,1,1])

    ax = fig.add_subplot(gs[:, 0])
    im = plt.tricontourf(trimesh, fsum)
    divider = make_axes_locatable(ax)
    #cax = divider.append_axes("right", size="10%", pad=0.05)
    #clb = plt.colorbar(im, cax=cax)
    clb = plt.colorbar(im, orientation='horizontal', pad=0.01)
    clb.ax.set_title('log10(max)', fontsize=10)
    clb.ax.set_xticklabels(clb.ax.get_xticklabels(),rotation=90)
    plt.axis('scaled')
    plt.axis('off')
    #plt.triplot(trimesh, alpha=0.3, c='0.3')
    plt.scatter(r[inode], z[inode], c='r', marker='x', s=80)

    nx, ny = Z0.shape
    x = np.arange(nx)
    y = np.arange(ny)
    X, Y = np.meshgrid(x, y)
    
    ax = fig.add_subplot(gs[0, 1])
    im = plt.imshow(Z0, origin='lower')
    plt.colorbar(im, orientation='horizontal', pad=0.01)
    plt.axvline(x=nx/2, c='w', alpha=0.3, ls='dashed')
    plt.contour(X, Y, Z0, levels=5, colors='w', alpha=0.3, origin='lower')
    plt.title('f0')
    plt.axis('scaled')
    plt.axis('off')

    ax = fig.add_subplot(gs[0, 3])
    im = plt.imshow(N0, origin='lower')
    plt.colorbar(im, orientation='horizontal', pad=0.01)
    plt.axvline(x=nx/2, c='w', alpha=0.3, ls='dashed')
    #plt.contour(X, Y, N0, levels=5, colors='w', alpha=0.3, origin='lower')
    plt.title('f0 nonadia n0')
    plt.axis('scaled')
    plt.axis('off')

#     ax = fig.add_subplot(gs[0, 3])
#     im = plt.imshow(T0, origin='lower')
#     plt.colorbar(im, orientation='horizontal', pad=0.01)
#     plt.axvline(x=nx/2, c='w', alpha=0.3, ls='dashed')
#     #plt.contour(X, Y, T0, levels=5, colors='w', alpha=0.3, origin='lower')
#     plt.title('f0 nonadia turb')
#     plt.axis('scaled')
#     plt.axis('off')

    ax = fig.add_subplot(gs[0, 2])
    im = plt.imshow(T0, origin='lower')
    plt.colorbar(im, orientation='horizontal', pad=0.01)
    plt.axvline(x=nx/2, c='w', alpha=0.3, ls='dashed')
    #plt.contour(X, Y, T0, levels=5, colors='w', alpha=0.3, origin='lower')
    plt.title('den')
    plt.axis('scaled')
    plt.axis('off')


    ax = fig.add_subplot(gs[1, 1])
    im = plt.imshow(Z1, origin='lower')
    plt.colorbar(im, orientation='horizontal', pad=0.01)
    plt.axvline(x=nx/2, c='w', alpha=0.3, ls='dashed')
    plt.contour(X, Y, Z1, levels=5, colors='w', alpha=0.3, origin='lower')
    plt.title('f0 (recon)')
    plt.axis('scaled')
    plt.axis('off')

    ax = fig.add_subplot(gs[1, 3])
    im = plt.imshow(N1, origin='lower')
    plt.colorbar(im, orientation='horizontal', pad=0.01)
    plt.axvline(x=nx/2, c='w', alpha=0.3, ls='dashed')
    #plt.contour(X, Y, N1, levels=5, colors='w', alpha=0.3, origin='lower')
    plt.title('f0 nonadia n0 (recon)')
    plt.axis('scaled')
    plt.axis('off')

#     ax = fig.add_subplot(gs[1, 3])
#     im = plt.imshow(T1, origin='lower')
#     plt.colorbar(im, orientation='horizontal', pad=0.01)
#     plt.axvline(x=nx/2, c='w', alpha=0.3, ls='dashed')
#     #plt.contour(X, Y, T1, levels=5, colors='w', alpha=0.3, origin='lower')
#     plt.title('f0 nonadia turb (recon)')
#     plt.axis('scaled')
#     plt.axis('off')

    ax = fig.add_subplot(gs[1, 2])
    im = plt.imshow(T1, origin='lower')
    plt.colorbar(im, orientation='horizontal', pad=0.01)
    plt.axvline(x=nx/2, c='w', alpha=0.3, ls='dashed')
    #plt.contour(X, Y, T1, levels=5, colors='w', alpha=0.3, origin='lower')
    plt.title('den (recon)')
    plt.axis('scaled')
    plt.axis('off')


    fname = '%s/%06d.jpg'%(outdir,seq)
    #if inode%100 == 0: print (fname)
    #set_size(fig, (8, 6))
    #plt.savefig(fname, bbox_inches='tight')
    ## (2021/02) not working
    # plt.savefig(fname, bbox_inches='tight', pad_inches=1.0) 
    if save:
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
    parser.add_argument('--nworkers', type=int, help='nworkers', default=32)
    args = parser.parse_args()

    exp = args.exp #'d3d_coarse_v2_4x'
    print (exp)
    with ad2.open('%s/xgc.mesh.bp'%exp, 'r') as f:
        nnodes = int(f.read('n_n', ))
        #ncells = int(f.read('n_t', ))
        rz = f.read('rz')
        conn = f.read('nd_connect_list')
        #psi = f.read('psi')
        nextnode = f.read('nextnode')
        #epsilon = f.read('epsilon')
        #node_vol = f.read('node_vol')
        #node_vol_nearest = f.read('node_vol_nearest')
        #psi_surf = f.read('psi_surf')
        surf_idx = f.read('surf_idx')
        surf_len = f.read('surf_len')

    r = rz[:,0]
    z = rz[:,1]
    trimesh = tri.Triangulation(r, z, conn)
    print (nnodes)

    bl = np.zeros_like(nextnode, dtype=bool)
    for i in range(len(surf_len)):
        n = surf_len[i]
        k = surf_idx[i,:n]-1
        for j in k:
            bl[j] = True

    not_in_surf=np.arange(len(nextnode))[~bl]

#     with ad2.open('%s/restart_dir/xgc.f0.00420.bp'%exp,'r') as f:
#         i_f = f.read('i_f')
#     i_f = np.moveaxis(i_f,1,2)
#     print (i_f.shape)

#     with ad2.open('%s-recon.bp'%exp,'r') as f:
#         X0 = f.read('i_f_recon')
#     print (X0.shape)

    with ad2.open('%s/restart_dir/xgc.f0.%05d.bp'%(exp, 420), 'r') as f:
        i_f = f.read('i_f')
    f0_f = np.moveaxis(i_f, 2, 1).copy()

    with ad2.open('%s-recon.bp'%exp, 'r') as f:
        f0_g = f.read('i_f_recon')
    print (f0_f.shape, f0_g.shape)

    with ad2.open('%s-physics1.bp'%exp, 'r') as f:
        den_f = f.read('den_f')
        den_g = f.read('den_g')

    with ad2.open('%s-physics3.bp'%exp, 'r') as f:
        fn_n0_all_f = f.read('fn_n0_all_f')
        fn_n0_all_g = f.read('fn_n0_all_g')
        fn_turb_all_f = f.read('fn_turb_all_f')
        fn_turb_all_g = f.read('fn_turb_all_g')
        
    print (den_f.shape, fn_n0_all_f.shape, fn_turb_all_f.shape)

    outdir = '%s-%s'%(args.prefix,exp)
    print ('outdir:', outdir)
    os.makedirs(outdir, exist_ok=True)

    if args.grey:    
        for iphi in range(1): #,f0_f.shape[0]):
            for inode in range(f0_f.shape[1]):
                #print (iphi, inode)
                X  = f0_f[iphi,inode,:,:]
                X = (X-X.min())/(X.max()-X.min())*255
                X = X.astype(np.float32).copy()

                im = Image.fromarray(np.uint8(X))
                im = im.resize((256, 256))
                #fname = '%s/%d-%d-%05d.jpg'%(outdir,420,iphi,inode)
                fname = '%s/%06d.jpg'%(outdir,inode)
                im.save(fname)

                if inode%1000 == 0: print (fname)
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.nworkers) as executor:
            future_list = list()
            seq = 0
            for iphi in range(1): #,f0_f.shape[0]):
                #fsum = np.mean(f0_f[iphi,:], axis=(1,2))
                fsum = np.log10(np.max(f0_f[iphi,:], axis=(1,2)))
                for i in tqdm(range(min(len(surf_idx), args.onlyn))):
                    for j in range(surf_len[i]):
                        inode = surf_idx[i,j]-1
                        
                        Z0 = f0_f[iphi,inode,:,:]
                        Z1 = f0_g[iphi,inode,:,:]
                        N0 = fn_n0_all_f[iphi,inode,:,:]
                        N1 = fn_n0_all_g[iphi,inode,:,:]
                        #T0 = fn_turb_all_f[iphi,inode,:,:]
                        #T1 = fn_turb_all_g[iphi,inode,:,:]
                        T0 = den_f[iphi,inode,:,:]
                        T1 = den_g[iphi,inode,:,:]


                        title = 'node: %d (surfid: %d)'%(inode,i)
                        if not args.nofuture: 
                            future = executor.submit(dowork, Z0, Z1, N0, N1, T0, T1, trimesh, r, z, fsum, inode, title, outdir, seq)
                            future_list.append(future)
                        else:
                            dowork(Z0, Z1, N0, N1, T0, T1, trimesh, r, z, fsum, inode, title, outdir, seq)
                        seq += 1
                
                ub = min(len(not_in_surf), args.onlyn)
                for inode in tqdm(not_in_surf[:ub]):
                    Z0 = f0_f[iphi,inode,:,:]
                    Z1 = f0_g[iphi,inode,:,:]
                    N0 = fn_n0_all_f[iphi,inode,:,:]
                    N1 = fn_n0_all_g[iphi,inode,:,:]
                    #T0 = fn_turb_all_f[iphi,inode,:,:]
                    #T1 = fn_turb_all_g[iphi,inode,:,:]
                    T0 = den_f[iphi,inode,:,:]
                    T1 = den_g[iphi,inode,:,:]


                    title = 'node: %d (surfid: %d)'%(inode,-1)
                    if not args.nofuture: 
                        future = executor.submit(dowork, Z0, Z1, N0, N1, T0, T1, trimesh, r, z, fsum, inode, title, outdir, seq)
                        future_list.append(future)
                    else:
                        dowork(Z0, Z1, N0, N1, T0, T1, trimesh, r, z, fsum, inode, title, outdir, seq)
                    seq += 1


            if not args.nofuture:
                for future in tqdm(future_list):
                    _ = future.result()
                
