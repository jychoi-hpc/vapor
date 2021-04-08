import numpy as np
import adios2 as ad2
import os
import logging
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--exp', help='exp')
parser.add_argument('--istep', help='istep', type=int, default=None)
parser.add_argument('FILES', help='recon files', nargs='+', type=str)
parser.add_argument('--output', help='exp', default='recon.bp')
args = parser.parse_args()

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(message)s')

expdir = args.exp
istep = args.istep
recon_files = args.FILES

fname = os.path.join(expdir, 'restart_dir/xgc.f0.%05d.bp'%istep)
logging.debug (f"Reading: {fname}")
with ad2.open(fname, 'r') as f:
    Zif = f.read('i_f')

Xif = np.zeros_like(Zif)
for fname in recon_files:
    logging.debug (f"Reading: {fname}")    
    with ad2.open(fname, 'r') as f:
        X0 = f.read('X0')
        zlb = f.read('zlb')
    logging.debug (f"X0, zlb: {X0.shape} {zlb.shape}")
    
    for i,istep,iphi,inode in zlb:
        Xif[iphi,:,inode,:] = X0[i,:,:]

fname = args.output
logging.debug (f"Saving: {fname}")
with ad2.open(fname, 'w') as fw:
    shape = Xif.shape
    start = [0,]*len(shape)
    count = shape
    fw.write('i_f', Xif.copy(), shape, start, count)

print ("Done.")