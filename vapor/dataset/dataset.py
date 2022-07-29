import torch
import numpy as np
import adios2 as ad2
import os

from vapor.util.logging import log, log0


class XGC_F0_Dataset(torch.utils.data.Dataset):
    """XGC F0 dataset for pytorch"""

    def __init__(
        self,
        prefix,
        prefix2,
        istep,
        iphi=None,
        inode=0,
        nnodes=None,
        normalize=False,
        transform=None,
    ):
        """
        Args:
        """

        self.transform = transform

        def adios2_get_shape(f, varname):
            nstep = int(f.available_variables()[varname]["AvailableStepsCount"])
            shape = f.available_variables()[varname]["Shape"]
            lshape = None
            if shape == "":
                ## Accessing Adios1 file
                ## Read data and figure out
                v = f.read(varname)
                lshape = v.shape
            else:
                lshape = tuple([int(x.strip(",")) for x in shape.strip().split()])
            return (nstep, lshape)

        def read_f0(prefix):
            fname = os.path.join(prefix, "xgc.f0.%05d.bp" % istep)
            with ad2.open(fname, "r") as f:
                nstep, nsize = adios2_get_shape(f, "i_f")
                ndim = len(nsize)
                nphi = nsize[0] if iphi is None else 1
                _iphi = 0 if iphi is None else iphi
                _nnodes = nsize[2] - inode if nnodes is None else nnodes
                nmu = nsize[1]
                nvp = nsize[3]
                start = (_iphi, 0, inode, 0)
                count = (nphi, nmu, _nnodes, nvp)
                log0(f"Reading: {start} {count}")
                i_f = f.read("i_f", start=start, count=count)

                li = list(range(inode, inode + _nnodes))
                lb = np.array(li, dtype=np.int32)

            Z0 = np.moveaxis(i_f, 1, 2)
            Z0 = Z0.reshape((-1, Z0.shape[2], Z0.shape[3]))
            _lb = list()
            for i in range(nphi):
                for k in lb:
                    _lb.append((istep, i, k))
            zlb = np.array(_lb)

            ## Normalize
            # zmin = np.min(Z0, axis=(1,2))
            # zmax = np.max(Z0, axis=(1,2))
            zmin = np.min(Z0)
            zmax = np.max(Z0)

            if normalize:
                # Zif = (Z0 - zmin[:,np.newaxis,np.newaxis])/(zmax-zmin)[:,np.newaxis,np.newaxis]
                Zif = (Z0 - zmin) / (zmax - zmin)
            else:
                Zif = Z0

            return (Zif.astype(np.single), zlb)

        Zif, zlb = read_f0(prefix)
        self.lr = Zif
        self.lb = zlb
        assert len(self.lr) == len(self.lb)

        Hif, _ = read_f0(prefix2)
        self.hr = Hif

        # m = np.mean(self.hr, axis=0)
        # score_list = list()
        # for i in range(0, len(self.hr)):
        #     x = self.hr[i,:,:]
        #     score = ssim(x, m)
        #     score_list.append(score)
        # scores = np.array(score_list)
        # od = scores<0.8

        # self.lr = self.lr[od,:,:]
        # self.hr = self.hr[od,:,:]
        # self.lb = self.lb[od]

    def __len__(self):
        return len(self.lr)

    def __getitem__(self, idx):

        i = idx

        x = self.lr[i, :]
        y = self.hr[i, :]

        lr = np.stack((x, x, x), axis=-1)
        hr = np.stack((y, y, y), axis=-1)
        lb = self.lb[i]

        sample = {"lr": lr, "hr": hr, "lb": lb}

        if self.transform:
            sample = self.transform(sample)

        return sample
