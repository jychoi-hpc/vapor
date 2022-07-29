import torch
import numpy as np


class Crop(object):
    """
    Crop the image in a sample.
    """

    def __init__(self, offset, width):
        o0, o1 = offset
        w0, w1 = width
        self.offset = offset
        self.width = width

    def __call__(self, sample):
        o0, o1 = self.offset
        w0, w1 = self.width

        lr = sample["lr"][o0: o0+w0, o1: o1+w1, :]
        hr = sample["hr"][o0: o0+w0, o1: o1+w1, :]
        lb = sample["lb"]

        return {"lr": lr, "hr": hr, "lb": lb}


class RandomCrop(object):
    """
    Crop randomly the image in a sample.
    """

    def __init__(self, n):
        self.n = n

    def __call__(self, sample):
        h, w, c = sample["lr"].shape
        new_h, new_w = self.n, self.n

        top, left = 0, 0
        if (h - new_h) > 0 and (w - new_w) > 0:
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)

        lr = sample["lr"][top : top + new_h, left : left + new_w, :]
        hr = sample["hr"][top : top + new_h, left : left + new_w, :]
        lb = sample["lb"]

        return {"lr": lr, "hr": hr, "lb": lb}


class ToTensor(object):
    """
    Convert ndarrays in sample to Tensors.
    """

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        lr = sample["lr"].transpose((2, 0, 1))
        hr = sample["hr"].transpose((2, 0, 1))
        lb = sample["lb"]

        return {
            "lr": torch.from_numpy(lr),
            "hr": torch.from_numpy(hr),
            "lb": torch.from_numpy(lb),
        }
