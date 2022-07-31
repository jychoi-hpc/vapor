import logging
import pathlib
import os

import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import numpy as np

from .config import getconf


def log(*args, sep=" "):
    """
    Helper function to print/log messages.
    rank parameter is to limit which rank should print. if rank is None, all processes print.
    """
    logger = logging.getLogger("vapor")
    logger.info(sep.join(map(str, args)))


def log0(*args, sep=" "):
    """
    Helper function to print/log messages.
    rank parameter is to limit which rank should print. if rank is None, all processes print.
    """
    if getconf("rank", 0) == 0:
        logger = logging.getLogger("vapor")
        logger.info(sep.join(map(str, args)))


def setup_log(prefix, rank):
    """
    Setup logging to print messages for both screen and file.
    """

    fmt = "%d: %%(message)s" % (rank)
    logFormatter = logging.Formatter(fmt)

    logger = logging.getLogger("vapor")
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    pathlib.Path(prefix).mkdir(parents=True, exist_ok=True)
    fname = os.path.join(prefix, "run.log")
    fileHandler = logging.FileHandler(fname)

    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)


def print_model(model):
    """print model's parameter size layer by layer"""
    log(type(model.module).__name__)
    log("-" * 50)
    num_params = 0
    for k, v in model.state_dict().items():
        log("%50s\t%20s\t%10d" % (k, list(v.shape), v.numel()))
        num_params += v.numel()
    log("-" * 50)
    log("%50s\t%20s\t%10d" % ("Total", "", num_params))
    log("All (total, bytes): %g %g" % (num_params, num_params * 4))


def plot_one(
    inputs,
    originals,
    reconstructions,
    istep=None,
    prefix=None,
    scale_each=False,
):
    plt.figure(99, figsize=[12, 4])
    x = inputs
    y = originals
    z = reconstructions

    out0 = make_grid(
        x[:8, ...].cpu().data, ncol=4, normalize=True, scale_each=scale_each, padding=0
    )
    out1 = make_grid(
        y[:8, ...].cpu().data, ncol=4, normalize=True, scale_each=scale_each, padding=0
    )
    out2 = make_grid(
        z[:8, ...].cpu().data, ncol=4, normalize=True, scale_each=scale_each, padding=0
    )
    out3 = make_grid(
        (y[:8, ...] - z[:8, ...]).abs().cpu().data,
        ncol=4,
        normalize=True,
        scale_each=scale_each,
        padding=0,
    )

    out = torch.cat((out0, out1, out2, out3), dim=1)
    out = out.numpy()
    out = np.transpose(out, (1, 2, 0))

    plt.imshow(out[..., 0])
    plt.axis("off")
    plt.title("epoch: %d" % istep)
    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    if prefix is not None:
        path_name = os.path.join(prefix, "img-%d.jpg" % istep)
        plt.savefig(path_name)
