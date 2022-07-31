import os
import sched
import sys
import argparse
from venv import create
import yaml
from mpi4py import MPI

import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

import numpy as np

from torch.utils.data import DataLoader, Subset, DistributedSampler
from sklearn.model_selection import train_test_split

from vapor.util import *
from vapor.model import *
from vapor.dataset.trasnform import Crop, ToTensor
from vapor.dataset import XGC_F0_Dataset

from tqdm import tqdm
import time


def create_model(config):
    model_class = config.get("model_class")
    model_params = config.get("model_params", {})
    model = None
    if model_class == "fc":
        model = FC(**config)
    elif model_class == "fno":
        num_blocks = model_params.get("num_blocks", [3, 4, 23, 3])
        modes = model_params.get("modes", 3)
        model = FNO(num_blocks=num_blocks, modes=modes)
    else:
        raise NotImplementedError
    return model


def train(k, model, loader, optimizer, loss_fn, rank, prefix):
    model.train()

    if getattr(loader.sampler, "set_epoch", None) is not None:
        loader.sampler.set_epoch(k)

    m = len(loader)
    loss_list = list()
    for i, sample in enumerate(loader):
        lr = sample["lr"].to(device)
        hr = sample["hr"].to(device)
        lb = sample["lb"].to(device)

        optimizer.zero_grad()

        recon = model(lr)
        loss = loss_fn(recon, hr)
        loss.backward()
        optimizer.step()

        loss_list.append(loss.item())

    return loss_list, lr, hr, recon


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        initconf(config)

    batch_size = getconf("batch_size", 128)
    start_epoch = getconf("start_epoch", 0)
    num_epochs = getconf("num_epochs", 10)
    lr = getconf("learning_rate", 1.0e-3)
    wdir = getconf("wdir", "wdir")
    jobname = getconf("jobname", "vapor")
    prefix = os.path.join(wdir, jobname)
    restart = getconf("restart", False)
    checkpoint_period = getconf("checkpoint_period", 100)
    log_period = getconf("log_period", 100)
    plot_period = getconf("plot_period", 100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_size, rank = MPI.COMM_WORLD.Get_size(), MPI.COMM_WORLD.Get_rank()

    setconf("prefix", prefix)
    setconf("device", device)
    setconf("rank", rank)
    setconf("world_size", world_size)

    ## setup
    setup_log(prefix, rank)
    setup_ddp(rank, world_size)

    composed = transforms.Compose([Crop([0, 4], [32, 32]), ToTensor()])
    dataset = XGC_F0_Dataset(
        "d3d_coarse_v2/restart_dir",
        "d3d_coarse_v2_4x/restart_dir",
        istep=420,
        iphi=0,
        inode=12000,
        nnodes=1000,
        normalize=True,
        transform=composed,
    )
    log(len(dataset), dataset[0]["lr"].shape, dataset[0]["lr"].dtype)

    train_indices, test_indices, _, _ = train_test_split(
        range(len(dataset)),
        range(len(dataset)),
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    # generate subset based on indices
    training_data = Subset(dataset, train_indices)
    validation_data = Subset(dataset, test_indices)
    log("training_data,validation_data", len(training_data), len(validation_data))

    nc, nh, nw = training_data[0]["lr"].shape
    log("training_data.shape", training_data[0]["lr"].shape)

    ## drop_last in sampler and dataloader is different
    sampler = DistributedSampler(training_data, shuffle=True, seed=42)
    training_loader = DataLoader(
        training_data,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=0,
        sampler=sampler,
        drop_last=True,
    )
    validation_loader = DataLoader(
        validation_data,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
    )

    # for k in range(5):
    #     if getattr(training_loader.sampler, "set_epoch", None) is not None:
    #         training_loader.sampler.set_epoch(k)
    #     for i, sample in enumerate(training_loader):
    #         lr = sample["lr"].to(device)
    #         hr = sample["hr"].to(device)
    #         lb = sample["lb"].to(device)
    #         log(f"[{k}/{i}] lb: {lb.shape} {lb[:,2]}")

    in_channels = dataset[0]["lr"].shape[0]
    out_channels = dataset[0]["hr"].shape[0]

    model = create_model(config)
    model = model.to(device)
    model = DDP(model)
    if rank == 0:
        print_model(model)

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    if restart:
        fname = "model-%d" % (start_epoch)
        load_model(model, prefix, fname, device=device, optimizer=optimizer)
        log0("load model", fname)

    step_size = 10
    gamma = 0.1
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=100, verbose=True
    )
    loss_fn = nn.MSELoss()

    train_loss = list()
    t0 = time.time()
    for k in range(start_epoch, start_epoch + num_epochs):
        batch_loss, lr, hr, recon = train(
            k, model, training_loader, optimizer, loss_fn, rank, prefix
        )

        bx = np.mean(batch_loss)
        if scheduler is not None:
            scheduler.step(bx)

        if (k + 1) % log_period == 0 and rank == 0:
            log(
                "Epoch %d loss,lr: %g %g %g"
                % (k, bx, optimizer.param_groups[0]["lr"], time.time() - t0)
            )

        if (k + 1) % plot_period == 0 and rank == 0:
            plot_one(lr, hr, recon, istep=k + 1, scale_each=False, prefix=prefix)

        if (
            (k + 1) % checkpoint_period == 0
            or (k + 1) == start_epoch + num_epochs
            and rank == 0
        ):
            fname = "model-%d" % (k + 1)
            save_model(model, optimizer, prefix, fname)
            log("Save model:", fname)

    log("Done.")
