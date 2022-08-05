import os
import sys
import argparse
import yaml
from mpi4py import MPI
import numpy as np

import torch
import torchvision.transforms as transforms

from torch.utils.data import DataLoader, Subset, DistributedSampler
from sklearn.model_selection import train_test_split

from vapor.util import *
from vapor.exp import *
from vapor.model import *
from vapor.dataset.trasnform import Crop, ToTensor
from vapor.dataset import XGC_F0_Dataset

from tqdm import tqdm
import time

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        initconf(config)

    batch_size = dget(config, "batch_size", 128)
    start_epoch = dget(config, "start_epoch", 0)
    num_epochs = dget(config, "num_epochs", 10)
    wdir = dget(config, "wdir", "wdir")
    jobname = dget(config, "jobname", "vapor")
    prefix = os.path.join(wdir, jobname)
    restart = dget(config, "restart", False)
    checkpoint_period = dget(config, "checkpoint_period", 100)
    log_period = dget(config, "log_period", 100)
    plot_period = dget(config, "plot_period", 100)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    world_size, rank = MPI.COMM_WORLD.Get_size(), MPI.COMM_WORLD.Get_rank()

    setconf("prefix", prefix)
    setconf("device", device)
    setconf("rank", rank)
    setconf("world_size", world_size)

    ## setup
    setup_log(prefix, rank)
    setup_ddp(rank, world_size)

    dataset_params = dget(config, "dataset_params", {})
    iphi = dget(dataset_params, "iphi", 0)
    inode = dget(dataset_params, "inode", 12000)
    nnodes = dget(dataset_params, "nnodes", 2000)
    composed = transforms.Compose([Crop([0, 4], [32, 32]), ToTensor()])
    dataset = XGC_F0_Dataset(
        "d3d_coarse_v2/restart_dir",
        "d3d_coarse_v2_4x/restart_dir",
        istep=420,
        iphi=iphi,
        inode=inode,
        nnodes=nnodes,
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

    ## variance sigma^2 (numpy version): (\sum x^2)/N - ((\sum x)/N)^2
    ## Unbiased variance (torch version): (N-1)/N sigma^2
    s, ss, cnt = 0, 0, 0
    for i, sample in enumerate(training_loader):
        hr = sample["hr"]
        ss += (hr ** 2).sum().item()
        s += (hr).sum().item()
        cnt += hr.numel()

    ## Aggregate
    x = torch.tensor([s, ss, cnt]).to(device)
    torch.distributed.all_reduce(x, op=torch.distributed.ReduceOp.SUM)

    s, ss, cnt = x[0].item(), x[1].item(), x[2].item()
    data_variance = (ss / cnt - (s / cnt) ** 2) * (cnt - 1) / cnt
    log("data_variance", data_variance)

    if config["model_class"] == "vqvae":
        exp = Exp2(config, device, data_variance)
    else:
        exp = Exp(config, device)
    exp.train_loader = training_loader
    exp.validation_loader = validation_loader
    exp.loss_fn = torch.nn.MSELoss()

    if rank == 0:
        print_model(exp.model)

    if restart:
        fname = "model-%d" % (start_epoch)
        load_model(exp.model, prefix, fname, device=device, optimizer=exp.optimizer)
        log0("load model", fname)

    if rank == 0:
        with open(os.path.join(prefix, "config.yaml"), "w") as f:
            yaml.dump(config, f)

    train_loss = list()
    train_step = list()
    t0 = time.time()
    for k in range(start_epoch, start_epoch + num_epochs):
        batch_loss, lr, hr, recon = exp.train(k)

        bx = np.mean(batch_loss)
        train_loss.append(bx)
        train_step.append(k + 1)
        if exp.scheduler is not None:
            exp.scheduler.step(bx)

        if (k + 1) % log_period == 0 and rank == 0:
            log(
                "Epoch %d loss,lr: %g %g %g"
                % (k + 1, bx, exp.optimizer.param_groups[0]["lr"], time.time() - t0)
            )

        if (k + 1) % plot_period == 0 and rank == 0:
            plot_one(lr, hr, recon, istep=k + 1, scale_each=False, prefix=prefix)
            plot_loss(train_step, train_loss, istep=k + 1, prefix=prefix)

        if (k + 1) % checkpoint_period == 0 and rank == 0:
            fname = "model-%d" % (k + 1)
            save_model(exp.model, exp.optimizer, prefix, fname)
            log("Save model:", fname)

    log("Done.")
