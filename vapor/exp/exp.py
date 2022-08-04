from ..util import *
from ..model import *

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.optim as optim


def create_model(config):
    name = dget(config, "model_class")
    params = dget(config, "model_params", {})
    obj = None
    if name == "fc":
        obj = FC(**config)
    elif name == "fno":
        num_blocks = dget(params, "num_blocks", [3, 4, 23, 3])
        modes = dget(params, "modes", 3)
        obj = FNO(num_blocks=num_blocks, modes=modes)
    elif name == "f2f":
        in_channels = dget(params, "in_channels", 3)
        out_channels = dget(params, "out_channels", 3)
        num_hiddens = dget(params, "num_hiddens", 64)
        num_residual_layers = dget(params, "num_residual_layers", 32)
        ks = dget(params, "ks", [9, 3, 1])
        obj = F2F(in_channels, out_channels, num_hiddens, num_residual_layers, ks)
    elif name == "vqvae":
        in_channels = dget(params, "in_channels", 3)
        out_channels = dget(params, "out_channels", 3)
        num_hiddens = dget(params, "num_hiddens", 128)
        num_residual_layers = dget(params, "num_residual_layers", 16)

        num_residual_hiddens = dget(params, "num_residual_hiddens", 32)
        num_embeddings = dget(params, "num_embeddings", 512)
        embedding_dim = dget(params, "embedding_dim", 64)
        commitment_cost = dget(params, "commitment_cost", 0.25)
        decay = dget(params, "decay", 0.0)
        obj = VQVAE(
            in_channels,
            out_channels,
            num_hiddens,
            num_residual_layers,
            num_residual_hiddens,
            num_embeddings,
            embedding_dim,
            commitment_cost,
            decay,
        )
    else:
        raise NotImplementedError

    return obj


def create_opimizer(model, config):
    name = dget(config, "optimizer_class", "SGD")
    params = dget(config, "optimizer_params", {})
    obj = None
    if name == "SGD":
        lr = dget(params, "lr", 1.0e-3)
        momentum = dget(params, "momentum", 0.9)
        obj = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    else:
        raise NotImplementedError

    return obj


def create_scheduler(optimizer, config):
    name = dget(config, "scheduler_class", "ReduceLROnPlateau")
    params = dget(config, "scheduler_params", {})
    obj = None
    if name == "ReduceLROnPlateau":
        patience = dget(params, "patience", 1000)
        obj = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, "min", patience=patience, verbose=True
        )
    elif name == "StepLR":
        step_size = dget(params, "step_size", 1000)
        gamma = dget(params, "gamma", 0.1)
        obj = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=gamma, verbose=True
        )
    else:
        raise NotImplementedError

    return obj


class Exp:
    def __init__(self, config, device):
        self.model = create_model(config)
        self.model = self.model.to(device)
        self.model = DDP(self.model, find_unused_parameters=False)

        self.optimizer = create_opimizer(self.model, config)
        self.scheduler = create_scheduler(self.optimizer, config)

        self.train_loader = None
        self.validation_loader = None
        self.test_loader = None
        self.loss_fn = None
        self.device = device

    def train(self, k):
        self.model.train()

        if getattr(self.train_loader.sampler, "set_epoch", None) is not None:
            self.train_loader.sampler.set_epoch(k)

        m = len(self.train_loader)
        loss_list = list()
        for i, sample in enumerate(self.train_loader):
            lr = sample["lr"].to(self.device)
            hr = sample["hr"].to(self.device)
            lb = sample["lb"].to(self.device)

            self.optimizer.zero_grad()

            recon = self.model(lr)
            loss = self.loss_fn(recon, hr)
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.item())

        return loss_list, lr, hr, recon


class Exp2(Exp):
    def __init__(self, config, device):
        super().__init__(config, device)

    def train(self, k):
        self.model.train()

        if getattr(self.train_loader.sampler, "set_epoch", None) is not None:
            self.train_loader.sampler.set_epoch(k)

        m = len(self.train_loader)
        loss_list = list()
        for i, sample in enumerate(self.train_loader):
            lr = sample["lr"].to(self.device)
            hr = sample["hr"].to(self.device)
            lb = sample["lb"].to(self.device)

            self.optimizer.zero_grad()

            vq_loss, recon, perplexity = self.model(lr)
            recon_error = self.loss_fn(recon, hr)
            loss = recon_error + vq_loss
            loss.backward()
            self.optimizer.step()

            loss_list.append(loss.item())

            if (k + 1) % 10 == 0:
                log("recon_errror,vq_loss", recon_error.item(), vq_loss.item())

        return loss_list, lr, hr, recon
