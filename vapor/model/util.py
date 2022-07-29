
import os
import torch
from collections import OrderedDict

def save_model(model, optimizer, prefix, filename):
    """Save both model and optimizer state in a single checkpoint file"""
    path_name = os.path.join(prefix, filename + ".pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path_name,
    )

def load_model(model, prefix, filename, device=None, optimizer=None):
    """Load both model and optimizer state from a single checkpoint file"""
    path_name = os.path.join(prefix, filename + ".pt")
    # map_location = {"cuda:%d" % 0: get_device_name()}
    # print_master("Load existing model:", path_name)
    checkpoint = torch.load(path_name, map_location=None)
    state_dict = checkpoint["model_state_dict"]
    ## To be compatible with old checkpoint which was not written as a ddp model
    if not next(iter(state_dict)).startswith("module"):
        ddp_state_dict = OrderedDict()
        for k, v in state_dict.items():
            k = "module." + k
            ddp_state_dict[k] = v
        state_dict = ddp_state_dict
    model.load_state_dict(state_dict)
    if (optimizer is not None) and ("optimizer_state_dict" in checkpoint):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
