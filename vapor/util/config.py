from typing import *


def initconf(config):
    globals().update({"vapor_config": config})


def setconf(key, val):
    ## TODO: make hierarchical
    config = globals()["vapor_config"]
    config[key] = val


def getconf(key, default=None):
    config = globals()["vapor_config"]
    return config.get(key, default)


def dget(d: Dict, key, default=None):
    if key not in d:
        d[key] = default
    return d[key]
