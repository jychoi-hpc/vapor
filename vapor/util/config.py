
def setconf(key, val):
    ## TODO: make hierarchical
    config = globals()['vaporconf']
    config[key] = val

def initconf(config):
    globals().update({'vaporconf': config})

def getconf(key, default=None):
    ans = default
    try:
        config = globals()['vaporconf']
        if key in config:
            ans = config[key]
    except:
        pass

    return ans
