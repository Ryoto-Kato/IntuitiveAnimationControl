import re
import numpy as np
import torch
from .metadata_loader import load_RT

def vecnorm(vector):
    """return L2 norm (vector length) along the last axis
        Compute the length of the vector
    """
    return np.sqrt(np.sum(vector**2, axis = -1))

def normalized(vector):
    """normalize array of vector along the last axis"""
    return vector/vecnorm(vector[:, None])

def vecnorm_tensor(vector):
    """return L2 norm (vector length) along the last axis
        Compute the length of the vector
    """
    return torch.sqrt(torch.sum(vector**2, dim = -1))

def normalized_tensor(vector):
    """normalize array of vector along the last axis"""
    return vector/vecnorm_tensor(vector[:, None])

def project_weight(x):
    x = np.maximum(0., x)
    max_x = x.max()
    if max_x == 0:
        return x
    else:
        return x/max_x

def proxy_l1l2(Lambda, x, beta):
    xlen = np.sqrt((x**2).sum(axis = -1))
    with np.errstate(divide='ignore'): #floating-point error handling (ignore error (dividivion by 0))
        shrinkage = np.maximum(0.0, 1-beta*Lambda/xlen)
    return (x*shrinkage[..., np.newaxis])

def uptoscale_transform(path2txt):
    cano2world = np.zeros((4, 4))
    cano2world[3, 3] = 1.0
    cano2world[:3, :] = load_RT(path2txt)
    cano2world = upscale_rot(cano2world)
    return cano2world

def upscale_rot(mat4x4):
    Uc, Sc, Vhc = np.linalg.svd(mat4x4[:3, :3])
    mat4x4[:3, :3] = np.matmul(Uc, Vhc)
    return mat4x4