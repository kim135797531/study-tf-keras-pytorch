# -*- coding: utf-8 -*-

import numpy as np
import torch
from visdom import Visdom


#####################
# Torch 관련
#####################
def get_device(force_cpu=False):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.device('cpu') if force_cpu else device


# 머신 숫자 표현 최저값
def get_np_float32_eps():
    return np.finfo(np.float32).eps.item()


#####################
# Torch 텐서 관련
#####################
def t_uint8(item, device=None):
    device = device if device else get_device()
    return torch.tensor([item], device=device, dtype=torch.float32)


def t_float32(item, device=None):
    device = device if device else get_device()
    return torch.tensor([item], device=device, dtype=torch.float32)


def t_long(item, device=None):
    device = device if device else get_device()
    return torch.tensor([item], device=device, dtype=torch.long)


def t_from_np_to_float32(item, device=None):
    device = device if device else get_device()
    return torch.from_numpy(item).float().to(device)


#####################
# Torch 신경망 가중치 초기화 및 조작 관련
#####################
def fanin_init(in_tensor):
    tensor = in_tensor.detach()
    size = tensor.size()
    bound = 1. / np.sqrt(size[0])
    return tensor.uniform_(-bound, bound)


def soft_update_from_to(src_nn, dst_nn, tau=1.0):
    for target_param, param in zip(dst_nn.parameters(), src_nn.parameters()):
        target_param.data.copy_(
            target_param.data * (1.0 - tau) + param.data * tau
        )
