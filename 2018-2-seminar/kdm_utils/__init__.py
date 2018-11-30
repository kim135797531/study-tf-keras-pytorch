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


def copy_without_bias(src_nn, dst_nn, tau=1.0):
    src_state_dict = src_nn.state_dict()
    dst_state_dict = dst_nn.state_dict()
    for k in dst_state_dict.keys():
        if 'weight' in k:
            src_weight, dst_weight = src_state_dict[k], dst_state_dict[k]
            dst_state_dict[k] = tau * src_weight + (1.0 - tau) * dst_weight
        # TODO: 없애야함
        # if 'bias' in k:
        #     src_weight, dst_weight = src_state_dict[k], dst_state_dict[k]
        #     dst_state_dict[k] = tau * src_weight + (1.0 - tau) * dst_weight
    dst_nn.load_state_dict(dst_state_dict)
    del dst_state_dict
