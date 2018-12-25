# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from utils_kdm.manage_device import ManageDevice


#####################
# Torch 관련
#####################
# 머신 숫자 표현 최저값

def get_np_float32_eps():
    return np.finfo(np.float32).eps.item()


# 저장 가능한 클래스는 이 추상클래스를 구현하도록 강제하기
class TorchSerializable(object):
    __metaclass__ = ABCMeta

    def __init__(self):
        self._registered_variables = list()

    def register_serializable(self, variables):
        for k in variables:
            if 'self.' in k:
                k = k[5:]
            self._registered_variables.append(k)

    def unregister_serializable(self, variables):
        for k in variables:
            if 'self.' in k:
                k = k[5:]
            if k in self._registered_variables:
                self._registered_variables.remove(k)

    def state_dict(self):
        ret = dict()
        for k in self._registered_variables:
            variable = getattr(self, k, None)
            if variable is None:
                print('state_dict(): Skipping not registered variable: {}'.format(k))
            state_dict_method = getattr(variable, "state_dict", None)
            ret[k] = variable.state_dict() if state_dict_method else variable

        return ret

    def load_state_dict(self, var_state):
        for k in self._registered_variables:
            if k not in var_state:
                print('load_state_dict(): Skipping not registered variable: {}'.format(k))
                continue

            variable = getattr(self, k, None)
            if variable is None:
                print('load_state_dict(): Skipping not registered variable: {}'.format(k))
            state_dict_method = getattr(variable, "state_dict", None)
            if state_dict_method:
                variable.load_state_dict(var_state[k])
            else:
                setattr(self, k, var_state[k])


#####################
# Torch 텐서 관련
#####################
def t_uint8(item, device=None):
    device = device if device else ManageDevice().get()
    if isinstance(item, np.ndarray):
        return t_from_np_to_uint8(item, device)
    else:
        return torch.tensor([item], device=device, dtype=torch.uint8)


def t_float32(item, device=None):
    device = device if device else ManageDevice().get()
    if isinstance(item, np.ndarray):
        return t_from_np_to_float32(item, device)
    else:
        return torch.tensor([item], device=device, dtype=torch.float32)


def t_long(item, device=None):
    device = device if device else ManageDevice().get()
    if isinstance(item, np.ndarray):
        return t_from_np_to_long(item, device)
    else:
        return torch.tensor([item], device=device, dtype=torch.long)


def t_from_np_to_uint8(item, device=None):
    device = device if device else ManageDevice().get()
    return torch.from_numpy(item).int().to(device)


def t_from_np_to_float32(item, device=None):
    device = device if device else ManageDevice().get()
    return torch.from_numpy(item).float().to(device)


def t_from_np_to_long(item, device=None):
    device = device if device else ManageDevice().get()
    return torch.from_numpy(item).long().to(device)


def maybe_float(item):
    # 텐서가 들어오든 numpy float가 들어오든 그냥 float가 들어오든 무조건 float로 반환
    """
    if isinstance(item, torch.Tensor):
        item = item.item()
    elif isinstance(item, np.floating):
        item = item.tolist()
    """
    return float(item)


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
