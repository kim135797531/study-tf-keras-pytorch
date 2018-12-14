# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import numpy as np
import torch

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

    @abstractmethod
    def state_dict_impl(self):
        raise NotImplementedError("Please implement this method.")

    def state_dict(self):
        ret = self.state_dict_impl()
        return ret

    @abstractmethod
    def load_state_dict_impl(self, state_dict):
        raise NotImplementedError("Please implement this method.")

    def load_state_dict(self, state_dict):
        ret = self.load_state_dict_impl(state_dict)
        return ret


#####################
# Torch 텐서 관련
#####################
def t_uint8(item, device=None):
    device = device if device else ManageDevice().get()
    return torch.tensor([item], device=device, dtype=torch.float32)


def t_float32(item, device=None):
    device = device if device else ManageDevice().get()
    return torch.tensor([item], device=device, dtype=torch.float32)


def t_long(item, device=None):
    device = device if device else ManageDevice().get()
    return torch.tensor([item], device=device, dtype=torch.long)


def t_from_np_to_float32(item, device=None):
    device = device if device else ManageDevice().get()
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
