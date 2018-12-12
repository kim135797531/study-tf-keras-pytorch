# -*- coding: utf-8 -*-

import os
from abc import ABCMeta, abstractmethod
import time
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
import torch.optim as optim

import gym

import utils_kdm as u
from utils_kdm.checkpoint import Checkpoint, TorchSerializable
from utils_kdm.drawer import Drawer
from utils_kdm.noise import OrnsteinUhlenbeckNoise
from utils_kdm.replay_memory import ReplayMemory
from utils_kdm import global_device
from utils_kdm import drawer


class IntrinsicMotivation(TorchSerializable):
    __metaclass__ = ABCMeta

    def __init__(self, state_size, action_size, device, viz=None):
        self._set_hyper_parameters()
        self.device = device
        self.viz = viz

        self.state_size, self.action_size = state_size, action_size

    def _set_hyper_parameters(self):
        # TODO: 지연된 시작?
        self.delayed_start = False
        self.intrinsic_reward_start = 3000

        # TODO: 가중치 조절,
        # TODO: 근데 어차피 내발적 동기에 파라미터 C를 곱해줄 거면 굳이 가중합을 해 줄 필요가 있나?
        # 0 = 내적 동기 0% (순수 외적 동기)
        # 0.5 = 내적/외적 반반 (균등 분배)
        # 1 = 내적 동기 100% (순수 내적 동기)
        self.intrinsic_reward_ratio = 0.5

    def state_dict_impl(self):
        return {}

    def load_state_dict_impl(self, var_state):
        pass

    def weighted_reward_batch(self, intrinsic, extrinsic):
        # Oudeyer는 내적 동기랑 환경으로부터의 보상이랑 합칠 때 가중합을 제안
        # 파라미터(비율)는 미제시
        if self.intrinsic_reward_ratio == 0:
            return extrinsic
        elif self.intrinsic_reward_ratio == 1:
            return intrinsic
        else:
            return self.intrinsic_reward_ratio * intrinsic + \
                   (1 - self.intrinsic_reward_ratio) * extrinsic

    @abstractmethod
    def intrinsic_motivation_impl(self, i_episode, step, transitions, state_batch, action_batch, next_state_batch):
        raise NotImplementedError("Please implement this method.")

    def intrinsic_motivation(self, i_episode, step, transitions, state_batch, action_batch, next_state_batch):
        intrinsic_reward_batch = self.intrinsic_motivation_impl(i_episode, step, transitions, state_batch, action_batch, next_state_batch)
        return intrinsic_reward_batch

    def get_reward(self, i_episode, step, transitions, state_batch, action_batch, next_state_batch):
        if self.intrinsic_reward_ratio == 0:
            return torch.zeros_like(state_batch).to(self.device)

        return self.intrinsic_motivation(i_episode, step, transitions, state_batch, action_batch, next_state_batch)
