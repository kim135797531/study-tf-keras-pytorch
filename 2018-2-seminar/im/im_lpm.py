# -*- coding: utf-8 -*-

import os
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
from im.region import RegionManager
from utils_kdm.checkpoint import Checkpoint, TorchSerializable
from utils_kdm.drawer import Drawer
from utils_kdm.noise import OrnsteinUhlenbeckNoise
from utils_kdm.replay_memory import ReplayMemory

from im.im_base import IntrinsicMotivation
from utils_kdm import global_device


class StatePredictor(nn.Module):

    def __init__(self, state_size, action_size):
        super(StatePredictor, self).__init__()
        # TODO: 상태 예측은 망 별로 안 커도 학습될 듯? (상태 예측만 테스트 해 보기)
        # 내발적 동기를 위해서 상태를 예측한다는 개념 = 2007년 Oudeyer 논문을 참조한 것
        sensorimotor_size = state_size + action_size
        self.layer_sizes = [sensorimotor_size, 32, 16, state_size]

        self.linear1 = nn.Linear(self.layer_sizes[0], self.layer_sizes[1])
        self.linear2 = nn.Linear(self.layer_sizes[1], self.layer_sizes[2])
        self.head = nn.Linear(self.layer_sizes[2], self.layer_sizes[3])
        u.fanin_init(self.linear1.weight)
        u.fanin_init(self.linear2.weight)
        nn.init.uniform_(self.head.weight, a=-3*10e-4, b=3*10e-4)

    def forward(self, state, action):
        # 그냥 일렬로 합쳐기
        # Oudeyer (2007)
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.head(x)


class LearningProgressMotivation(IntrinsicMotivation):

    def __init__(self, state_size, action_size, device, viz=None):
        super().__init__(state_size, action_size, device, viz)
        self._set_hyper_parameters()
        self.region_manager = RegionManager(self.state_size, self.action_size)

    def _set_hyper_parameters(self):
        super()._set_hyper_parameters()

    def state_dict_impl(self):
        todo = super().state_dict_impl()
        # TODO: 저장 불러오기
        todo = {
        }
        return todo

    def load_state_dict_impl(self, var_state):
        # TODO: 저장 불러오기
        super().load_state_dict(var_state)

    def intrinsic_motivation_impl(self, i_episode, step, transitions, state_batch, action_batch, next_state_batch):
        # Predictive novelty motivation (NM)
        for transition in transitions:
            # TODO: 속도..
            examplar = self.region_manager.exemplar_structure(
                transition.state, transition.action, transition.next_state
            )
            self.region_manager.add(examplar)

        intrinsic_reward_batch = torch.zeros(128).to(self.device)

        # TODO: 환경 평소 보상 (1) 정도로 clip 해줄까?
        # intrinsic_reward_batch = torch.clamp(intrinsic_reward_batch, min=-2, max=2)
        # self.viz.draw_line(y=torch.mean(intrinsic_reward_batch), interval=1000, name="intrinsic_reward_batch")

        return intrinsic_reward_batch
