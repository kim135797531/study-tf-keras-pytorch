# -*- coding: utf-8 -*-
# (0) Random motivation
# 랜덤으로 보상 주기 테스트

import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
import torch.optim as optim

import utils_kdm as u
from algorithm_im.im_base import IntrinsicMotivation
from utils_kdm.trainer_metadata import TrainerMetadata


class RandomMotivation(IntrinsicMotivation):

    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self._set_hyper_parameters()

    def _set_hyper_parameters(self):
        super()._set_hyper_parameters()
        # x = [0, 1) 균등 분포 랜덤
        # reward = ax + b
        self.a = 2
        self.b = 0

    def _train_model(self, s, a, n_s):
        pass

    def intrinsic_motivation_impl(self, i_episode, step, current_sars, current_done):
        # Random motivation
        current_state, current_action, current_reward, current_next_state = current_sars

        intrinsic_reward = torch.rand_like(u.t_float32(current_reward))
        intrinsic_reward = (self.a * intrinsic_reward) + self.b

        return intrinsic_reward
