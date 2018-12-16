# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod

import torch

from utils_kdm import TorchSerializable
from utils_kdm.trainer_metadata import TrainerMetadata


# noinspection PyPep8Naming


class IntrinsicMotivation(TorchSerializable):
    __metaclass__ = ABCMeta

    def __init__(self, state_size, action_size):
        self._set_hyper_parameters()
        self.device = TrainerMetadata().device

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
        self.intrinsic_reward_ratio = 0.9
        self.intrinsic_reward_ratio_annealing = True
        self.intrinsic_reward_ratio_decay = 0.999
        self.intrinsic_reward_ratio_min = 0.001

    def state_dict_impl(self):
        return {}

    def load_state_dict_impl(self, var_state):
        pass

    def scale_annealing(self):
        if self.intrinsic_reward_ratio_annealing and \
                self.intrinsic_reward_ratio > self.intrinsic_reward_ratio_min:
            self.intrinsic_reward_ratio *= self.intrinsic_reward_ratio_decay

    def weighted_reward_batch(self, intrinsic, extrinsic):
        # Oudeyer는 내적 동기랑 환경으로부터의 보상이랑 합칠 때 가중합을 제안
        # 파라미터(비율)는 미제시
        if self.intrinsic_reward_ratio == 0:
            return extrinsic, 0, extrinsic
        elif self.intrinsic_reward_ratio == 1:
            return intrinsic, intrinsic, 0
        else:
            weighted_int = self.intrinsic_reward_ratio * intrinsic
            weighted_ext = (1 - self.intrinsic_reward_ratio) * extrinsic
            weighted_int_ext = weighted_int + weighted_ext
            return weighted_int_ext, weighted_int, weighted_ext

    @abstractmethod
    def intrinsic_motivation_impl(self, i_episode, step, batch_tuple, current_sars, current_done):
        raise NotImplementedError("Please implement this method.")

    def intrinsic_motivation(self, i_episode, step, batch_tuple, current_sars, current_done):
        intrinsic_reward_batch = self.intrinsic_motivation_impl(i_episode, step, batch_tuple, current_sars, current_done)
        return intrinsic_reward_batch

    def get_reward(self, i_episode, step, batch_tuple, current_sars, current_done):
        if self.intrinsic_reward_ratio == 0:
            transitions, state_batch, action_batch, reward_batch, next_state_batch = batch_tuple
            return torch.zeros_like(state_batch).to(self.device)

        return self.intrinsic_motivation(i_episode, step, batch_tuple, current_sars, current_done)
