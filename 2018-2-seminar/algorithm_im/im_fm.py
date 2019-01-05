# -*- coding: utf-8 -*-
# (4) Predictive familiarity motivation (FM)
# FM = 각 '지역'별로 오차가 작을수록 보상 높음

import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F

import utils_kdm as u
from algorithm_im.im_base import IntrinsicMotivation
from algorithm_im.region import RegionManager, ExemplarStructure
from utils_kdm.trainer_metadata import TrainerMetadata


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
        x = torch.cat((state, action), dim=0)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.head(x)


class PredictiveFamiliarityMotivation(IntrinsicMotivation):

    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self._set_hyper_parameters()
        self.region_manager = RegionManager(self.state_size, self.action_size)

        self.register_serializable([
            'self.region_manager',
        ])

    def _set_hyper_parameters(self):
        super()._set_hyper_parameters()

        # TODO: 적절한 C는 내가 찾아야 함 (일단 알고리즘 밖에서 전체 decay 중)
        self.intrinsic_scale_1 = 0.001

    def intrinsic_motivation_impl(self, i_episode, step, current_sars, current_done):
        # Predictive familiarity motivation (FM)
        current_state, current_action, current_reward, current_next_state = current_sars

        examplar = ExemplarStructure(
            u.t_float32(current_state),
            u.t_float32(current_action),
            u.t_float32(current_next_state)
        )
        self.region_manager.add(examplar)

        region = self.region_manager.find_region(examplar)
        current_error = region.get_current_error_mean()
        intrinsic_reward = self.intrinsic_scale_1 / current_error

        intrinsic_reward = u.t_float32(intrinsic_reward)
        # TrainerMetadata().log(value=intrinsic_reward, indicator='intrinsic_reward',
        # variable='raw', interval=1, show_only_last=False, compute_maxmin=False)
        intrinsic_reward = torch.clamp(intrinsic_reward, min=-2, max=2)
        # TrainerMetadata().log(value=intrinsic_reward, indicator='intrinsic_reward',
        # variable='clamp', interval=1, show_only_last=False, compute_maxmin=False)

        intrinsic_reward = intrinsic_reward.item()
        return intrinsic_reward
