# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F

import utils_kdm as u
from algorithm_im.im_base import IntrinsicMotivation
from algorithm_im.region import RegionManager


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

    def __init__(self, state_size, action_size):
        super().__init__(state_size, action_size)
        self._set_hyper_parameters()
        self.region_manager = RegionManager(self.state_size, self.action_size)

    def _set_hyper_parameters(self):
        super()._set_hyper_parameters()

    def state_dict_impl(self):
        # TODO: 저장 불러오기
        todo = {
        }
        return todo

    def load_state_dict_impl(self, var_state):
        # TODO: 저장 불러오기
        pass

    def intrinsic_motivation_impl(self, i_episode, step, current_sars, current_done):
        # Learning progress motivation (LPM)
        current_state, current_action, current_reward, current_next_state = current_sars

        examplar = self.region_manager.exemplar_structure(
            u.t_float32(current_state),
            u.t_float32(current_action),
            u.t_float32(current_next_state)
        )
        self.region_manager.add(examplar)

        # intrinsic_reward_batch = torch.zeros(128).to(self.device)
        #
        # for i, transition in enumerate(transitions):
        #     region = self.region_manager.find_region(transition)
        #     past_error = region.get_past_error_mean()
        #     current_error = region.get_current_error_mean()
        #     intrinsic_reward_batch[i] = past_error - current_error

        region = self.region_manager.find_region(examplar)
        past_error = region.get_past_error_mean()
        current_error = region.get_current_error_mean()
        intrinsic_reward = past_error - current_error

        # TODO: 환경 평소 보상 (1) 정도로 clip 해줄까?
        # intrinsic_reward_batch = torch.clamp(intrinsic_reward_batch, min=-2, max=2)
        # self.viz.draw_line(y=torch.mean(intrinsic_reward_batch), interval=1000, name="intrinsic_reward_batch")

        return intrinsic_reward
