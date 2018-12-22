# -*- coding: utf-8 -*-
# DQN
#
# 참조한 소스
# https://github.com/rlcode/reinforcement-learning-kr/blob/master/2-cartpole/1-dqn/cartpole_dqn.py
# https://pytorch.org/tutorials/_downloads/reinforcement_q_learning.py
#

import random
from collections import namedtuple
from itertools import compress

import numpy as np
import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
import torch.optim as optim

import utils_kdm as u
from utils_kdm.replay_memory import ReplayMemory
from utils_kdm.trainer_metadata import TrainerMetadata

# Python Pickle은 nested namedtuple save를 지원하지 않음
# https://stackoverflow.com/questions/4677012/python-cant-pickle-type-x-attribute-lookup-failed

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class DQNNetwork(nn.Module):

    def __init__(self, state_size, action_size, action_range=(-1, 1)):
        super(DQNNetwork, self).__init__()
        self.device = TrainerMetadata().device
        self.layer_sizes = [state_size, 24, 24, action_size]

        # TODO: 정규화된 입력인지 검사 문구 넣고 range 빼기
        self.action_low, self.action_high = action_range
        self.linear1 = nn.Linear(self.layer_sizes[0], self.layer_sizes[1])
        self.linear2 = nn.Linear(self.layer_sizes[1], self.layer_sizes[2])
        self.head = nn.Linear(self.layer_sizes[2], self.layer_sizes[3])
        u.fanin_init(self.linear1.weight)
        u.fanin_init(self.linear2.weight)
        nn.init.uniform_(self.head.weight, a=-3*10e-3, b=3*10e-3)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        return self.head(x)


class DQN(u.TorchSerializable):

    def __init__(self, state_size, action_size, action_range=(-1, 1)):
        super().__init__()

        self._set_hyper_parameters()
        self.device = TrainerMetadata().device

        # 기본 설정
        self.state_size, self.action_size = state_size, action_size
        # TODO: 정규화된 입력인지 검사 문구 넣고 range 빼기
        self.action_low, self.action_high = action_range

        # 모델 빌드
        self.policy = DQNNetwork(self.state_size, self.action_size).to(self.device)
        self.target_policy = DQNNetwork(self.state_size, self.action_size).to(self.device)

        # 타겟 정책망을 정책망 가중치로 초기화
        self.target_policy.load_state_dict(self.policy.state_dict())

        # 타겟망은 오차계산 및 업데이트 안 하는 평가 전용모드임을 선언
        self.target_policy.eval()

        # Optimizer
        self.policy_optimizer = optim.Adam(
            self.policy.parameters(),
            lr=self.learning_rate_policy
        )

        # 리플레이 메모리
        # DQN, DDPG에서 제안하고 쓰는 개념이므로 정의는 따로 두더라도 인스턴스는 알고리즘 내부에서 갖고 있는다
        self.transition_structure = Transition
        self.memory = ReplayMemory(self.memory_maxlen, self.transition_structure)

        self.register_serializable([
            'self.policy',
            'self.target_policy',
            'self.policy_optimizer',
            'self.memory',
            'self.epsilon',
            'self.epsilon_decay',
            'self.epsilon_min',
        ])

    def _set_hyper_parameters(self):
        # Adam 하이퍼 파라미터
        self.learning_rate_policy = 0.001

        # 평가망 학습 하이퍼 파라미터
        self.discount_factor = 0.99

        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01

        # 리플레이 메모리 관련
        self.batch_size = 64
        self.memory_maxlen = 2000
        self.train_start = 64

    def reset(self):
        # 정책망에서 타겟망으로 가중치 복사 (한 에피소드 끝날 때마다 호출됨)
        self.target_policy.load_state_dict(self.policy.state_dict())

    def append_sample(self, sars, done):
        state, action, reward, next_state = sars
        self.memory.push(
            u.t_float32(state),
            u.t_long(action),
            u.t_float32(reward),
            u.t_float32(next_state),
            u.t_uint8(done)
        )

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # 낮은 확률로 랜덤으로 선택한다
            return random.randrange(self.action_size)
        else:
            # 현재 상태 기준으로 정책망에서 행동 보상을 예측한 값을 갖고 오고, 큰 쪽을 행동으로 취한다
            state = u.t_from_np_to_float32(state)
            max_val, max_index = self.policy(state).max(dim=0)
            return max_index.item()

    def train_model(self, sars, done):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 일정 크기만큼 기억을 불러온다
        # 그 후 기억을 모아 각 변수별로 모은다. (즉, 전치행렬)
        # TODO: random과 zip func 글카에서 하기
        transitions = self.memory.sample(self.batch_size)
        # SARS = State, Action, Reward, next State
        sars_batch = self.transition_structure(*zip(*transitions))

        # TODO: 이거 튜플로 묶으면 다시 GPU에서 CPU로 오나?
        # 텐서의 집합에서 고차원 텐서로
        # tuple(tensor, ...) -> tensor()
        s_batch = torch.stack(sars_batch.state).to(self.device)
        a_batch = torch.stack(sars_batch.action).to(self.device)
        r_batch = torch.stack(sars_batch.reward).to(self.device)
        next_s_batch = torch.stack(sars_batch.next_state).to(self.device)
        done_batch = torch.stack(sars_batch.done).to(self.device)

        # 정책망에 각각의 기억에 대해 상태를 넣어서 각각의 액션 보상을 구한다.
        # 그 다음에 선택한 액션 쪽의 보상을 가져온다.
        state_action_values = self.policy(s_batch).gather(1, a_batch)

        # 타겟망 예측에서, 아직 안 죽은 거에만 큐함수 추정을 더해주기 위해 마스크를 만들기
        # 어렵게 마스크를 만드는 이유? 한번에 모아서 신경망에 보내면 실행 속도가 빨라짐..
        # 안 죽었을 때의 상태들만 가져오기
        not_done = [not i for i in done_batch]
        non_final_mask = u.t_uint8(not_done).squeeze().to(self.device)
        non_final_next_states = torch.stack(list(compress(next_s_batch, not_done)))

        # 안 죽었을 때의 타겟망 보상 추정하기
        next_state_values = torch.zeros(len(s_batch), device=self.device)
        next_state_values[non_final_mask] = self.target_policy(non_final_next_states).max(1)[0].detach()
        next_state_values = next_state_values.unsqueeze(dim=1)
        # 기존 보상에 안 죽었을 때만 큐함수 추정을 더하기
        expected_state_action_values = r_batch + (self.discount_factor * next_state_values)

        # 정책망의 예측 보상과 타겟망의 예측 보상을 MSE 비교
        self.policy_optimizer.zero_grad()
        loss = nn.MSELoss().to(self.device)
        loss = loss(state_action_values, expected_state_action_values)
        loss.backward()
        self.policy_optimizer.step()

        if done:
            TrainerMetadata().log(loss, 'policy_loss')
