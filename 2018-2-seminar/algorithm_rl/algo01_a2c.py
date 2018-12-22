# -*- coding: utf-8 -*-
# DDPG

from collections import namedtuple

import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

import utils_kdm as u
from utils_kdm.trainer_metadata import TrainerMetadata

# Python Pickle은 nested namedtuple save를 지원하지 않음
# https://stackoverflow.com/questions/4677012/python-cant-pickle-type-x-attribute-lookup-failed

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class Actor(nn.Module):

    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.device = TrainerMetadata().device
        self.layer_sizes = [state_size, 24, action_size]

        self.linear1 = nn.Linear(self.layer_sizes[0], self.layer_sizes[1])
        self.linear2 = nn.Linear(self.layer_sizes[1], self.layer_sizes[2])
        self.head = nn.Softmax(dim=-1)
        u.fanin_init(self.linear1.weight)
        u.fanin_init(self.linear2.weight)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        # TODO: 원래 relu 없나?
        x = F.relu(self.linear2(x))
        return self.head(x)


class Critic(nn.Module):

    def __init__(self, state_size, value_size):
        super(Critic, self).__init__()
        self.device = TrainerMetadata().device
        self.layer_sizes = [state_size, 24, 24, value_size]

        self.linear1 = nn.Linear(self.layer_sizes[0], self.layer_sizes[1])
        self.linear2 = nn.Linear(self.layer_sizes[1], self.layer_sizes[2])
        self.head = nn.Linear(self.layer_sizes[2], self.layer_sizes[3])
        u.fanin_init(self.linear1.weight)
        u.fanin_init(self.linear2.weight)
        nn.init.uniform_(self.head.weight, a=-3*10e-4, b=3*10e-4)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.head(x)


class A2C(u.TorchSerializable):

    def __init__(self, state_size, action_size):
        super().__init__()

        self._set_hyper_parameters()
        self.device = TrainerMetadata().device

        # 기본 설정
        self.state_size, self.action_size = state_size, action_size
        self.value_size = 1

        # 모델 빌드
        self.actor = Actor(self.state_size, self.action_size).to(self.device)
        self.critic = Critic(self.state_size, self.value_size).to(self.device)

        # Optimizer
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.learning_rate_actor
        )
        # critic 에만 L2 weight decay 넣음
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.learning_rate_critic
        )

        self.register_serializable([
            'self.actor',
            'self.critic',
            'self.actor_optimizer',
            'self.critic_optimizer',
        ])

    def _set_hyper_parameters(self):
        # Adam 하이퍼 파라미터
        self.learning_rate_actor = 0.001
        self.learning_rate_critic = 0.005

        # 평가망 학습 하이퍼 파라미터
        self.discount_factor = 0.99

    def reset(self):
        pass

    def get_action(self, state):
        state = u.t_from_np_to_float32(state)
        probs = self.actor(state)
        return Categorical(probs).sample().item()

    def train_model(self, sars, done):
        (state, action, reward, next_state) = sars

        state = u.t_float32(state)
        action = u.t_float32(action)
        reward = u.t_float32(reward)
        next_state = u.t_float32(next_state)

        value = self.critic(state)
        next_value = self.critic(next_state)

        if done:
            advantage = reward - value
            target = reward
        else:
            advantage = (reward + self.discount_factor * next_value) - value
            target = reward + self.discount_factor * next_value

        self.actor_optimizer.zero_grad()
        probs = self.actor(state)
        actor_loss = -Categorical(probs).log_prob(action) * advantage
        actor_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss = torch.mean((target.detach() - self.critic(state)) ** 2)
        critic_loss.backward()
        self.critic_optimizer.step()

        if done:
            TrainerMetadata().log(critic_loss, 'critic_loss')
            TrainerMetadata().log(actor_loss, 'actor_loss')
