# -*- coding: utf-8 -*-

import numpy as np

from utils_kdm import TorchSerializable

# Ornstein–Uhlenbeck process (오른스타인-우렌벡 과정) [1930]
# 중심값 0을 주변으로, 일정 시간 간에는 서로 상관이 있는 방식으로 꿈틀꿈틀 진동(Brawnian particle)
# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
# Implemented by OpenAI on https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py

class OrnsteinUhlenbeckNoise(TorchSerializable):

    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.2):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.X = np.ones(self.action_dim) * self.mu

    def reset(self):
        self.X = np.ones(self.action_dim) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.X)
        dx = dx + self.sigma * np.random.randn(len(self.X))
        self.X = self.X + dx
        return self.X

    def state_dict_impl(self):
        return {
            'action_dim': self.action_dim,
            'mu': self.mu,
            'theta': self.theta,
            'sigma': self.sigma,
            'X': self.X
        }

    def load_state_dict_impl(self, state_dict):
        self.action_dim = state_dict['action_dim']
        self.mu = state_dict['mu']
        self.theta = state_dict['theta']
        self.sigma = state_dict['sigma']
        self.X = state_dict['X']
