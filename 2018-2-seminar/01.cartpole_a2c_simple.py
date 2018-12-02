# -*- coding: utf-8 -*-
# CartPole-v1 with A2C
#
# 참조한 소스
# https://github.com/rlcode/reinforcement-learning-kr/blob/master/2-cartpole/2-actor-critic/cartpole_a2c.py
# https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
#

import sys
import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import gym
from visdom import Visdom

#####################
# 1. 공통 환경설정과 유틸리티
#####################
Visdom().delete_env('main')
viz = Visdom()

# CPU로 할래 GPU로 할래?
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# 머신 숫자 표현 최저값
eps = np.finfo(np.float32).eps.item()


def _t_uint8(item):
    return torch.tensor([item], device=device, dtype=torch.float32)


def _t_float32(item):
    return torch.tensor([item], device=device, dtype=torch.float32)


def _t_long(item):
    return torch.tensor([item], device=device, dtype=torch.long)


def _t_from_np_to_float32(item):
    return torch.from_numpy(item).float().to(device)


#####################
# 2. 데이터 형태 정의
#####################
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


#####################
# 3. 학습 알고리즘 정의
#####################
class A2CAgent:

    def __init__(self, state_size, action_size):
        self.render = False
        # self.load_model = False
        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        self.learning_rate_actor = 0.001
        self.learning_rate_critic = 0.005
        self.discount_factor = 0.99

        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate_critic)

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.kaiming_uniform_(m.weight)

    def build_actor(self):
        actor = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size),
            nn.Softmax(dim=-1)  # 가장 오른쪽 차원
        )
        actor.apply(self._init_weights).to(device)
        return actor

    def build_critic(self):
        critic = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.value_size)
        )
        critic.apply(self._init_weights).to(device)
        return critic

    def get_action(self, state):
        state = _t_from_np_to_float32(state)
        probs = self.actor(state)
        return Categorical(probs).sample().item()

    def actor_update(self, state, action, advantage):
        self.actor_optimizer.zero_grad()
        probs = self.actor(state)
        loss = -Categorical(probs).log_prob(action) * advantage
        loss.backward()
        self.actor_optimizer.step()
        return loss

    def critic_update(self, state, target):
        self.critic_optimizer.zero_grad()
        target = _t_float32(target)
        loss = torch.mean((target - self.critic(state)) ** 2)
        loss.backward()
        self.critic_optimizer.step()
        return loss

    def train_model(self, state, action, reward, next_state, done):
        state = _t_from_np_to_float32(state)
        action = _t_float32(action)
        next_state = _t_from_np_to_float32(next_state)
        value = self.critic(state)
        next_value = self.critic(next_state)

        if done:
            advantage = reward - value
            target = reward
        else:
            advantage = (reward + self.discount_factor * next_value) - value
            target = reward + self.discount_factor * next_value

        actor_loss = self.actor_update(state, action, advantage)
        critic_loss = self.critic_update(state, target)

        return actor_loss, critic_loss


if __name__ == "__main__":
    EPISODES = 10000
    MAX_STEP = 3000
    LOG_INTERVAL = 1
    env = gym.make('CartPole-v1')
    """
    상태 공간 4개, 범위 -∞ < s < ∞
    행동 공간 1개, 이산값 0 or 1
    """
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = A2CAgent(state_size, action_size)

    # 프로그램 흐름 제어 임시 변수
    scores = list()
    last_actor_loss, last_critic_loss = _t_float32(0), _t_float32(0)

    for e in range(EPISODES):
        state = env.reset()
        score = 0

        for t in range(MAX_STEP):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            if agent.render:
                env.render()
            reward = reward if not done or score == 499 else -100

            last_actor_loss, last_critic_loss = agent.train_model(state, action, reward, next_state, done)

            score += reward
            state = next_state
            if done:
                break

        score = score if score == 500.0 else score + 100
        scores.append(score)
        if e % LOG_INTERVAL == 0:
            print('Episode {}\tLast length: {:5d}'.format(e, t))

            viz.line(X=np.array([e]), Y=np.array([score]), name='score', win='score', update='append', opts={'title':'score'})
            viz.line(X=np.array([e]), Y=np.array([last_actor_loss.item()]), name='actor_loss', win='actor_loss', update='append', opts={'title':'actor_loss'})
            viz.line(X=np.array([e]), Y=np.array([last_critic_loss.item()]), name='critic_loss', win='critic_loss', update='append', opts={'title':'critic_loss'})

            # 이전 10개 에피소드의 점수 평균이 490보다 크면 학습 중단
            if np.mean(scores[-min(10, len(scores)):]) > 490:
                sys.exit()
