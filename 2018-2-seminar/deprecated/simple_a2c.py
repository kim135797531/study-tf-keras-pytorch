# -*- coding: utf-8 -*-
# CartPole-v1 with A2C
#
# 참조한 소스
# https://github.com/rlcode/reinforcement-learning-kr/blob/master/2-cartpole/2-actor-critic/cartpole_a2c.py
# https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
#

import sys
from collections import namedtuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import utils_kdm as u
from utils_kdm.checkpoint import Checkpoint
from utils_kdm.drawer import Drawer
# Python Pickle은 nested namedtuple save를 지원하지 않음
# https://stackoverflow.com/questions/4677012/python-cant-pickle-type-x-attribute-lookup-failed
from utils_kdm.trainer_metadata import TrainerMetadata

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class A2CAgent(u.TorchSerializable):

    def __init__(self, state_size, action_size):
        self._set_hyper_parameters()
        self.device = TrainerMetadata().device

        self.state_size = state_size
        self.action_size = action_size
        self.value_size = 1

        self.actor = self.build_actor()
        self.critic = self.build_critic()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate_critic)

    def _set_hyper_parameters(self):
        self.learning_rate_actor = 0.001
        self.learning_rate_critic = 0.005
        self.discount_factor = 0.99

    def state_dict_impl(self):
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict()
        }

    def load_state_dict_impl(self, var_state):
        self.actor.load_state_dict(var_state['actor'])
        self.critic.load_state_dict(var_state['critic'])
        self.actor_optimizer.load_state_dict(var_state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(var_state['critic_optimizer'])

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
        actor.apply(self._init_weights).to(self.device)
        return actor

    def build_critic(self):
        critic = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.value_size)
        )
        critic.apply(self._init_weights).to(self.device)
        return critic

    def get_action(self, state):
        state = u.t_from_np_to_float32(state)
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
        target = u.t_float32(target)
        loss = torch.mean((target - self.critic(state)) ** 2)
        loss.backward()
        self.critic_optimizer.step()
        return loss

    def train_model(self, state, action, reward, next_state, done):
        state = u.t_from_np_to_float32(state)
        action = u.t_float32(action)
        next_state = u.t_from_np_to_float32(next_state)
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
    #####################
    # 환경 설정
    #####################

    # 0. 일반 설정
    FORCE_CPU = False
    TrainerMetadata().set_device(force_cpu=FORCE_CPU)

    # 1. 시각화 관련 설정
    VISDOM_RESET = True
    # VIZ_ENV_NAME = os.path.basename(os.path.realpath(__file__))
    VIZ_ENV_NAME = '01.cartpole_a2c_simple'

    # 2. 저장 관련 설정
    VERSION = 1
    IS_LOAD, IS_SAVE, SAVE_INTERVAL = False, True, 400
    SAVE_FULL_PATH = __file__

    # 3. 실험 환경 관련 설정
    GYM_ENV = 'CartPole-v1'
    RENDER = False
    LOG_INTERVAL = 1
    EPISODES = 10000

    #####################
    # 객체 구성
    #####################
    viz = Drawer(reset=VISDOM_RESET, env=VIZ_ENV_NAME)
    checkpoint = Checkpoint(VERSION, IS_SAVE, SAVE_INTERVAL)

    # Agent 생성
    env = gym.make(GYM_ENV)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = A2CAgent(state_size, action_size)

    # 메타데이터 관리 클래스 설정
    TrainerMetadata().reset(
        viz=viz,
        checkpoint=checkpoint,
        agent=agent,
        force_cpu=FORCE_CPU,
        log_interval=LOG_INTERVAL,
        save_full_path=SAVE_FULL_PATH
    )

    if IS_LOAD:
        TrainerMetadata().load()

    # 최대 에피소드 수만큼 돌린다
    for i_episode in range(TrainerMetadata().current_epoch, EPISODES):
        TrainerMetadata().start_episode()
        state = env.reset()
        score = u.t_float32(0)

        # 각 에피소드당 환경에 정의된 최대 스텝 수만큼 돌린다
        # 단 그 전에 환경에서 정의된 종료 상태(done)가 나오면 거기서 끝낸다
        for t in range(env.spec.max_episode_steps):
            TrainerMetadata().start_step()

            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done or score == 499 else -100

            agent.train_model(state, action, reward, next_state, done)

            score += reward
            state = next_state

            env.render() if RENDER else None
            TrainerMetadata().finish_step()
            if done:
                break

        TrainerMetadata().log(score + 100, 'score')
        TrainerMetadata().finish_episode(i_episode)

        if IS_SAVE:
            TrainerMetadata().save()

        # 이전 10개 에피소드의 점수 평균이 490보다 크면 학습 중단
        scores = TrainerMetadata().indicators['score']['default_var']
        if np.mean(scores[-10:]) > 490:
            sys.exit()
