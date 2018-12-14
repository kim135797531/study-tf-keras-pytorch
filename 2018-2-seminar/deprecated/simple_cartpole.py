# -*- coding: utf-8 -*-
# CartPole-v1 with DQN
#
# 참조한 소스
# https://github.com/rlcode/reinforcement-learning-kr/blob/master/2-cartpole/1-dqn/cartpole_dqn.py
# https://pytorch.org/tutorials/_downloads/reinforcement_q_learning.py
#

import random
import sys
import time
from collections import namedtuple

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import utils_kdm as u
from utils_kdm.checkpoint import Checkpoint
from utils_kdm.drawer import Drawer
from utils_kdm.replay_memory import ReplayMemory
# Python Pickle은 nested namedtuple save를 지원하지 않음
# https://stackoverflow.com/questions/4677012/python-cant-pickle-type-x-attribute-lookup-failed
from utils_kdm.trainer_metadata import TrainerMetadata

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class DQNAgent(u.TorchSerializable):

    def __init__(self, state_size, action_size):
        self._set_hyper_parameters()
        self.device = TrainerMetadata().device

        self.state_size = state_size
        self.action_size = action_size

        self.policy_model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.target_model.eval()

        self.policy_optimizer = optim.Adam(self.policy_model.parameters(), lr=self.learning_rate_dqn)

        self.transition_structure = Transition
        self.memory = ReplayMemory(self.memory_maxlen, self.transition_structure)

    def _set_hyper_parameters(self):
        self.discount_factor = 0.99
        self.learning_rate_dqn = 0.001
        self.train_start = 1000
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.memory_maxlen = 2000

    def state_dict_impl(self):
        return {
            'memory': self.memory.state_dict(),
            'policy_model': self.policy_model.state_dict(),
            'target_model': self.target_model.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict()
        }

    def load_state_dict_impl(self, var_state):
        self.memory.load_state_dict(var_state['memory'])
        self.policy_model.load_state_dict(var_state['policy_model'])
        self.target_model.load_state_dict(var_state['target_model'])
        self.policy_optimizer.load_state_dict(var_state['policy_optimizer'])

    def _init_weights(self, m):
        if type(m) == nn.Linear:
            nn.init.kaiming_uniform_(m.weight)

    def build_model(self):
        actor = nn.Sequential(
            nn.Linear(self.state_size, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, self.action_size)
        )
        actor.apply(self._init_weights).to(self.device)
        return actor

    def update_target_model(self):
        self.target_model.load_state_dict(self.policy_model.state_dict())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # 낮은 확률로 랜덤으로 선택한다
            return random.randrange(self.action_size)
        else:
            # 현재 상태 기준으로 정책망에서 행동 보상을 예측한 값을 갖고 오고, 큰 쪽을 행동으로 취한다
            state = u.t_from_np_to_float32(state)
            _, index = self.policy_model(state).max(dim=0)
            return index.item()

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.push(
            u.t_float32(state),
            action,
            reward,
            u.t_float32(next_state),
            u.t_uint8(done)
        )

    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 일정 크기만큼 기억을 불러온다
        # 그 후 기억을 모아 각 변수별로 모은다. (즉, 전치행렬)
        transitions = self.memory.sample(self.batch_size)
        batch = self.transition_structure(*zip(*transitions))

        target = []
        target_val = []

        # 정책망에 각각의 기억에 대해 상태를 넣어서 각각의 액션 보상을 구한다.
        # 그 다음에 선택한 액션 쪽의 보상을 가져온다.
        for i in range(self.batch_size):
            state = batch.state[i]
            action = batch.action[i]
            target.append(self.policy_model(state).squeeze()[action])

        # 안 죽었을 때의 타겟망 보상 추정하기
        for i in range(self.batch_size):
            next_state = batch.next_state[i]
            target_val.append(self.target_model(next_state))

        # 기존 보상에 안 죽었을 때만 큐함수 추정을 더하기
        for i in range(self.batch_size):
            done = batch.done[i]
            reward = batch.reward[i]
            if done:
                target_val[i] = u.t_float32(reward).squeeze()
            else:
                target_val[i] = reward + self.discount_factor * torch.max(target_val[i]).to(self.device)

        # 정책망의 예측 보상과 타겟망의 예측 보상을 MSE 비교
        self.policy_optimizer.zero_grad()
        loss = nn.MSELoss().to(self.device)
        loss = loss(torch.stack(target), torch.stack(target_val))
        loss.backward()
        self.policy_optimizer.step()

        TrainerMetadata().log(loss, 'policy_loss')


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
    VIZ_ENV_NAME = '02.cartpole_dqn_simple'

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

    agent = DQNAgent(state_size, action_size)

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
    for episode in range(TrainerMetadata().current_epoch, EPISODES):
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

            agent.append_sample(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                last_loss = agent.train_model()

            score += reward
            state = next_state

            env.render() if RENDER else None
            TrainerMetadata().finish_step()
            if done:
                break

        agent.update_target_model()
        TrainerMetadata().log(score + 100, 'score')
        TrainerMetadata().log(len(agent.memory), 'memory_len')
        TrainerMetadata().finish_episode(episode)

        if IS_SAVE:
            TrainerMetadata().save()

        scores = TrainerMetadata().indicators['score']['default_var']
        if np.mean(scores[-10:]) > 490:
            sys.exit()
