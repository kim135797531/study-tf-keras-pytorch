# -*- coding: utf-8 -*-
# CartPole-v1 with DQN
#
# 참조한 소스
# https://github.com/rlcode/reinforcement-learning-kr/blob/master/2-cartpole/1-dqn/cartpole_dqn.py
# https://pytorch.org/tutorials/_downloads/reinforcement_q_learning.py
#

import sys
import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

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
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


#####################
# 3. 학습 알고리즘 정의
#####################
class DQNAgent:

    def __init__(self, state_size, action_size):
        self.render = False
        # self.load_model = False
        self.state_size = state_size
        self.action_size = action_size

        # DQN 파라미터
        self.discount_factor = 0.99
        self.learning_rate_dqn = 0.001
        self.train_start = 1000
        self.epsilon = 1.0
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.01
        self.batch_size = 64
        self.memory_maxlen = 2000

        self.policy_model = self.build_model()
        self.target_model = self.build_model()
        self.update_target_model()
        self.target_model.eval()

        self.policy_optimizer = optim.Adam(self.policy_model.parameters(), lr=self.learning_rate_dqn)

        self.memory = ReplayMemory(self.memory_maxlen)

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
        actor.apply(self._init_weights).to(device)
        return actor

    def update_target_model(self):
        self.target_model.load_state_dict(self.policy_model.state_dict())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            # 낮은 확률로 랜덤으로 선택한다
            return random.randrange(self.action_size)
        else:
            # 현재 상태 기준으로 정책망에서 행동 보상을 예측한 값을 갖고 오고, 큰 쪽을 행동으로 취한다
            state = _t_from_np_to_float32(state)
            _, index = self.policy_model(state).max(dim=0)
            return index.item()

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.push(
            _t_float32(state),
            action,
            reward,
            _t_float32(next_state),
            done
        )

    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 일정 크기만큼 기억을 불러온다
        # 그 후 기억을 모아 각 변수별로 모은다. (즉, 전치행렬)
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

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
                target_val[i] = _t_float32(reward).squeeze()
            else:
                target_val[i] = reward + self.discount_factor * (
                    torch.max(target_val[i]).to(device)
                )

        # 정책망의 예측 보상과 타겟망의 예측 보상을 MSE 비교
        self.policy_optimizer.zero_grad()
        loss = nn.MSELoss().to(device)
        loss = loss(torch.stack(target), torch.stack(target_val))
        loss.backward()
        self.policy_optimizer.step()

        # 그래프 그리기용
        return loss


if __name__ == "__main__":
    EPISODES = 1000
    MAX_STEP = 3000
    LOG_INTERVAL = 1
    env = gym.make('CartPole-v1')
    """
    상태 공간 4개, 범위 -∞ < s < ∞
    행동 공간 1개, 이산값 0 or 1
    """
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    # 프로그램 흐름 제어 임시 변수
    scores = list()
    last_loss = torch.Tensor([0])

    # 1 에피소드 = 카트 넘어가거나 3000스텝 넘게 버티거나
    for e in range(EPISODES):
        state = env.reset()
        score = 0

        # 각 에피소드마다 최대 3000스텝까지 시뮬레이션 한다
        for t in range(MAX_STEP):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            if agent.render:
                env.render()
            reward = reward if not done or score == 499 else -100

            agent.append_sample(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                last_loss = agent.train_model()

            score += reward
            state = next_state
            if done:
                break

        # 정책망의 가중치를 가져와서 타겟망의 가중치에 덮어씌운다
        agent.update_target_model()
        score = score if score == 500.0 else score + 100
        scores.append(score)
        if e % LOG_INTERVAL == 0:
            print('Episode {}\tScore: {}\tMem Length: {}\t epsilon: {}'.format(e, score, len(agent.memory), agent.epsilon))

            viz.line(X=np.array([e]), Y=np.array([score]), name='score', win='reward', update='append', opts={'title':'score'})
            viz.line(X=np.array([e]), Y=np.array([last_loss.item()]), name='loss', win='loss', update='append', opts={'title':'loss'})
            viz.line(X=np.array([e]), Y=np.array([agent.epsilon]), name='epsilon', win='epsilon', update='append', opts={'title':'epsilon'})

        # 이전 10개 에피소드의 점수 평균이 490보다 크면 학습 중단
        if np.mean(scores[-min(10, len(scores)):]) > 490:
            sys.exit()
