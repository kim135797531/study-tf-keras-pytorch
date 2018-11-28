# -*- coding: utf-8 -*-
# Swimmer-v2 with DDPG
#
# 최대 스텝 제한이 1000번일때 리플레이 메모리 크기는 어느 정도여야 하나?
# 액션이 -1~1 사이인데 noise를 더하면 1이 넘는데 어케함?


import sys
import shutil
import random
import time
from collections import namedtuple
from copy import deepcopy
from itertools import compress

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym
from visdom import Visdom

#####################
# 1. 공통 환경설정과 유틸리티
#####################
Visdom().delete_env('main')
viz = Visdom()
global_step_for_visdom = 0

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


def fanin_init(in_tensor):
    tensor = in_tensor.detach()
    size = tensor.size()
    bound = 1. / np.sqrt(size[0])
    return tensor.uniform_(-bound, bound)


def draw_line(x, y, name):
    name = 'main' + name
    viz.line(X=np.array([x]), Y=np.array([y]), name=name, win=name, update='append', opts={'title': name})


class Checkpoint:

    def __init__(self, save_model=True, save_interval=10):
        self.version = 1
        self.save_model = save_model
        self.save_interval = save_interval
        self.default_file_name = self._get_current_file_name()

    def _get_current_file_name(self):
        return __file__ if '__file__' in vars() or '__file__' in globals() else 'undefined_name'

    def is_saving_episode(self, current_epoch):
        return self.save_model and current_epoch % self.save_interval == 0

    def get_best_model_file_name(self, file_name=None):
        file_name = self.default_file_name if not file_name else file_name
        return file_name + '.best.pt'

    def save_checkpoint(self, var_state, is_best=False, file_name=None):
        file_name = self.default_file_name if not file_name else file_name
        full_name = "{}.ep{}.pt".format(file_name, str(var_state['current_epoch']))
        var_state['version'] = self.version
        torch.save(var_state, full_name)
        if is_best:
            shutil.copyfile(full_name, self.get_best_model_file_name(file_name))

    # noinspection PyUnresolvedReferences
    def load_model(self, file_name=None):
        return torch.load(self.get_best_model_file_name(file_name), map_location=device)
        # return torch.load(self.get_best_model_file_name(file_name))


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
        return random.sample(self.memory, min(len(self.memory), batch_size))

    def __len__(self):
        return len(self.memory)

    def state_dict(self):
        return {
            'capacity': self.capacity,
            'memory': self.memory,
            'position': self.position
        }

    def load_state_dict(self, state_dict):
        self.capacity = state_dict['capacity']
        self.memory = state_dict['memory']
        self.position = state_dict['position']


#####################
# 3. 학습 알고리즘 정의
#####################
# noinspection PyShadowingNames
class Actor(nn.Module):

    def __init__(self, state_size, action_size, action_min=-1, action_max=1):
        super(Actor, self).__init__()
        self.action_min = action_min
        self.action_max = action_max
        self.linear1 = nn.Linear(state_size, 400)
        self.linear2 = nn.Linear(400, 300)
        self.head = nn.Linear(300, action_size)
        # self.linear1 = nn.Linear(state_size, 256)
        # self.linear2 = nn.Linear(256, 128)
        # self.head = nn.Linear(128, action_size)
        fanin_init(self.linear1.weight)
        fanin_init(self.linear2.weight)
        # nn.init.kaiming_uniform_(self.linear1.weight)
        # nn.init.kaiming_uniform_(self.linear2.weight)
        nn.init.uniform_(self.head.weight, a=-3*10e-3, b=3*10e-3)
        # TODO: torch.clamp (v -mean) 으로 정규화 필요

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.head(torch.tanh(x))
        # 값 잘라야 하나?
        return torch.clamp(x, min=self.action_min, max=self.action_max).to(device)


# noinspection PyShadowingNames
class Critic(nn.Module):

    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.linear1 = nn.Linear(state_size + action_size, 400)
        self.linear2 = nn.Linear(400, 300)
        self.head = nn.Linear(300, 1)
        # self.state_linear1 = nn.Linear(state_size, 256)
        # self.state_linear2 = nn.Linear(256, 128)
        # self.action_linear1 = nn.Linear(action_size, 128)
        # self.linear2 = nn.Linear(256, 128)
        # self.head = nn.Linear(128, 1)
        fanin_init(self.linear1.weight)
        fanin_init(self.linear2.weight)
        # nn.init.kaiming_uniform_(self.state_linear1.weight)
        # nn.init.kaiming_uniform_(self.state_linear2.weight)
        # nn.init.kaiming_uniform_(self.action_linear1.weight)
        # nn.init.kaiming_uniform_(self.linear2.weight)
        nn.init.uniform_(self.head.weight, a=-3*10e-4, b=3*10e-4)
        # TODO: torch.clamp (v -mean) 으로 정규화 필요

    def forward(self, state, action):
        # Actions were not included until the 2nd Hidden layer of Q
        # TODO: 해석 불분명: **2번째 은닉층까지 안 넣었다**
        # (a) 2번째 직전까진 안 넣었으니 2번째에 넣었을 것이다
        # (b) 2번째 까지는 안 넣었으니 3번째에 넣었을 것이다
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # x = F.relu(self.state_linear1(state))
        # x = F.relu(self.state_linear2(x))
        # y = F.relu(self.action_linear1(action))
        return self.head(x)


# Ornstein–Uhlenbeck process (오른스타인-우렌벡 과정) [1930]
# 중심값 0을 주변으로, 일정 시간 간에는 서로 상관이 있는 방식으로 꿈틀꿈틀 진동(Brawnian particle)
# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
# Implemented by OpenAI on https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class OrnsteinUhlenbeckNoise:

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

    def state_dict(self):
        return {
            'action_dim': self.action_dim,
            'mu': self.mu,
            'theta': self.theta,
            'sigma': self.sigma,
            'X': self.X
        }

    def load_state_dict(self, state_dict):
        self.action_dim = state_dict['action_dim']
        self.mu = state_dict['mu']
        self.theta = state_dict['theta']
        self.sigma = state_dict['sigma']
        self.X = state_dict['X']


# noinspection PyShadowingNames,PyMethodMayBeStatic
class DDPGAgent:
    def __init__(self, state_size, action_size, action_min=-1, action_max=1):
        # 1. 기본 설정
        self.render = True

        self.state_size = state_size
        self.action_size = action_size
        self.action_min = action_min
        self.action_max = action_max

        # 2. Adam 하이퍼 파라미터
        # 아래 값들은 저차원 입력(관절 수치 등)일 때 이다
        # 화상 자체를 입력으로 받을 때는 값 다름
        self.learning_rate_actor = 0.0001
        self.learning_rate_critic = 0.001
        self.l2_weight_decay = 0.01
        self.soft_target_update_tau = 0.001

        # 3. 평가망 학습 하이퍼 파라미터
        self.discount_factor = 0.99

        # 4. 오른스타인-우렌벡 과정 관련
        self.mu = 0
        self.theta = 0.15
        self.sigma = 0.2

        # 5. 리플레이 메모리 관련
        self.batch_size = 128
        # self.memory_maxlen = 2000
        # self.memory_maxlen = int(10e+6)
        self.memory_maxlen = int(1000000)
        self.train_start = 100

        # 6. 모델 빌드
        self.actor = Actor(self.state_size, self.action_size).to(device)
        self.critic = Critic(self.state_size, self.action_size).to(device)
        self.target_actor = Actor(self.state_size, self.action_size).to(device)
        self.target_critic = Critic(self.state_size, self.action_size).to(device)

        # 타겟 정책망, 타겟 평가망 가중치를 각각 정책망, 평가망 가중치로 초기화
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        # 타겟망들은 오차계산 및 업데이트 안 하는 평가 전용모드임을 선언
        self.target_actor.eval()
        self.target_critic.eval()

        # Optimizer
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=self.learning_rate_actor
        )
        # critic 에만 L2 weight decay 넣음
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=self.learning_rate_critic,
            weight_decay=self.l2_weight_decay
        )

        # 리플레이 메모리
        self.memory = ReplayMemory(self.memory_maxlen)

        # 오른스타인-우렌벡 과정
        self.noise = OrnsteinUhlenbeckNoise(self.action_size, self.mu, self.theta, self.sigma)

    def state_dict(self):
        return {
            'memory': self.memory.state_dict(),
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'noise': self.noise.state_dict(),
        }

    def load_state_dict(self, var_state):
        self.memory.load_state_dict(var_state['memory'])
        self.actor.load_state_dict(var_state['actor'])
        self.critic.load_state_dict(var_state['critic'])
        self.target_actor.load_state_dict(var_state['target_actor'])
        self.target_critic.load_state_dict(var_state['target_critic'])
        self.actor_optimizer.load_state_dict(var_state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(var_state['critic_optimizer'])
        self.noise.load_state_dict(var_state['noise'])

    def get_action(self, state):
        state = _t_from_np_to_float32(state)
        noise = self.noise.sample()
        action = self.actor(state).detach().cpu().numpy()
        action += noise
        # TODO: 이렇게 하는게 맞나?
        return np.clip(action, a_min=self.action_min, a_max=self.action_max)
        # return action

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.push(
            _t_float32(state),
            _t_float32(action),
            _t_float32(reward),
            _t_float32(next_state),
            _t_uint8(done)
        )

    def copy_only_weight(self, src_nn, dst_nn, tau=1.0):
        src_state_dict = deepcopy(src_nn.state_dict())
        dst_state_dict = dst_nn.state_dict()
        for k in dst_state_dict.keys():
            if 'weight' in k:
                src_weight, dst_weight = src_state_dict[k], dst_state_dict[k]
                dst_state_dict[k] = tau * src_weight + (1.0 - tau) * dst_weight
            # TODO: 없애야함
            if 'bias' in k:
                src_weight, dst_weight = src_state_dict[k], dst_state_dict[k]
                dst_state_dict[k] = tau * src_weight + (1.0 - tau) * dst_weight
        dst_nn.load_state_dict(dst_state_dict)

    def train_model(self):
        # 메모리에서 일정 크기만큼 기억을 불러온다
        # 그 후 기억을 모아 각 변수별로 모은다. (즉, 전치행렬)
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # 텐서의 집합에서 고차원 텐서로
        # tuple(tensor, ...) -> tensor()
        state_batch = torch.cat(batch.state).to(device)
        action_batch = torch.cat(batch.action).to(device)
        reward_batch = torch.cat(batch.reward).to(device)
        next_state_batch = torch.cat(batch.next_state).to(device)

        # <평가망 최적화>
        # (무엇을, 어디서, 어떻게, 왜)
        # 각각의 기억에 대해, 타겟 정책망에, 다음 상태를 넣어서, 다음 타겟 액션을 구한다.
        # 각각의 기억에 대해, 타겟 평가망에, 다음 상태와 타겟 액션을 넣어서, 다음 타겟 보상을 구한다.
        target_actions = self.target_actor(next_state_batch)
        target_rewards = self.target_critic(next_state_batch, target_actions)
        # 현재 보상에 타겟 보상을 더해서 예측한 보상을 구한다.
        expected_rewards = reward_batch.unsqueeze(dim=1) + self.discount_factor * target_rewards
        predicted_rewards = self.critic(state_batch, action_batch)

        global global_step_for_visdom

        # 평가망의 예측 보상과 타겟 평가망의 예측 보상을 MSE 비교 후 업데이트
        self.critic_optimizer.zero_grad()
        critic_loss = nn.MSELoss().to(device)
        critic_loss = critic_loss(expected_rewards, predicted_rewards)
        if global_step_for_visdom % 100 == 0:
            draw_line(global_step_for_visdom, critic_loss.item(), "critic_loss")
        critic_loss.backward()
        self.critic_optimizer.step()

        # <정책망 최적화>
        self.actor_optimizer.zero_grad()
        predicted_actions = self.actor(state_batch)
        q_output = self.critic(state_batch, predicted_actions)
        # actor_loss = -1*torch.sum(q_output).to(device)
        # TODO: sum 이 아니라 mean 인 이유
        actor_loss = -1 * torch.mean(q_output).to(device)
        if global_step_for_visdom % 100 == 0:
            draw_line(global_step_for_visdom, actor_loss.item(), "actor_loss")

        global_step_for_visdom += 1

        # 정책망의 예측 보상을 정책 그라디언트로 업데이트
        actor_loss.backward()
        self.actor_optimizer.step()

        # 현재 평가망의 가중치를 타겟 평가망에다 덮어쓰기
        self.copy_only_weight(src_nn=self.critic, dst_nn=self.target_critic, tau=self.soft_target_update_tau)

        # 현재 정책망의 가중치를 타겟 정책망에다 덮어쓰기
        self.copy_only_weight(src_nn=self.actor, dst_nn=self.target_actor, tau=self.soft_target_update_tau)

        # 그래프 그리기용
        return critic_loss, actor_loss


if __name__ == "__main__":
    EPISODES = 30000
    LOG_INTERVAL = 1
    LOAD_MODEL = True
    SAVE_MODEL = True
    SAVE_INTERVAL = 400

    env = gym.make('Swimmer-v2')
    """
    상태 공간 8개, 범위 -∞ < s < ∞
    행동 공간 2개, 범위 -1 < a < 1
    """
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    agent = DDPGAgent(state_size, action_size)

    # 프로그램 흐름 제어 임시 변수
    scores = list()
    last_critic_losses = list()
    last_actor_losses = list()

    last_critic_loss = _t_float32(0)
    last_actor_loss = _t_float32(0)

    current_epoch = 0
    best_score = 0
    resumer = Checkpoint(SAVE_MODEL, SAVE_INTERVAL)
    if LOAD_MODEL:
        print("Loading checkpoint '{}'".format(resumer.get_best_model_file_name()))
        var_state = resumer.load_model()
        # noinspection PyRedeclaration
        current_epoch = var_state['current_epoch']
        scores.extend(var_state['scores'])
        last_critic_losses.extend(var_state['last_critic_losses'])
        last_actor_losses.extend(var_state['last_actor_losses'])
        agent.load_state_dict(var_state)
        for e in range(0, len(scores)):
            draw_line(e, scores[e], 'score')
            draw_line(e, last_actor_losses[e].item(), 'last_actor_loss')
            draw_line(e, last_critic_losses[e].item(), 'last_critic_loss')
        print("Loading complete. Resuming from episode: {}, score: {:.2f}".format(current_epoch - 1, max(scores, default=0)))

    # 1 에피소드 = 카트 넘어가거나 3000스텝 넘게 버티거나
    for e in range(current_epoch, EPISODES):
        start_time = time.time()
        agent.noise.reset()
        state = env.reset()
        score = 0

        # 각 에피소드마다 최대 3000스텝까지 시뮬레이션 한다
        for t in range(env.spec.max_episode_steps):
            action = agent.get_action(state)

            next_state, reward, done, _ = env.step(action)
            if agent.render:
                env.render()

            agent.append_sample(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                last_critic_loss, last_actor_loss = agent.train_model()

            score += reward
            state = next_state
            if done:
                break

        scores.append(score)
        last_actor_losses.append(last_actor_loss)
        last_critic_losses.append(last_critic_loss)
        if e % LOG_INTERVAL == 0:
            print('Episode {}\tScore: {:.2f}\tMem Length: {}\tCompute Time: {:.2f}'.format(
                e, score, len(agent.memory), time.time() - start_time))
            draw_line(e, score, 'score')
            draw_line(e, last_actor_loss.item(), 'last_actor_loss')
            draw_line(e, last_critic_loss.item(), 'last_critic_loss')

        if resumer.is_saving_episode(e):
            var_state = agent.state_dict()
            var_state['current_epoch'] = e + 1
            var_state['scores'] = scores
            var_state['last_actor_losses'] = last_actor_losses
            var_state['last_critic_losses'] = last_critic_losses
            is_best = False
            if max(scores) > best_score:
                best_score = max(scores)
                is_best = True
            resumer.save_checkpoint(var_state, is_best)
