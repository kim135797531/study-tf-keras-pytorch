# -*- coding: utf-8 -*-
# CartPole-v1 with DQN
#
# 참조한 소스
# https://github.com/rlcode/reinforcement-learning-kr/blob/master/2-cartpole/1-dqn/cartpole_dqn.py
# https://pytorch.org/tutorials/_downloads/reinforcement_q_learning.py
#

import os
import sys
import random
import time
from collections import namedtuple
from itertools import compress

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym

import kdm_utils as u
from kdm_utils.checkpoint import Checkpoint, TorchSerializable
from kdm_utils.drawer import Drawer
from kdm_utils.replay_memory import ReplayMemory


# Python Pickle은 nested namedtuple save를 지원하지 않음
# https://stackoverflow.com/questions/4677012/python-cant-pickle-type-x-attribute-lookup-failed
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class DQN(nn.Module):

    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.layer_sizes = [state_size, 24, 24, action_size]

        self.linear1 = nn.Linear(self.layer_sizes[0], self.layer_sizes[1])
        self.linear2 = nn.Linear(self.layer_sizes[1], self.layer_sizes[2])
        self.head = nn.Linear(self.layer_sizes[2], self.layer_sizes[3])
        nn.init.kaiming_uniform_(self.linear1.weight)
        nn.init.kaiming_uniform_(self.linear2.weight)
        nn.init.kaiming_uniform_(self.head.weight)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.head(x)


class DQNAgent(TorchSerializable):

    def __init__(self, state_size, action_size):
        self._set_hyper_parameters()

        self.state_size = state_size
        self.action_size = action_size

        self.policy_model = DQN(self.state_size, self.action_size).to(device)
        self.target_model = DQN(self.state_size, self.action_size).to(device)
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

    def update_target_model(self):
        # 정책망에서 타겟망으로 가중치 복사 (주로 한 에피소드 끝날 때마다 호출됨)
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
            u.t_long(action),
            u.t_float32(reward),
            u.t_float32(next_state),
            u.t_uint8(done)
        )

    def train_model(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        # 메모리에서 일정 크기만큼 기억을 불러온다
        # 그 후 기억을 모아 각 변수별로 모은다. (즉, 전치행렬)
        transitions = self.memory.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        # 텐서의 집합에서 고차원 텐서로
        # tuple(tensor, ...) -> tensor()
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).unsqueeze(dim=1)
        reward_batch = torch.cat(batch.reward)

        # 정책망에 각각의 기억에 대해 상태를 넣어서 각각의 액션 보상을 구한다.
        # 그 다음에 선택한 액션 쪽의 보상을 가져온다.
        state_action_values = self.policy_model(state_batch).gather(1, action_batch)

        # 타겟망 예측에서, 아직 안 죽은 거에만 큐함수 추정을 더해주기 위해 마스크를 만들기
        # 어렵게 마스크를 만드는 이유? 한번에 모아서 신경망에 보내면 실행 속도가 빨라짐..
        # 안 죽었을 때의 상태들만 가져오기
        not_done = [not i for i in batch.done]
        non_final_mask = torch.tensor(not_done, device=device, dtype=torch.uint8)
        non_final_next_states = torch.cat(list(compress(batch.next_state, not_done)))

        # 안 죽었을 때의 타겟망 보상 추정하기
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.target_model(non_final_next_states).max(1)[0].detach()
        # 기존 보상에 안 죽었을 때만 큐함수 추정을 더하기
        expected_state_action_values = reward_batch + (self.discount_factor * next_state_values)

        # 정책망의 예측 보상과 타겟망의 예측 보상을 MSE 비교
        self.policy_optimizer.zero_grad()
        loss = nn.MSELoss().to(device)
        loss = loss(state_action_values, expected_state_action_values.unsqueeze(1))
        loss.backward()
        self.policy_optimizer.step()

        # 그래프 그리기용
        return loss


class TrainerMetadata(TorchSerializable):

    def __init__(self):
        self.current_epoch = 0

        self.scores = list()
        self.best_score = 0

        self.last_losses = list()

    def state_dict_impl(self):
        return {
            'current_epoch': self.current_epoch + 1,
            'scores': self.scores,
            'best_score': self.best_score,
            'last_losses': self.last_losses,
            'agent': agent.state_dict()
        }

    def load_state_dict_impl(self, var_state):
        self.current_epoch = var_state['current_epoch']
        self.scores.extend(var_state['scores'])
        self.best_score = var_state['best_score']
        self.last_losses.extend(var_state['last_losses'])
        agent.load_state_dict(var_state['agent'])

    def save(self, checkpoint):
        # state_dict 구성 속도가 느리므로 필요할 때만 구성
        if checkpoint.is_saving_episode(self.current_epoch):
            var_state = self.state_dict()
            is_best = False
            if max(self.scores) > self.best_score:
                self.best_score = max(self.scores)
                is_best = True

            checkpoint.save_checkpoint(__file__, var_state, is_best)

    def load(self, checkpoint, viz):
        full_path = checkpoint.get_best_model_file_name(__file__)
        print("Loading checkpoint '{}'".format(full_path))
        var_state = checkpoint.load_model(full_path=full_path)
        self.load_state_dict(var_state)
        for e in range(0, len(self.scores)):
            viz.draw_line(x=e, y=self.scores[e], name='score')
            viz.draw_line(x=e, y=self.last_losses[e].item(), name='last_loss')
        print("Loading complete. Resuming from episode: {}, score: {:.2f}".format(self.current_epoch - 1, max(self.scores, default=0)))

    def finish_episode(self, viz, episode, score, last_loss):
        # 정책망의 가중치를 가져와서 타겟망의 가중치에 덮어씌운다
        agent.update_target_model()

        score = score.item()
        score = score if score == 500.0 else score + 100
        self.current_epoch = episode
        self.scores.append(score)
        self.last_losses.append(last_loss)

        if episode % LOG_INTERVAL == 0:
            print('Episode {}\tScore: {:.2f}\tMem Length: {}\tEpsilon: {}\tCompute Time: {:.2f}'.format(
                episode, score, len(agent.memory), agent.epsilon, time.time() - start_time))

            viz.draw_line(y=score, x=episode, name='score')
            viz.draw_line(y=last_loss.item(), x=episode, name='last_loss')
            viz.draw_line(y=agent.epsilon, x=episode, name='epsilon')


if __name__ == "__main__":
    VERSION = 1
    RENDER = True
    LOG_INTERVAL = 1
    IS_LOAD, IS_SAVE, SAVE_INTERVAL = False, True, 100
    EPISODES = 30000

    device = u.set_device(force_cpu=False)
    viz_env_name = os.path.basename(os.path.realpath(__file__))
    viz = Drawer(reset=True, env=viz_env_name)

    metadata = TrainerMetadata()
    checkpoint_inst = Checkpoint(VERSION, IS_SAVE, SAVE_INTERVAL)

    """
    상태 공간 4개, 범위 -∞ < s < ∞
    행동 공간 1개, 이산값 0 or 1
    """
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = DQNAgent(state_size, action_size)

    # 최대 에피소드 수만큼 돌린다
    for episode in range(metadata.current_epoch, EPISODES):
        start_time = time.time()
        state = env.reset()
        score = last_loss = u.t_float32(0)

        # 각 에피소드당 환경에 정의된 최대 스텝 수만큼 돌린다
        # 단 그 전에 환경에서 정의된 종료 상태(done)가 나오면 거기서 끝낸다
        for t in range(env.spec.max_episode_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done or score == 499 else -100

            agent.append_sample(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                last_loss = agent.train_model()

            score += reward
            state = next_state

            env.render() if RENDER else None
            if done: break

        metadata.finish_episode(viz, episode, score, last_loss)

        if IS_SAVE:
            metadata.save(checkpoint_inst)

        # 이전 10개 에피소드의 점수 평균이 490보다 크면 학습 중단
        if np.mean(metadata.scores[-min(10, len(metadata.scores)):]) > 490:
            sys.exit()
