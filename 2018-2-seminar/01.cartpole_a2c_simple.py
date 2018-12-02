# -*- coding: utf-8 -*-
# CartPole-v1 with A2C
#
# 참조한 소스
# https://github.com/rlcode/reinforcement-learning-kr/blob/master/2-cartpole/2-actor-critic/cartpole_a2c.py
# https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py
#

import sys
import time
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

import gym

import kdm_utils as u
from kdm_utils.checkpoint import Checkpoint, TorchSerializable
from kdm_utils.drawer import Drawer


# Python Pickle은 nested namedtuple save를 지원하지 않음
# https://stackoverflow.com/questions/4677012/python-cant-pickle-type-x-attribute-lookup-failed
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class A2CAgent(TorchSerializable):

    def __init__(self, state_size, action_size):
        self._set_hyper_parameters()

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


class TrainerMetadata(TorchSerializable):

    def __init__(self):
        self.current_epoch = 0

        self.scores = list()
        self.best_score = 0

        self.last_critic_losses = list()
        self.last_actor_losses = list()

    def state_dict_impl(self):
        return {
            'current_epoch': self.current_epoch + 1,
            'scores': self.scores,
            'best_score': self.best_score,
            'last_critic_losses': self.last_critic_losses,
            'last_actor_losses': self.last_actor_losses,
            'agent': agent.state_dict()
        }

    def load_state_dict_impl(self, var_state):
        self.current_epoch = var_state['current_epoch']
        self.scores.extend(var_state['scores'])
        self.best_score = var_state['best_score']
        self.last_critic_losses.extend(var_state['last_critic_losses'])
        self.last_actor_losses.extend(var_state['last_actor_losses'])
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
            viz.draw_line(x=e, y=self.last_actor_losses[e].item(), name='last_actor_loss')
            viz.draw_line(x=e, y=self.last_critic_losses[e].item(), name='last_critic_loss')
        print("Loading complete. Resuming from episode: {}, score: {:.2f}".format(self.current_epoch - 1, max(self.scores, default=0)))

    def finish_episode(self, viz, episode, score, last_actor_loss, last_critic_loss):
        score = score.item()
        score = score if score == 500.0 else score + 100
        self.current_epoch = episode
        self.scores.append(score)
        self.last_actor_losses.append(last_actor_loss)
        self.last_critic_losses.append(last_critic_loss)

        if episode % LOG_INTERVAL == 0:
            print('Episode {}\tScore: {:.2f}\tCompute Time: {:.2f}'.format(
                episode, score, time.time() - start_time))

            viz.draw_line(y=score, x=episode, name='score')
            viz.draw_line(y=last_actor_loss.item(), x=episode, name='last_actor_loss')
            viz.draw_line(y=last_critic_loss.item(), x=episode, name='last_critic_loss')


if __name__ == "__main__":
    VERSION = 1
    RENDER = True
    LOG_INTERVAL = 1
    IS_LOAD, IS_SAVE, SAVE_INTERVAL = False, True, 100
    EPISODES = 10000

    device = u.set_device(force_cpu=False)
    viz = Drawer(reset=True, env='main')

    metadata = TrainerMetadata()
    checkpoint_inst = Checkpoint(VERSION, IS_SAVE, SAVE_INTERVAL)

    """
    상태 공간 4개, 범위 -∞ < s < ∞
    행동 공간 1개, 이산값 0 or 1
    """
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    agent = A2CAgent(state_size, action_size)

    if IS_LOAD:
        metadata.load(checkpoint_inst, viz)

    # 최대 에피소드 수만큼 돌린다
    for episode in range(metadata.current_epoch, EPISODES):
        start_time = time.time()
        state = env.reset()
        score = last_actor_loss = last_critic_loss = u.t_float32(0)

        # 각 에피소드당 환경에 정의된 최대 스텝 수만큼 돌린다
        # 단 그 전에 환경에서 정의된 종료 상태(done)가 나오면 거기서 끝낸다
        for t in range(env.spec.max_episode_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            reward = reward if not done or score == 499 else -100

            last_actor_loss, last_critic_loss = agent.train_model(state, action, reward, next_state, done)

            score += reward
            state = next_state

            env.render() if RENDER else None
            if done: break

        metadata.finish_episode(viz, episode, score, last_actor_loss, last_critic_loss)

        if IS_SAVE:
            metadata.save(checkpoint_inst)

        # 이전 10개 에피소드의 점수 평균이 490보다 크면 학습 중단
        if np.mean(metadata.scores[-min(10, len(metadata.scores)):]) > 490:
            sys.exit()
