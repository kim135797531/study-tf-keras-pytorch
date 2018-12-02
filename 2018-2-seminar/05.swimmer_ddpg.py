# -*- coding: utf-8 -*-
# Swimmer-v2 with DDPG
#

import os
import time
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import gym

import kdm_utils as u
from kdm_utils.checkpoint import Checkpoint, TorchSerializable
from kdm_utils.drawer import Drawer
from kdm_utils.noise import OrnsteinUhlenbeckNoise
from kdm_utils.replay_memory import ReplayMemory


# Python Pickle은 nested namedtuple save를 지원하지 않음
# https://stackoverflow.com/questions/4677012/python-cant-pickle-type-x-attribute-lookup-failed
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class Actor(nn.Module):

    def __init__(self, state_size, action_size, action_range=(-1, 1)):
        super(Actor, self).__init__()
        self.layer_sizes = [state_size, 400, 300, action_size]
        # self.layer_size = [state_size, 256, 128, action_size]

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
        x = self.head(torch.tanh(x))
        # TODO: 값 잘라야 하나?
        return torch.clamp(x, min=self.action_low, max=self.action_high).to(device)


class Critic(nn.Module):

    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.layer_sizes = [state_size + action_size, 400, 300, action_size]

        self.linear1 = nn.Linear(self.layer_sizes[0], self.layer_sizes[1])
        self.linear2 = nn.Linear(self.layer_sizes[1], self.layer_sizes[2])
        self.head = nn.Linear(self.layer_sizes[2], self.layer_sizes[3])
        u.fanin_init(self.linear1.weight)
        u.fanin_init(self.linear2.weight)
        nn.init.uniform_(self.head.weight, a=-3*10e-4, b=3*10e-4)

    # Actions were not included until the 2nd Hidden layer of Q
    # TODO: 해석 불분명: **2번째 은닉층까지 안 넣었다**
    # (a) 2번째 직전까진 안 넣었으니 2번째에 넣었을 것이다
    # (b) 2번째 까지는 안 넣었으니 3번째에 넣었을 것이다
    def forward(self, state, action):
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # TODO: torch.clamp (v -mean) 으로 정규화 필요
        return self.head(x)


class DDPGAgent(TorchSerializable):
    def __init__(self, state_size, action_size, action_range=(-1, 1)):
        self._set_hyper_parameters()

        # 기본 설정
        self.state_size = state_size
        self.action_size = action_size

        # TODO: 정규화된 입력인지 검사 문구 넣고 range 빼기
        self.action_low, self.action_high = action_range

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
        self.transition_structure = Transition
        self.memory = ReplayMemory(self.memory_maxlen, self.transition_structure)

        # 오른스타인-우렌벡 과정
        self.noise = OrnsteinUhlenbeckNoise(self.action_size)

    def _set_hyper_parameters(self):
        # Adam 하이퍼 파라미터
        self.learning_rate_actor = 0.0001
        self.learning_rate_critic = 0.001
        # TODO: 뭔 뜻인지 찾아보자
        self.l2_weight_decay = 0.01

        # 평가망 학습 하이퍼 파라미터
        self.discount_factor = 0.99

        # 타겟망 덮어 씌우기 하이퍼 파라미터
        self.soft_target_update_tau = 0.001

        # 리플레이 메모리 관련
        self.batch_size = 128
        # self.memory_maxlen = int(10e+6)
        self.memory_maxlen = int(300000)
        self.train_start = 2000

    def state_dict_impl(self):
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

    def load_state_dict_impl(self, var_state):
        self.memory.load_state_dict(var_state['memory'])
        self.actor.load_state_dict(var_state['actor'])
        self.critic.load_state_dict(var_state['critic'])
        self.target_actor.load_state_dict(var_state['target_actor'])
        self.target_critic.load_state_dict(var_state['target_critic'])
        self.actor_optimizer.load_state_dict(var_state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(var_state['critic_optimizer'])
        self.noise.load_state_dict(var_state['noise'])

    def get_action(self, state):
        state = u.t_from_np_to_float32(state)
        noise = self.noise.sample()
        action = self.actor(state).detach().cpu().numpy()
        action += noise
        # TODO: 이렇게 하는게 맞나?
        return np.clip(action, a_min=self.action_low, a_max=self.action_high)
        # return action

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.push(
            u.t_float32(state),
            u.t_float32(action),
            u.t_float32(reward),
            u.t_float32(next_state),
            u.t_uint8(done)
        )

    def train_model(self):
        # 메모리에서 일정 크기만큼 기억을 불러온다
        # 그 후 기억을 모아 각 변수별로 모은다. (즉, 전치행렬)
        transitions = self.memory.sample(self.batch_size)
        batch = self.transition_structure(*zip(*transitions))

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

        # 평가망의 예측 보상과 타겟 평가망의 예측 보상을 MSE 비교 후 업데이트
        self.critic_optimizer.zero_grad()
        critic_loss = nn.MSELoss().to(device)
        critic_loss = critic_loss(expected_rewards, predicted_rewards)
        critic_loss.backward()
        self.critic_optimizer.step()

        # <정책망 최적화>
        self.actor_optimizer.zero_grad()
        predicted_actions = self.actor(state_batch)
        q_output = self.critic(state_batch, predicted_actions)
        # actor_loss = -1*torch.sum(q_output).to(device)
        # TODO: sum 이 아니라 mean 인 이유
        actor_loss = -1 * torch.mean(q_output).to(device)

        # 정책망의 예측 보상을 정책 그라디언트로 업데이트
        actor_loss.backward()
        self.actor_optimizer.step()

        # 현재 평가망, 정책망의 가중치를 타겟 평가망에다 덮어쓰기
        u.soft_update_from_to(src_nn=self.critic, dst_nn=self.target_critic, tau=self.soft_target_update_tau)
        u.soft_update_from_to(src_nn=self.actor, dst_nn=self.target_actor, tau=self.soft_target_update_tau)

        # 그래프 그리기용
        # viz.draw_line(y=critic_loss.item(), interval=10, name="critic_loss")
        # viz.draw_line(y=actor_loss.item(), interval=10, name="actor_loss")
        # viz.step()
        return critic_loss, actor_loss


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
        self.current_epoch = episode
        self.scores.append(score)
        self.last_actor_losses.append(last_actor_loss)
        self.last_critic_losses.append(last_critic_loss)

        if episode % LOG_INTERVAL == 0:
            print('Episode {}\tScore: {:.2f}\tMem Length: {}\tCompute Time: {:.2f}'.format(
                episode, score, len(agent.memory), time.time() - start_time))

            viz.draw_line(y=score, x=episode, name='score')
            viz.draw_line(y=last_actor_loss.item(), x=episode, name='last_actor_loss')
            viz.draw_line(y=last_critic_loss.item(), x=episode, name='last_critic_loss')


if __name__ == "__main__":
    VERSION = 2
    # TODO: 시드 넣기
    # env.seed(args.seed)
    # torch.manual_seed(args.seed)
    RENDER = True
    LOG_INTERVAL = 1
    IS_LOAD, IS_SAVE, SAVE_INTERVAL = False, True, 400
    EPISODES = 30000

    device = u.get_device(force_cpu=False)
    viz = Drawer(reset=True, env='main')

    metadata = TrainerMetadata()
    checkpoint_inst = Checkpoint(VERSION, IS_SAVE, SAVE_INTERVAL)

    """
    상태 공간 8개, 범위 -∞ < s < ∞
    행동 공간 2개, 범위 -1 < a < 1
    """
    env = gym.make('Swimmer-v2')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    agent = DDPGAgent(state_size, action_size)

    if IS_LOAD:
        metadata.load(checkpoint_inst, viz)

    # 최대 에피소드 수만큼 돌린다
    for episode in range(metadata.current_epoch, EPISODES):
        start_time = time.time()
        agent.noise.reset()
        state = env.reset()
        score = last_actor_loss = last_critic_loss = u.t_float32(0)

        # 각 에피소드당 환경에 정의된 최대 스텝 수만큼 돌린다
        # 단 그 전에 환경에서 정의된 종료 상태(done)가 나오면 거기서 끝낸다
        for t in range(env.spec.max_episode_steps):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            agent.append_sample(state, action, reward, next_state, done)

            if len(agent.memory) >= agent.train_start:
                last_critic_loss, last_actor_loss = agent.train_model()

            score += reward
            state = next_state

            env.render() if RENDER else None
            if done: break

        metadata.finish_episode(viz, episode, score, last_actor_loss, last_critic_loss)

        if IS_SAVE:
            metadata.save(checkpoint_inst)

        if score > env.spec.reward_threshold:
            print("Solved! Running reward is now {}".format(score))
            break
