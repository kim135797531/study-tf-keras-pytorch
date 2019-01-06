# -*- coding: utf-8 -*-
# PPO
# Proximal Policy Optimization (Schulman et al. 2017)
#

import copy
import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autograd
from torch.nn.utils import parameters_to_vector

import utils_kdm as u
from utils_ext.kl_divergence import kl_divergence
from utils_ext.gae import GAE
from utils_kdm.trainer_metadata import TrainerMetadata


# Python Pickle은 nested namedtuple save를 지원하지 않음
# https://stackoverflow.com/questions/4677012/python-cant-pickle-type-x-attribute-lookup-failed

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done'))


class Actor(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()
        self.device = TrainerMetadata().device
        # 64, 64는 논문 저자 공식 레포지토리
        self.layer_sizes = [state_size, 64, 64, action_size]

        self.linear1 = nn.Linear(self.layer_sizes[0], self.layer_sizes[1])
        self.linear2 = nn.Linear(self.layer_sizes[1], self.layer_sizes[2])
        self.head = nn.Linear(self.layer_sizes[2], self.layer_sizes[3])

        # TODO: (개선) 초기화 이해 안 감 -> 공식 레포가 이렇지만 다른 초기화는?
        self.head.weight.data.mul_(0.1)
        self.head.bias.data.mul_(0.0)

    def forward(self, x):
        # TODO: (개선) 저자 공식 레포가 tanh지만 다른 거 써볼까
        # x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        # μ의 발음 표현 mu는 아무리 봐도 무 같아서
        # 내가 편하려고 meow로 씀ㅋ
        meow = self.head(x)

        # logstd 이렇게 반환하면 그냥 0 아닌가? RLKR이 이렇게 함
        # -> 비록 현재는 0 이지만, logstd를 지수함수 씌운 것이 std 변수로 남는다.
        # -> 이는 추후 그라디언트를 계산할 때 미분 등을 자동으로 연관시켜 할 수 있으므로,
        # -> 이런 식으로 자동 그래프가 구성되게 만든다.
        logstd = torch.zeros_like(meow)
        std = torch.exp(logstd)

        # DDPG의 actor와는 다르게 행동을 반환하는 것이 아니라 현재 정책 분포에 따른 기대값(μ)을 반환
        # -> 정책 분포의 평균과 표준편차를 반환하여 필요할 땐 정규분포에 의해 action 계산을 하고,
        # -> 후에 정규분포 특성에 의한 그라디언트 계산이 가능하게 한다.
        return meow, logstd, std


class Critic(nn.Module):

    def __init__(self, state_size):
        super().__init__()
        self.device = TrainerMetadata().device
        # 액션 크기는 안 받는 이유는? Q(s, a) 아닌가?
        # Q 함수 추정이 아니라 V (Value) 추정이다
        # 나중에 GAE 에서 이득(A) 계산할 때 V가 필요
        self.layer_sizes = [state_size, 64, 64, 1]

        self.linear1 = nn.Linear(self.layer_sizes[0], self.layer_sizes[1])
        self.linear2 = nn.Linear(self.layer_sizes[1], self.layer_sizes[2])
        self.head = nn.Linear(self.layer_sizes[2], self.layer_sizes[3])

        # TODO: 초기화 이해 안 감 -> 공식 레포가 이렇지만 다른 초기화는?
        self.head.weight.data.mul_(0.1)
        self.head.bias.data.mul_(0.0)

    def forward(self, x):
        # TODO: (개선) 저자 공식 레포가 tanh지만 다른 거 써볼까
        # x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        v = self.head(x)

        return v


class PPO(u.TorchSerializable):

    def __init__(self, state_size, action_size):
        super().__init__()

        self._set_hyper_parameters()
        self.device = TrainerMetadata().device

        # 기본 설정
        self.state_size, self.action_size = state_size, action_size

        # 모델 빌드
        self.actor = Actor(self.state_size, self.action_size).to(self.device)
        self.critic = Critic(self.state_size).to(self.device)

        # Optimizer
        self.actor_critic_optimizer = optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.learning_rate
        )

        self.transition_structure = Transition
        self.memory = list()

        self.gae = GAE(gamma=self.gae_gamma)

        self.register_serializable([
            'self.actor',
            'self.critic',
        ])

    def _set_hyper_parameters(self):
        self.learning_rate = 0.0003
        # Loss 클립할 때 지정할 범위, 0.2면 정책 비율을 0.8~1.2에서 유지
        # 0.2는 논문에서 실험적으로 이게 제일 좋았다고 함
        self.epsilon = 0.2
        self.num_epochs = 50
        self.batch_size = 64
        self.discount_factor = 0.99
        self.gae_gamma = 0.95

        # Early Stopping
        self.max_kl = 0.01

    def reset(self):
        self._memory_clear()

    def _memory_clear(self):
        self.memory.clear()

    def append_sample(self, sars, done):
        state, action, reward, next_state = sars
        state, action, reward = u.t_float32(state), u.t_float32(action), u.t_float32(reward)
        # FIXME: 이거 t_uint8로도 할 수 있을텐데 GAE 파트에서 실수 값에 Byte 곱한다고 에러 뿜뿜
        done = u.t_float32(done)
        transition = self.transition_structure(state, action, reward, done)
        self.memory.append(transition)

    def get_action(self, state):
        # 왜 액터에서 바로 안 구하고 뮤랑 표준편차 꺼내서 다시 계산? 모듈화 때문인가?
        # -> 이 알고리즘에서의 actor는 표준분포에 의한 행동을 뽑아내는게 아니라 표준분포 그 자체를 생성한다
        # -> 생성한 표준 분포를 따라서 행동을 하나 샘플
        state = u.t_from_np_to_float32(state)
        meow, logstd, std = self.actor(state)
        meow, std = meow.detach(), std.detach()
        action = torch.normal(meow, std).cpu().numpy()
        return action

    def _improve_ratio(self, old_policy, new_policy):
        # 현재 정책과 과거 정책의 비율, 즉 r(θ_old) = 1
        return torch.exp(new_policy - old_policy)

    def _surrogate_loss(self, old_policy, new_policy, advantage_batch):
        # CPI = Conservative Policy Iteration
        improve_ratio = self._improve_ratio(old_policy, new_policy)
        clipped_ratio = torch.clamp(improve_ratio, 1 - self.epsilon, 1 + self.epsilon)

        TrainerMetadata().log(torch.max(improve_ratio),
                              'improve_ratio', 'max', show_only_last=True, compute_maxmin=True)
        TrainerMetadata().log(torch.min(improve_ratio),
                              'improve_ratio', 'min', show_only_last=True, compute_maxmin=True)
        # TODO: advantage 나중에 곱해보기?
        loss_cpi = improve_ratio * advantage_batch
        loss_clip = clipped_ratio * advantage_batch
        loss = torch.min(loss_cpi, loss_clip).mean()
        return loss.to(self.device)

    def get_critic_loss(self, v_batch, return_batch):
        critic_loss = nn.MSELoss().to(self.device)
        critic_loss = critic_loss(v_batch, return_batch)

        return critic_loss

    # TODO: OpenAI SpinningUp에서는 이거 안 썼다??
    def _actor_critic_loss(self, actor_loss, critic_loss):
        # Actor와 Critic이 가중치 공유 안 할 때에는 c_1 = 1 (독립이므로)
        c_1 = 1
        # 논문에서도 Entropy Bonus 적용 안 한게 제일 성능이 좋았다고 한다
        entropy_bonus = 0
        # TODO: 음수 양수 처리;;
        actor_critic_loss = actor_loss + (c_1 * critic_loss) + entropy_bonus
        return actor_critic_loss

    def _log_density(self, x, meow, std, logstd):
        # 확률밀도함수에 log 씌운 것을 반환
        #
        # 예를 들어 TRPO에서 그라디언트 계산할 때 정책(a|s),
        # 즉 확률밀도함수를 이용해서 현재 행동(x)을 했을 확률이 필요하다.
        # 그런데 이거 그냥 구해서 추후 미분하는 것 보단 애초에 확률밀도함수의 log를 구해서 던지는게
        # 계산이 쉽다고 한다 (잘 모르겟지만 위키백과가 그럼)
        # https://ko.wikipedia.org/wiki/%EA%B0%80%EB%8A%A5%EB%8F%84
        var = std.pow(2)
        log_density = -(x - meow).pow(2) / (2 * var) \
                      - 0.5 * math.log(2 * math.pi) - logstd
        return log_density.sum(1, keepdim=True).to(self.device)

    def train_model(self):
        transitions = self.memory
        sar_batch = self.transition_structure(*zip(*transitions))
        s_batch = torch.stack(sar_batch.state).to(self.device)
        a_batch = torch.stack(sar_batch.action).to(self.device)
        r_batch = torch.stack(sar_batch.reward).to(self.device)
        done_batch = torch.stack(sar_batch.done).to(self.device)

        # 현재 가치 함수 (V)를 기반으로 추정 advantage (A) 구하기 (GAE)
        v_batch = self.critic(s_batch)
        return_batch, advantage_batch = self.gae.get_return_advantage(r_batch, done_batch, v_batch)

        # 그라디언트 안 구하고 이렇게 원본 복사해서 detach 해서 구하는 아이디어가 맞는 방법?
        # ->　맞음. 그러나 애초에 상태 저장할 때 분포랑 이득까지 저장하는 방법도 있음.
        old_actor = copy.deepcopy(self.actor)
        meow, logstd, std = old_actor(s_batch)
        meow = meow.detach()
        logstd = logstd.detach()
        std = std.detach()
        # PPO도 log 씌운 확률분포로 구해도 됨
        old_policy = self._log_density(a_batch, meow, std, logstd)

        n = len(s_batch)
        arr = np.arange(n)
        for epoch in range(self.num_epochs):
            np.random.shuffle(arr)

            for i in range(n // self.batch_size):
                batch_index = arr[self.batch_size * i: self.batch_size * (i + 1)]
                batch_index = u.t_long(batch_index)

                sampled_s_batch = s_batch[batch_index]
                sampled_a_batch = a_batch[batch_index]
                sampled_return_batch = return_batch[batch_index]
                sampled_advantage_batch = advantage_batch[batch_index]

                # TODO: 엔트로피 넣기
                sampled_v_batch = self.critic(sampled_s_batch)
                meow, logstd, std = self.actor(sampled_s_batch)
                new_policy = self._log_density(sampled_a_batch, meow, std, logstd)

                surrogate_loss = -1 * self._surrogate_loss(
                    old_policy=old_policy[batch_index],
                    new_policy=new_policy,
                    advantage_batch=sampled_advantage_batch
                )

                critic_loss = self.get_critic_loss(sampled_v_batch, sampled_return_batch)
                loss = self._actor_critic_loss(surrogate_loss, critic_loss)
                self.actor_critic_optimizer.zero_grad()
                loss.backward()
                self.actor_critic_optimizer.step()

                TrainerMetadata().log(surrogate_loss, 'actor_loss', show_only_last=True, compute_maxmin=True)
                TrainerMetadata().log(critic_loss, 'critic_loss', show_only_last=True, compute_maxmin=True)
                TrainerMetadata().log(loss, 'loss', show_only_last=True, compute_maxmin=True)

            kl = kl_divergence(new_actor=self.actor, old_actor=old_actor, s_batch=s_batch)
            kl = kl.mean()
            TrainerMetadata().log(kl, 'KL', 'current_kl', compute_maxmin=True)
            TrainerMetadata().log(self.max_kl, 'KL', 'max_kl')

            if kl > self.max_kl:
                TrainerMetadata().log(epoch, 'early_stopped_epoch', compute_maxmin=True)
                break
