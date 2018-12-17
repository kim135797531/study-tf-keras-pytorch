# -*- coding: utf-8 -*-
# DDPG

from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
import torch.optim as optim

import utils_kdm as u
from utils_kdm.noise import OrnsteinUhlenbeckNoise
from utils_kdm.replay_memory import ReplayMemory
from utils_kdm.trainer_metadata import TrainerMetadata

# Python Pickle은 nested namedtuple save를 지원하지 않음
# https://stackoverflow.com/questions/4677012/python-cant-pickle-type-x-attribute-lookup-failed

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class Actor(nn.Module):

    def __init__(self, state_size, action_size, action_range=(-1, 1)):
        super(Actor, self).__init__()
        self.device = TrainerMetadata().device
        self.layer_sizes = [state_size, 400, 300, action_size]

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
        return torch.clamp(x, min=self.action_low, max=self.action_high).to(self.device)


class Critic(nn.Module):

    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.device = TrainerMetadata().device
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
        # TODO: 매 번 cat 하지 말고 메모리에 넣을 때 state, action 붙여서 넣기 (꼼수)
        x = torch.cat((state, action), dim=1).to(self.device)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        # TODO: torch.clamp (v -mean) 으로 정규화 필요
        return self.head(x)


class DDPG(u.TorchSerializable):

    def __init__(self, state_size, action_size, action_range=(-1, 1)):
        self._set_hyper_parameters()
        self.device = TrainerMetadata().device

        # 기본 설정
        self.state_size, self.action_size = state_size, action_size
        # TODO: 정규화된 입력인지 검사 문구 넣고 range 빼기
        self.action_low, self.action_high = action_range

        # 모델 빌드
        self.actor = Actor(self.state_size, self.action_size).to(self.device)
        self.critic = Critic(self.state_size, self.action_size).to(self.device)
        self.target_actor = Actor(self.state_size, self.action_size).to(self.device)
        self.target_critic = Critic(self.state_size, self.action_size).to(self.device)

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
        # DQN, DDPG에서 제안하고 쓰는 개념이므로 정의는 따로 두더라도 인스턴스는 알고리즘 내부에서 갖고 있는다
        self.transition_structure = Transition
        self.memory = ReplayMemory(self.memory_maxlen, self.transition_structure)

        # 오른스타인-우렌벡 과정
        self.noise = OrnsteinUhlenbeckNoise(self.action_size)

    def _set_hyper_parameters(self):
        # Adam 하이퍼 파라미터
        self.learning_rate_actor = 0.0001
        self.learning_rate_critic = 0.001

        # 그냥 오차함수에 가중치의 제곱합을 더한 뒤, 이것을 최소화 하는 기술
        # E(w) = mean(E(w)+ λ/2*(|w|^2))
        # 이렇게 하면 조금이라도 더 작은 가중치가 선호되게 된다
        self.l2_weight_decay = 0.01

        # 평가망 학습 하이퍼 파라미터
        self.discount_factor = 0.99

        # 타겟망 덮어 씌우기 하이퍼 파라미터
        self.soft_target_update_tau = 0.001

        # 리플레이 메모리 관련
        self.batch_size = 128
        # self.memory_maxlen = int(1e+6)
        self.memory_maxlen = 750000
        self.train_start = 2000

    def state_dict_impl(self):
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'target_actor': self.target_actor.state_dict(),
            'target_critic': self.target_critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'memory': self.memory.state_dict(),
            'noise': self.noise.state_dict()
        }

    def load_state_dict_impl(self, var_state):
        self.actor.load_state_dict(var_state['actor'])
        self.critic.load_state_dict(var_state['critic'])
        self.target_actor.load_state_dict(var_state['target_actor'])
        self.target_critic.load_state_dict(var_state['target_critic'])
        self.actor_optimizer.load_state_dict(var_state['actor_optimizer'])
        self.critic_optimizer.load_state_dict(var_state['critic_optimizer'])
        self.memory.load_state_dict(var_state['memory'])
        self.noise.load_state_dict(var_state['noise'])

    def reset(self):
        self.noise.reset()

    def append_sample(self, sars, done):
        state, action, reward, next_state = sars
        self.memory.push(
            u.t_float32(state),
            u.t_float32(action),
            u.t_float32(reward),
            u.t_float32(next_state),
            u.t_uint8(done)
        )

    def get_action(self, state):
        state = u.t_from_np_to_float32(state)
        noise = self.noise.sample()
        action = self.actor(state).detach().cpu().numpy()
        action += noise
        # TODO: 이렇게 하는게 맞나?
        return np.clip(action, a_min=self.action_low, a_max=self.action_high)
        # return action

    def get_critic_loss(self, s_batch, a_batch, r_batch, next_s_batch):
        # <평가망(critic) 최적화>
        # (무엇을, 어디서, 어떻게, 왜)
        # 각각의 기억에 대해, 타겟 정책망에, 다음 상태를 넣어서, 다음 타겟 액션을 구한다.
        # 각각의 기억에 대해, 타겟 평가망에, 다음 상태와 타겟 액션을 넣어서, 다음 타겟 보상을 구한다.
        # (s+1) => (a+1)
        # (s+1), (a+1) => (r+1)'
        target_actions = self.target_actor(next_s_batch)
        target_rewards = self.target_critic(next_s_batch, target_actions)
        # (보상) + (다음 보상)' = 기대하는 보상 (재귀적으로 생각하자 = 큐러닝 기본 상기하기)
        # (보상)'             = 예측한 보상
        # r + (r+1)'
        # r'
        expected_rewards = r_batch.unsqueeze(dim=1) + self.discount_factor * target_rewards
        predicted_rewards = self.critic(s_batch, a_batch)

        critic_loss = nn.MSELoss().to(self.device)
        critic_loss = critic_loss(expected_rewards, predicted_rewards)

        return critic_loss

    def get_actor_loss(self, s_batch):
        # <정책망(actor) 최적화>
        # 무엇을? = 현재 상태와 예측한 액션을 이용해서, Q함수 예측값을 최소화 하자 (0 만드는게 아니다 음수로 쭉쭉 가즈아)
        # 어떻게? = 최소가 나오게 하는 액션을 예측하게 만들어서 (액션망 가중치(+편향)로 그라디언트)
        #
        # 이것은
        #     조건: s, a=µ(s|θµ)
        #     식:  ∇θµ[Q(s,a|θ)] 과 같다
        #
        #     조건: 상태, 액션=정책망(상태) (정책망 가중치=θµ)
        #     식:  평가망(상태, 액션) (평가망 가중치=θ) 일 때, 정책망 가중치로 그라디언트 구하기
        #
        # 다시 표현하면 이것은
        #     조건: s, a=
        #               조건: s
        #               식:   ∇θµ[µ(s|θµ)]
        #     식:  ∇a[Q(s,a|θ)] 과 같다
        #
        #     조건: 상태, 액션=
        #                   조건: 상태
        #                   식:   정책망(상태) (정책망 가중치=θµ) 일 때, 정책망 가중치로 그라디언트 구하기
        #     식:   평가망(상태, 액션) (평가망 가중치=θ) 일 때, 액션으로 그라디언트 구하기
        # µ(s|θµ)
        predicted_actions = self.actor(s_batch)
        # Q(s,a|θ)
        q_output = self.critic(s_batch, predicted_actions)

        # sum 이 아니라 mean 인 이유
        # -> sum이든 mean이든 똑같으나 (N은 같으므로)
        # -> sum 했을 때 값이 많이 커지니까 그냥 보기 좋게 mean으로..
        # actor_loss = -1*torch.sum(q_output).to(self.device)
        actor_loss = -1 * torch.mean(q_output).to(self.device)

        return actor_loss

    def train_model(self, sars, done):
        # 메모리에서 일정 크기만큼 기억을 불러온다
        # 그 후 기억을 모아 각 변수별로 모은다. (즉, 전치행렬)
        # TODO: random과 zip func 글카에서 하기
        transitions = self.memory.sample(self.batch_size)
        # SARS = State, Action, Reward, next State
        sars_batch = self.transition_structure(*zip(*transitions))

        # TODO: 이거 튜플로 묶으면 다시 GPU에서 CPU로 오나?
        # 텐서의 집합에서 고차원 텐서로
        # tuple(tensor, ...) -> tensor()
        s_batch = torch.cat(sars_batch.state).to(self.device)
        a_batch = torch.cat(sars_batch.action).to(self.device)
        r_batch = torch.cat(sars_batch.reward).to(self.device)
        next_s_batch = torch.cat(sars_batch.next_state).to(self.device)

        self.critic_optimizer.zero_grad()
        critic_loss = self.get_critic_loss(s_batch, a_batch, r_batch, next_s_batch)
        # 예측한 보상과 향후 기대하는 보상을 MSE 비교 후 업데이트
        #   ||r' - [r + (r+1)']|| = 0
        # ∴ ||r' - (r+1)'      || = r  (현재 Q함수와 다음 Q함수 차이가 딱 실제 보상이 되도록 학습)
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss = self.get_actor_loss(s_batch)
        # 정책망의 예측 보상을 정책 그라디언트로 업데이트
        # ∇θµ[Q(s,a|θ)] ∇θµ[µ(s|θµ)]
        actor_loss.backward()
        self.actor_optimizer.step()

        # 현재 평가망, 정책망의 가중치를 타겟 평가망에다 덮어쓰기
        u.soft_update_from_to(src_nn=self.critic, dst_nn=self.target_critic, tau=self.soft_target_update_tau)
        u.soft_update_from_to(src_nn=self.actor, dst_nn=self.target_actor, tau=self.soft_target_update_tau)

        if done:
            TrainerMetadata().log(critic_loss, 'critic_loss', show_only_last=False)
            TrainerMetadata().log(actor_loss, 'actor_loss', show_only_last=False)


