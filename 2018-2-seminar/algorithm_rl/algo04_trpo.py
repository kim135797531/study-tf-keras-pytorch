# -*- coding: utf-8 -*-
# TRPO
# Trust Region Policy Optimization (Schulman et al. 2015)
#
# 참조: https://github.com/reinforcement-learning-kr/pg_travel/blob/master/mujoco/agent/trpo_gae.py
#
import math
import random
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim

import utils_kdm as u
from utils_kdm.replay_memory import ReplayMemory
from utils_kdm.trainer_metadata import TrainerMetadata


# Python Pickle은 nested namedtuple save를 지원하지 않음
# https://stackoverflow.com/questions/4677012/python-cant-pickle-type-x-attribute-lookup-failed

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done'))


# TODO: 논문 저자는 어떤 Actor Critic 을 썼나?
# 논문 저자:
# 입력: 상태
# 출력: 값 = 가우시안 분포를 이루게 하고 평균 뽑기
#           => 상관계수 행렬이 대각행렬이고 상태와 독립
# 신경망이 로그정규분포를 갖게
# => 따라서 정책이 정규분포(평균=신경망, 표준편차=exp(로그표준분포))를 갖게
#
# reinforcement-learning-kr:
# Actor: 입력->64->64->출력
# Critic: 입력->64->64->1
#     gamma = 0.99
#     lamda = 0.98
#     hidden = 64
#     critic_lr = 0.0003
#     actor_lr = 0.0003
#     batch_size = 64
#     l2_rate = 0.001
#     max_kl = 0.01
#     clip_param = 0.2
#
# OpenAI:
#
# Actor:
# Critic:
#     critic_lr = 0.001
#     steps_per_epoch=4000,
# epochs=50, gamma=0.99,
# delta=0.01 (이거 max_kl), vf_lr=0.001,
# train_v_iters=80,
# damping_coeff=0.1,
# cg_iters=10,
# backtrack_iters=10,
# backtrack_coeff=0.8,
# lam=0.97,
# max_ep_len=1000,
# save_freq=10
class Actor(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()
        self.device = TrainerMetadata().device
        # 64, 64는 논문 저자 공식 레포지토리
        self.layer_sizes = [state_size, 64, 64, action_size]

        self.linear1 = nn.Linear(self.layer_sizes[0], self.layer_sizes[1])
        self.linear2 = nn.Linear(self.layer_sizes[1], self.layer_sizes[2])
        self.head = nn.Linear(self.layer_sizes[2], self.layer_sizes[3])

        # TODO: 초기화 이해 안 감 -> 공식 레포가 이렇지만 다른 초기화는?
        self.head.weight.data.mul_(0.1)
        self.head.bias.data.mul_(0.0)

    def forward(self, x):
        # TODO: 저자 공식 레포가 tanh지만 다른 거 써볼까
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        # μ의 발음 표현 mu는 아무리 봐도 무 같아서
        # 내가 편하려고 meow로 씀ㅋ
        meow = self.head(x)

        # TODO: logstd 이렇게 반환하면 그냥 0 아닌가? RLKR이 이렇게 함
        logstd = torch.zeros_like(meow)
        std = torch.exp(logstd)

        # TODO: 원본은 logstd도 반환하는데 난 납득하기 전까지 반환 X
        # TODO: DDPG의 actor와는 다르게 행동을 반환하는 것이 아니라 현재 정책 분포의 평균만을 반환?
        # 후에 더 처리하나?
        return meow, std


class Critic(nn.Module):

    def __init__(self, state_size):
        super().__init__()
        self.device = TrainerMetadata().device
        # TODO: 액션 크기는 안 받는 이유는? Q(s, a) 아닌가?
        self.layer_sizes = [state_size, 64, 64, 1]

        self.linear1 = nn.Linear(self.layer_sizes[0], self.layer_sizes[1])
        self.linear2 = nn.Linear(self.layer_sizes[1], self.layer_sizes[2])
        self.head = nn.Linear(self.layer_sizes[2], self.layer_sizes[3])

        # TODO: 초기화 이해 안 감 -> 공식 레포가 이렇지만 다른 초기화는?
        self.head.weight.data.mul_(0.1)
        self.head.bias.data.mul_(0.0)

    def forward(self, x):
        # TODO: 저자 공식 레포가 tanh지만 다른 거 써볼까
        x = torch.tanh(self.linear1(x))
        x = torch.tanh(self.linear2(x))
        v = self.head(x)

        return v


class TRPO(u.TorchSerializable):

    def __init__(self, state_size, action_size):
        super().__init__()

        self._set_hyper_parameters()
        self.device = TrainerMetadata().device

        # 기본 설정
        self.state_size, self.action_size = state_size, action_size

        # 모델 빌드
        self.actor = Actor(self.state_size, self.action_size).to(self.device)
        self.critic = Critic(self.state_size).to(self.device)

        # TODO: ZFilter?
        # pass

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

        self.transition_structure = Transition
        self.memory = list()

        self.register_serializable([
            'self.actor',
            'self.critic',
            'self.actor_optimizer',
            'self.critic_optimizer',
        ])

    def _set_hyper_parameters(self):
        # TODO: 잘 정하기 공식 레포엔 없나??
        # Adam 하이퍼 파라미터
        self.learning_rate_actor = 0.0003
        self.learning_rate_critic = 0.0003
        self.l2_weight_decay = 0.001

        # value 함수 학습할 때 전체 스텝 (4000) 을 64개로 잘라서 학습
        # TODO: 필요한가?
        self.batch_size = 64

        # From OpenAI
        # GAE 에서 쓰는 하이퍼 파라미터 감마
        self.gamma = 0.99
        # 켤레 기울기법(Conjugate Gradient) 몇 번 돌 것인가
        self.cg_iters = 10
        # line search 최대 몇 번 할 것인가 / 얼마씩 줄여 갈 것인가
        self.backtrack_iters = 10
        self.backtrack_coeff = 0.8

        # From RLKR
        # KL 다이버전스 한계값 (신뢰 영역 범위)
        self.max_kl = 0.01
        # value 함수 학습을 같은 데이터에 대해 몇 번 할 것인가
        self.train_v_iters = 10

    def reset(self):
        pass

    def memory_clear(self):
        self.memory.clear()

    def append_sample(self, sar, done):
        state, action, reward = sar
        state, action, reward = u.t_float32(state), u.t_float32(action), u.t_float32(reward)
        # TODO: 이거 t_uint8로도 할 수 있을텐데 GAE 파트에서 실수 값에 Byte 곱한다고 에러 뿜뿜
        done = u.t_float32(done)
        transition = self.transition_structure(state, action, reward, done)
        self.memory.append(transition)

    def get_action(self, state):
        # TODO: 왜 액터에서 바로 안 구하고 뮤랑 표준편차 꺼내서 다시 계산? 모듈화 때문인가?
        state = u.t_from_np_to_float32(state)
        meow, std = self.actor(state)
        meow, std = meow.detach(), std.detach()
        action = torch.normal(meow, std).cpu().numpy()
        return action

    def _gae(self, r_batch, done_batch, v_batch):
        return_batch = torch.zeros_like(r_batch).to(self.device)
        advantage_batch = torch.zeros_like(r_batch).to(self.device)

        running_return_batch = 0
        previous_value = 0
        running_advantage_batch = 0

        for t in reversed(range(0, len(r_batch))):
            running_return_batch = r_batch[t] + self.gamma * running_return_batch * done_batch[t]
            running_tderror = r_batch[t] + self.gamma * previous_value * done_batch[t] - \
                              v_batch.data[t]
            running_advantage_batch = running_tderror + self.gamma * self.gamma * \
                              running_advantage_batch * done_batch[t]

            return_batch[t] = running_return_batch
            previous_value = v_batch.data[t]
            advantage_batch[t] = running_advantage_batch

        advantage_batch = (advantage_batch - advantage_batch.mean()) / advantage_batch.std()
        return return_batch, advantage_batch

    def get_critic_loss(self, s_batch, return_batch, advantage_batch):
        critic_loss = nn.MSELoss().to(self.device)
        # TODO: RLKR 에서는 전체 메모리를 64개씩 랜덤 순서로 훑어가며 학습시킴
        # TODO: 난 그냥 통째로 (기본 STEPS_PER_EPOCH = 4000개) 때려 박을 예정ㅋ 느리면 바꾸지 모
        v_batch = self.critic(s_batch)
        critic_loss = critic_loss(v_batch, return_batch + advantage_batch)

        return critic_loss

    # TODO: 진짜 이 함수는 어디서 튀어나온거야 논문에 눈 씻고 찾아봐도 없음
    def _log_density(self, x, meow, std, logstd):
        var = std.pow(2)
        log_density = -(x - meow).pow(2) / (2 * var) \
                      - 0.5 * math.log(2 * math.pi) - logstd
        return log_density.sum(1, keepdim=True).to(self.device)

    # TODO: 진짜 이 함수는 어디서 튀어나온거야 논문에 눈 씻고 찾아봐도 없음
    def _surrogate_loss(self, advantage_batch, s_batch, old_policy, a_batch):
        # TODO: 왜 이거 여기서 다시 하지? 바로 앞에서 했는데
        meow, std = self.actor(s_batch)
        # TODO: logstd 이렇게 만들면 그냥 0 아닌가? RLKR이 이렇게 함
        logstd = torch.zeros_like(meow).to(self.device)
        new_policy = self._log_density(a_batch, meow, std, logstd)

        surrogate = advantage_batch * torch.exp(new_policy - old_policy).to(self.device)
        return surrogate.mean()

    # TODO: 이것도 베껴오긴 했는데, list로 바꿨다가 다시 cat 하지 말고 바로 하는 방법?
    def _flat_grad(self, grads):
        grad_flatten = []
        for grad in grads:
            grad_flatten.append(grad.view(-1))
        grad_flatten = torch.cat(grad_flatten).to(self.device)
        return grad_flatten

    # TODO: 이것도 베껴오긴 했는데, list로 바꿨다가 다시 cat 하지 말고 바로 하는 방법?
    def _flat_hessian(self, hessians):
        hessians_flatten = []
        for hessian in hessians:
            hessians_flatten.append(hessian.contiguous().view(-1))
        # TODO: 뒤에 data 붙여야 하나 (torch.cat().data?)
        hessians_flatten = torch.cat(hessians_flatten).to(self.device)
        return hessians_flatten

    # TODO: 이것도 베껴오긴 했는데, list로 바꿨다가 다시 cat 하지 말고 바로 하는 방법?
    def _flat_params(self, model):
        params = []
        for param in model.parameters():
            params.append(param.data.view(-1))
        params_flatten = torch.cat(params).to(self.device)
        return params_flatten

    # TODO: 이것도 베껴오긴 했는데, list로 바꿨다가 다시 cat 하지 말고 바로 하는 방법?
    # TODO: 결국 위에 params 갖고 노는 것들은 모델 파라미터를 갱신을 위한 거 였구나.
    # TODO: 이렇게 빼서 안 하고 아름답게 못 하나? CPU<->GPU 통신 불탄다
    def _update_model(self, model, new_params):
        index = 0
        for params in model.parameters():
            params_length = len(params.view(-1))
            new_param = new_params[index: index + params_length]
            new_param = new_param.view(params.size())
            params.data.copy_(new_param)
            index += params_length

    # TODO: 논문에서 다시 공부하기
    # TODO: residual_tol 변수 의미
    # RLKR에서 차용한 걸 다시 차용
    # from openai baseline code
    # https://github.com/openai/baselines/blob/master/baselines/common/cg.py
    def _conjugate_gradient(self, s_batch, loss_grad_data, nsteps=10, residual_tol=1e-10):
        x = torch.zeros(loss_grad_data.size()).to(self.device)
        r = loss_grad_data.clone()
        p = loss_grad_data.clone()
        rdotr = torch.dot(r, r).to(self.device)
        for i in range(nsteps):
            _Avp = self._fisher_vector_product(s_batch, p)
            alpha = rdotr / torch.dot(p, _Avp).to(self.device)
            x += alpha * p
            r -= alpha * _Avp
            new_rdotr = torch.dot(r, r).to(self.device)
            betta = new_rdotr / rdotr
            p = r + betta * p
            rdotr = new_rdotr
            if rdotr < residual_tol:
                break
        return x

    # TODO: 논문에서 다시 공부하기
    def _fisher_vector_product(self, s_batch, p):
        p.detach()
        kl = self._kl_divergence(new_actor=self.actor, old_actor=self.actor, s_batch=s_batch)
        kl = kl.mean()
        kl_grad = autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad = self._flat_grad(kl_grad)  # check kl_grad == 0

        kl_grad_p = (kl_grad * p).sum()
        kl_hessian_p = autograd.grad(kl_grad_p, self.actor.parameters())
        kl_hessian_p = self._flat_hessian(kl_hessian_p)

        return kl_hessian_p + 0.1 * p

    # TODO: 논문에서 다시 공부하기
    def _kl_divergence(self, new_actor, old_actor, s_batch):
        meow, std = new_actor(s_batch)
        # TODO: logstd 이렇게 만들면 그냥 0 아닌가? RLKR이 이렇게 함
        logstd = torch.zeros_like(meow).to(self.device)
        meow_old, std_old = old_actor(s_batch)
        # TODO: logstd 이렇게 만들면 그냥 0 아닌가? RLKR이 이렇게 함
        logstd_old = torch.zeros_like(meow_old).to(self.device)
        meow_old = meow_old.detach()
        std_old = std_old.detach()
        logstd_old = logstd_old.detach()

        # kl divergence between old policy and new policy : D( pi_old || pi_new )
        # pi_old -> mu0, logstd0, std0 / pi_new -> mu, logstd, std
        # be careful of calculating KL-divergence. It is not symmetric metric
        kl = logstd_old - logstd + (std_old.pow(2) + (meow_old - meow).pow(2)) / \
             (2.0 * std.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    def train_model(self):
        transitions = self.memory
        # random.shuffle(transitions)
        sar_batch = self.transition_structure(*zip(*transitions))
        s_batch = torch.stack(sar_batch.state).to(self.device)
        a_batch = torch.stack(sar_batch.action).to(self.device)
        r_batch = torch.stack(sar_batch.reward).to(self.device)
        done_batch = torch.stack(sar_batch.done).to(self.device)

        v_batch = self.critic(s_batch)

        # FIXME: 그냥 이 밑으론 일단 돌려 보고 다시 이해해야 될 듯... 하나도 모르겠음
        # FIXME: 그냥 이 밑으론 일단 돌려 보고 다시 이해해야 될 듯... 하나도 모르겠음
        # FIXME: 그냥 이 밑으론 일단 돌려 보고 다시 이해해야 될 듯... 하나도 모르겠음
        # FIXME: 그냥 이 밑으론 일단 돌려 보고 다시 이해해야 될 듯... 하나도 모르겠음
        # FIXME: 그냥 이 밑으론 일단 돌려 보고 다시 이해해야 될 듯... 하나도 모르겠음
        # 1단계: GAE로 return과 advantage 구하기
        # TODO: GAE 나중에 공부하자 일단은 갖다 씀
        return_batch, advantage_batch = self._gae(r_batch, done_batch, v_batch)

        # 2단계: Critic 최적화
        n = len(s_batch)
        arr = np.arange(n)

        for epoch in range(self.train_v_iters):
            np.random.shuffle(arr)
            for i in range(n // self.batch_size):
                batch_index = arr[self.batch_size * i: self.batch_size * (i + 1)]
                batch_index = u.t_long(batch_index)
                inputs = s_batch[batch_index]
                target1 = return_batch[batch_index]
                target2 = advantage_batch[batch_index]
                self.critic_optimizer.zero_grad()
                critic_loss = self.get_critic_loss(inputs, target1, target2)
                critic_loss.backward()
                self.critic_optimizer.step()

        # 3단계: loss의 그라디언트를 구하고 KL의 헤시안을 구하자
        meow, std = self.actor(s_batch)
        # TODO: logstd 이렇게 만들면 그냥 0 아닌가? RLKR이 이렇게 함
        logstd = torch.zeros_like(meow).to(self.device)
        old_policy = self._log_density(a_batch, meow, std, logstd)
        loss = self._surrogate_loss(advantage_batch, s_batch, old_policy.detach(), a_batch)
        # TODO: DDPG에서는 그냥 mean 때린거 바로 minimize 했는데 왜 여기선 autograd로?
        loss_grad = autograd.grad(loss, self.actor.parameters())
        loss_grad = self._flat_grad(loss_grad)
        step_dir = self._conjugate_gradient(s_batch, loss_grad.data, nsteps=self.cg_iters)
        loss = loss.detach().cpu().numpy()

        # 4단계: 스텝 방향, 스텝 크기를 구해서 스텝 벡터를 구하자 (방향 * 크기)
        actor_flat_params = self._flat_params(self.actor)
        shs = 0.5 * (step_dir * self._fisher_vector_product(s_batch, step_dir)).sum(0, keepdim=True)

        step_size = 1 / torch.sqrt(shs / self.max_kl).to(self.device)
        full_step = step_size * step_dir

        # 5단계: 백트래킹 line search를 n번 실행
        old_actor = Actor(self.state_size, self.action_size).to(self.device)
        self._update_model(old_actor, actor_flat_params)
        expected_improve = (loss_grad * full_step).sum(0, keepdim=True).to(self.device)
        expected_improve = expected_improve.cpu().numpy()

        line_search_succeed = False
        for i in range(self.backtrack_iters):
            backtrack_ratio = self.backtrack_coeff ** i
            constraint_params = actor_flat_params + backtrack_ratio * full_step
            self._update_model(self.actor, constraint_params)
            constraint_loss = self._surrogate_loss(advantage_batch, s_batch, old_policy.detach(), a_batch)
            constraint_loss = constraint_loss.detach().cpu().numpy()
            loss_improve = constraint_loss - loss
            expected_improve *= backtrack_ratio
            kl = self._kl_divergence(new_actor=self.actor, old_actor=old_actor, s_batch=s_batch)
            kl = kl.mean()

            # print('kl: {:.4f}  loss improve: {:.4f}  expected improve: {:.4f}  '
            #       'number of line search: {}'
            #       .format(kl.data.numpy(), loss_improve, expected_improve[0], i))
            TrainerMetadata().log(kl, 'KL', 'current_kl', compute_maxmin=True)
            TrainerMetadata().log(self.max_kl, 'KL', 'max_kl')
            TrainerMetadata().log(loss_improve / expected_improve, 'real / expected (improve)', 'real_ratio', compute_maxmin=True)
            TrainerMetadata().log(0.5, 'real / expected (improve)', 'threshold ')
            # TrainerMetadata().log(expected_improve, 'expected_improve', compute_maxmin=True)

            # see https://en.wikipedia.org/wiki/Backtracking_line_search
            # TODO: 0.5 인 이유?
            if kl < self.max_kl and (loss_improve / expected_improve) > 0.5:
                line_search_succeed = True
                break

        if not line_search_succeed:
            actor_flat_params = self._flat_params(old_actor)
            self._update_model(self.actor, actor_flat_params)
            print('policy update does not impove the surrogate')

