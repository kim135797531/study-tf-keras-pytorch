# -*- coding: utf-8 -*-
# TRPO
# Trust Region Policy Optimization (Schulman et al. 2015)
#
# 참조: https://github.com/reinforcement-learning-kr/pg_travel/blob/master/mujoco/agent/trpo_gae.py
#

import copy
import math
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters
import torch.autograd as autograd
import torch.optim as optim

import utils_kdm as u
from utils_ext.kl_divergence import kl_divergence
from utils_ext.conjugate_gradient import conjugate_gradient
from utils_kdm.trainer_metadata import TrainerMetadata


# Python Pickle은 nested namedtuple save를 지원하지 않음
# https://stackoverflow.com/questions/4677012/python-cant-pickle-type-x-attribute-lookup-failed

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'done'))


# -> 그냥 '적절한' 것을 쓰라고 함
# 논문 저자:
# 입력: 상태
# 출력: 값 = 가우시안 분포를 이루게 하고 평균 뽑기
#           => 상관계수 행렬이 대각행렬이고 상태와 독립
# 신경망이 로그정규분포를 갖게
# => 따라서 정책이 정규분포(평균=신경망, 표준편차=exp(로그표준분포))를 갖게
#
# Actor: 입력->64->64->출력
# Critic: 입력->64->64->1
# Critic Optimizer = L-BFGS (이차 미분)
# => Learning Rate = 1 (왜냐면 이건 제대로 된 최적기라..)
# => L2 decay = 0.001 * (신경망 내부 변수 제곱을 전부 합한 것)
#
# reinforcement-learning-kr:
#     lambda = 0.98
#
# OpenAI:
# lam=0.97, -> GAE
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
        # TODO: (GAE 모름) 액션 크기는 안 받는 이유는? Q(s, a) 아닌가?
        self.layer_sizes = [state_size, 64, 64, 1]

        self.linear1 = nn.Linear(self.layer_sizes[0], self.layer_sizes[1])
        self.linear2 = nn.Linear(self.layer_sizes[1], self.layer_sizes[2])
        self.head = nn.Linear(self.layer_sizes[2], self.layer_sizes[3])

        # TODO: 초기화 이해 안 감 -> 공식 레포가 이렇지만 다른 초기화는?
        self.head.weight.data.mul_(0.1)
        self.head.bias.data.mul_(0.0)

    def forward(self, x):
        # TODO: 저자 공식 레포가 tanh지만 다른 거 써볼까
        # x = F.relu(self.linear1(x))
        # x = F.relu(self.linear2(x))
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

        # Optimizer
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
            'self.critic_optimizer',
        ])

    def _set_hyper_parameters(self):
        # TODO: (개선) 잘 정하기 공식 레포엔 없나??
        # Adam 하이퍼 파라미터
        # self.learning_rate_critic = 0.0003
        self.learning_rate_critic = 0.001
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
        self.backtrack_iters = 20
        self.backtrack_coeff = 0.8
        # Conjugate gradient 구할 때 헤시안 행렬 추정하는데,
        # 이 때 피셔-벡터곱만 하면 불안정해서 결과값에다가 원본에 0.1 곱한거 더해 줌 (단위행렬)
        self.damping_coeff = 0.1

        # From RLKR
        # KL 다이버전스 한계값 (신뢰 영역 범위)
        self.max_kl = 0.01
        # value 함수 학습을 같은 데이터에 대해 몇 번 할 것인가
        self.train_v_iters = 10

    def reset(self):
        pass

    def memory_clear(self):
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

    def _flip_0_1(self, batch):
        batch = batch + 1
        batch[batch > 1] = 0
        return batch

    def _gae(self, r_batch, done_batch, v_batch):
        return_batch = torch.zeros_like(r_batch).to(self.device)
        advantage_batch = torch.zeros_like(r_batch).to(self.device)

        running_return = 0
        previous_v = 0
        running_advantage = 0

        # FIXME: (GAE 모름) 이거 왜 거꾸로?
        done_batch = self._flip_0_1(done_batch)

        for t in reversed(range(0, len(r_batch))):
            running_return = r_batch[t] + self.gamma * running_return * done_batch[t]
            running_tderror = r_batch[t] + self.gamma * previous_v * done_batch[t] - \
                              v_batch.data[t]
            running_advantage = running_tderror + self.gamma * self.gamma * \
                              running_advantage * done_batch[t]

            return_batch[t] = running_return
            previous_v = v_batch.data[t]
            advantage_batch[t] = running_advantage

        advantage_batch = (advantage_batch - advantage_batch.mean()) / advantage_batch.std()
        return return_batch, advantage_batch

    def get_critic_loss(self, s_batch, return_batch, advantage_batch):
        critic_loss = nn.MSELoss().to(self.device)
        v_batch = self.critic(s_batch)
        critic_loss = critic_loss(v_batch, return_batch + advantage_batch)

        return critic_loss

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

    def _surrogate_loss(self, old_policy, new_policy, advantage_batch):
        # TODO: new_policy detach 해도 되나?
        surrogate = advantage_batch * torch.exp(new_policy - old_policy.detach())
        return surrogate.mean().to(self.device)

    def _fisher_vector_product(self, vector_p_with_state_batch):
        """
        b=Hx 에서 x를 구하려면 H의 역행렬을 알아야 한다.
        근데 H의 역행렬 구하기 어렵다.
        그래서 피셔 벡터곱으로 Hx를 대충 추정해서 넘겨주자.

        D_KL
        ∇D_KL
        (∇D_KL)^T * x
        ∇((∇D_KL)^T * x)

        Hx = ∇((∇D_KL(새로운θ|옛날θ))^T * x)
        """
        (p, s_batch) = vector_p_with_state_batch
        p.detach()
        # 왜 같은게 들어가냐면 현재 policy에 대한 다이버전스를 구하는 거라서
        kl = kl_divergence(new_actor=self.actor, old_actor=self.actor, s_batch=s_batch)
        kl = kl.mean()
        kl_grad = autograd.grad(kl, self.actor.parameters(), create_graph=True)
        kl_grad = parameters_to_vector(kl_grad)  # check kl_grad == 0

        kl_grad_p = (kl_grad * p).sum()
        kl_hessian_p = autograd.grad(kl_grad_p, self.actor.parameters())
        kl_hessian_p = parameters_to_vector(kl_hessian_p)

        return kl_hessian_p + self.damping_coeff * p

    def _line_search(self, old_loss, loss_grad, step_vector_x, advantage_batch, s_batch, old_policy, a_batch):
        old_actor = copy.deepcopy(self.actor)

        actor_flat_params = parameters_to_vector(self.actor.parameters())
        expected_improve = (loss_grad * step_vector_x).sum(0, keepdim=True)
        expected_improve = expected_improve.cpu().numpy()

        i, line_search_succeed = -1, False
        for i in range(self.backtrack_iters):
            # 라인 서치로 정책 업데이트
            backtrack_ratio = self.backtrack_coeff ** i
            constraint_params = actor_flat_params + backtrack_ratio * step_vector_x
            vector_to_parameters(constraint_params, self.actor.parameters())

            # 바꾼 actor를 기반으로 다시 평균(log정책(a|s)*A) 구해봄
            meow, logstd, std = self.actor(s_batch)
            new_policy = self._log_density(a_batch, meow, std, logstd)
            constraint_loss = self._surrogate_loss(
                old_policy=old_policy,
                new_policy=new_policy,
                advantage_batch=advantage_batch
            )
            loss_improve = (constraint_loss - old_loss).detach().cpu().numpy()
            weighted_expected_improve = backtrack_ratio * expected_improve
            kl = kl_divergence(new_actor=self.actor, old_actor=old_actor, s_batch=s_batch)
            kl = kl.mean()

            TrainerMetadata().log(kl, 'KL', 'current_kl', compute_maxmin=True)
            TrainerMetadata().log(self.max_kl, 'KL', 'max_kl')
            TrainerMetadata().log(loss_improve / weighted_expected_improve, 'real / expected (improve)', 'real_ratio', compute_maxmin=True)
            TrainerMetadata().log(0.5, 'real / expected (improve)', 'threshold ')
            # TrainerMetadata().log(expected_improve, 'expected_improve', compute_maxmin=True)

            # see https://en.wikipedia.org/wiki/Backtracking_line_search
            # TODO: 0.5 인 이유? 1.0 보다 커야 개선된 것 아닌가
            # 일단 Armijo used ​1⁄2 for both c and tau in a paper he published in 1966
            if kl < self.max_kl and (loss_improve / weighted_expected_improve) > 0.5:
                line_search_succeed = True
                break

        TrainerMetadata().console_log('KL_iter', i)

        if not line_search_succeed:
            self.actor = copy.deepcopy(old_actor)
            print('policy update does not impove the surrogate')

    def train_model(self):
        # 알고리즘 줄 번호는 OpenAI 기준
        # 줄 1~3 = 초기화
        # 줄 4 = 현재 정책 π로 trajectory 모으기
        transitions = self.memory
        # random.shuffle(transitions)
        sar_batch = self.transition_structure(*zip(*transitions))
        s_batch = torch.stack(sar_batch.state).to(self.device)
        a_batch = torch.stack(sar_batch.action).to(self.device)
        r_batch = torch.stack(sar_batch.reward).to(self.device)
        done_batch = torch.stack(sar_batch.done).to(self.device)

        # 줄 5 = rewards-to-go (R) 구하기
        # 줄 6 = 현재 가치 함수 (V)를 기반으로 추정 advantage (A) 구하기
        # TODO: (GAE 모름) 나중에 공부하자 일단은 갖다 씀
        v_batch = self.critic(s_batch)
        return_batch, advantage_batch = self._gae(r_batch, done_batch, v_batch)

        # 줄 7 = 정책 그라디언트 구하기
        # 그라디언트 = '각 정책에 대한' 평균(∇log정책(a|s)*A)
        # 정책(a|s) ~~> 현재 정책에서 해당 행동을 할 확률
        # log정책(a|s)
        # 평균(log정책(a|s)*A)
        # g = '각 정책에 대한' 평균(∇log정책(a|s)*A)
        meow, logstd, std = self.actor(s_batch)
        old_policy = self._log_density(a_batch, meow, std, logstd)

        # 아래 3줄이 처음에 이해가 안 갔다
        #
        # - 왜 바로 minimize() 호출 안 하고 굳이 autograd 패키지를 써서 gradient를 구하는가?
        # -> DDPG에서는 그냥 그라디언트 최소화만 하면 됐지만, 여기서는 먼저 그라디언트 변수를 구하고,
        #    그걸 이용해서 다시 conjugate gradient algorithm 으로 차례차례 나아간다.
        #    왜냐면 애초에 이 논문의 목적이 그냥 파라미터 공간에서 그라디언트 때려버리는 게 아니라
        #    그라디언트 때린 거가 정책 공간에서 얼마나 변했는지 검사 후 적용하려고 하는 것이기 때문
        #
        # - surrogate_loss (대리 loss) 란 무엇이고,
        #   왜 바로 아래 줄에서는 의미 없이 같은 policy 2개 사이의 차이를 계산에서 advantage에 곱하는가?
        # -> 원래 우리가 구하고자 하는 방향은 advantage를 최대화 하는 방향 (그라디언트 구하기)
        #    따라서 그냥 advantage 만으로 그라디언트 구해서 최대화 하면 됨
        #    그런데 저 밑에 코드에서 line search 하면서 조금 방향을 바꿔보려고 함
        #    그리고 조금 방향 바뀌었을 때와 원래 방향일 때 최종 loss가 어떻게 변하는지 보려고 함
        #    따라서 원래의 방향을 미리 알아둬야 하고, 이를 위해 미리 구해두는 것
        loss = self._surrogate_loss(
            old_policy=old_policy,
            new_policy=old_policy,
            advantage_batch=advantage_batch
        )
        loss_grad = autograd.grad(loss, self.actor.parameters())
        loss_grad = parameters_to_vector(loss_grad)

        # 줄 8 = 켤레 기울기법 적용해서 x 추정하기
        # x = H의 역행렬 * 그라디언트
        # 결론으로 구한 x는 우리가 어디로 가야 할 지 알려주는 방향 = step_direction_x
        step_direction_x = conjugate_gradient(self._fisher_vector_product, s_batch, loss_grad.data, cg_iters=self.cg_iters)

        # 줄 9 = 백트래킹 방법으로 정책 업데이트하기
        # 새로운 파라미터 = 파라미터 + sqrt(2*최대 kl 크기 제한 / H의 이차형식) * x
        # xhx = (x^-1)(Hx)
        # 크기: sqrt(2*최대 kl 크기 제한 / (x^-1)(Hx))
        # 방향벡터: sqrt(2*최대 kl 크기 제한 / (x^-1)(Hx)) * x
        xhx = (step_direction_x * self._fisher_vector_product((step_direction_x, s_batch))).sum(0, keepdim=True)
        step_size_x = torch.sqrt((2 * self.max_kl) / xhx).to(self.device)
        step_vector_x = step_size_x * step_direction_x
        self._line_search(loss, loss_grad, step_vector_x, advantage_batch, s_batch, old_policy, a_batch)

        # 줄 10 = 가치 함수 MSE로 경사 하강법 최적화
        n = len(s_batch)
        arr = np.arange(n)
        for epoch in range(self.train_v_iters):
            np.random.shuffle(arr)
            critic_loss = 0

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

            TrainerMetadata().log(critic_loss, 'critic_loss', show_only_last=True, compute_maxmin=True)
