# -*- coding: utf-8 -*-
from collections import deque
from collections import namedtuple
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
import torch.optim as optim

import utils_kdm as u
from utils_kdm.trainer_metadata import TrainerMetadata


class Expert(nn.Module):

    def __init__(self, state_size, action_size):
        super(Expert, self).__init__()
        # TODO: 상태 예측은 망 별로 안 커도 학습될 듯? (상태 예측만 테스트 해 보기)
        # 내발적 동기를 위해서 상태를 예측한다는 개념 = 2007년 Oudeyer 논문을 참조한 것
        sensorimotor_size = state_size + action_size
        self.layer_sizes = [sensorimotor_size, 32, 16, state_size]

        self.linear1 = nn.Linear(self.layer_sizes[0], self.layer_sizes[1])
        self.linear2 = nn.Linear(self.layer_sizes[1], self.layer_sizes[2])
        self.head = nn.Linear(self.layer_sizes[2], self.layer_sizes[3])
        u.fanin_init(self.linear1.weight)
        u.fanin_init(self.linear2.weight)
        nn.init.uniform_(self.head.weight, a=-3*10e-4, b=3*10e-4)

    def forward(self, state, action):
        # 그냥 일렬로 합쳐기
        # Oudeyer (2007)
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.head(x)


class Region(u.TorchSerializable):

    # TODO: 리전별로 객체화 하면 엄청 느릴 것 같은데..
    # TODO: Torch stack으로?
    def __init__(self, state_size, action_size):
        self._set_hyper_parameters()
        self.device = TrainerMetadata().device
        self.state_size = state_size
        self.action_size = action_size

        self._is_leaf = True
        self.cutting_dim = None
        self.cutting_val = None
        self.left_child = None
        self.right_child = None

        self.samples = list()
        self.expert = Expert(self.state_size, self.action_size).to(self.device)
        self.expert_optimizer = optim.Adam(
            self.expert.parameters(),
            lr=self.learning_rate_expert
        )
        self.loss_queue = deque(maxlen=self.past_time + self.time_window)

        self.exemplar_structure = namedtuple('Exemplar', ('state', 'action', 'next_state'))

    def _set_hyper_parameters(self):
        # TODO: 저장, 로드
        # Expert망 Adam 학습률
        self.learning_rate_expert = 0.001
        # 아래와 같이 하면 (t-40 ~ t-15) 와 (t-25 ~ t) 사이의 에러를 비교하게 된다
        # 논문에서 theta, 얼마나 전의 기록이랑 비교할 것인가
        self.past_time = 15
        # 논문에서 tau, 얼마만큼의 오차 평균을 내서 비교할 것인가
        self.time_window = 25

    def state_dict_impl(self):
        # TODO: 저장, 로드
        return {
            'expert': self.expert.state_dict(),
            'expert_optimizer': self.expert_optimizer.state_dict()
        }

    def load_state_dict_impl(self, var_state):
        self.expert.load_state_dict(var_state['expert'])
        self.expert_optimizer.load_state_dict(var_state['expert_optimizer'])
        # noinspection PyAttributeOutsideInit
        self.intrinsic_scale_1 = var_state['intrinsic_scale_1']

    def is_leaf(self):
        return self._is_leaf

    def set_as_non_leaf(self, cutting_dim, cutting_val, left_child, right_child):
        self._is_leaf = False
        self.cutting_dim = cutting_dim
        self.cutting_val = cutting_val
        self.left_child = left_child
        self.right_child = right_child
        # TODO: 삭제하면 자식 노드에 영향 X?
        del self.samples

    def global_dim_to_local_dim(self, global_dim):
        # ex) state가 8개, action이 2개면
        # 전체 차원 범위 = 0~9
        # 전체 차원 0 => 로컬 차원 0, 0
        # 전체 차원 1 => 로컬 차원 0, 1
        # 전체 차원 2 => 로컬 차원 0, 2
        # ...
        # 전체 차원 7 => 로컬 차원 0, 7
        # 전체 차원 8 => 로컬 차원 1, 0
        # 전체 차원 9 => 로컬 차원 1, 1
        first = 0
        if global_dim >= self.state_size:
            first += 1
            global_dim -= self.state_size
        second = global_dim
        return first, second

    def _train_model(self):
        samples = self.exemplar_structure(*zip(*self.samples))
        s = torch.cat(samples.state).to(self.device)
        a = torch.cat(samples.action).to(self.device)
        next_s = torch.cat(samples.next_state).to(self.device)

        predicted_next_s = self.expert(s, a)

        # TODO: 상태 예측기도 DDPG 처럼 타겟망까지 만들어서 예측? 아니면 단순한 순차 선형 신경망?
        # state_prediction_error = nn.L1Loss(reduction='none').to(self.device)
        # state_prediction_error = state_prediction_error(predicted_next_s.detach(), next_s.detach())
        # state_prediction_error = torch.sum(state_prediction_error, dim=1).to(self.device)

        # 상태 예측기 최적화
        self.expert_optimizer.zero_grad()
        state_predictor_loss = nn.MSELoss().to(self.device)  # 배치니까 mean 해줘야 할 듯?
        state_predictor_loss = state_predictor_loss(predicted_next_s, next_s)
        state_predictor_loss.backward()
        self.expert_optimizer.step()

        self.loss_queue.append(state_predictor_loss.item())

    def add(self, sample):
        self.samples.append(sample)

        # TODO:
        # if len(self.samples) > 1:
        self._train_model()


class RegionManager(u.TorchSerializable):

    def __init__(self, state_size, action_size):
        self._set_hyper_parameters()
        self.device = TrainerMetadata().device
        self.region_head = Region(state_size, action_size)
        self.exemplar_structure = namedtuple('Exemplar', ('state', 'action', 'next_state'))

    def _set_hyper_parameters(self):
        # 논문의 C1 상수
        self.region_maxlen = 250
        # 분산 계산할 때 각 리전에 최소 2개 이상씩은 있어야 함
        assert(self.region_maxlen >= 4)

    def state_dict_impl(self):
        # TODO: 저장, 로드
        pass

    def load_state_dict_impl(self, state_dict):
        # TODO: 저장, 로드
        pass

    def add(self, sample):
        region = self.find_region(sample)
        region.add(sample)

        if self.met_criterion_1(region):
            self.split_region(region)

    def met_criterion_1(self, region):
        return True if len(region.samples) > self.region_maxlen else False

    def split_region(self, region):
        min_weighted_var, min_index, min_cutting_dim = None, None, None

        n_dim = region.state_size + region.action_size  # SM(t)
        for dim in range(n_dim):
            # FIXME: 여기 100% 동작 안 할 듯
            # TODO: 여러 dim에 대해 동시에 돌리기
            # weighted_var, index = self.find_minimum_variance(region, dim)
            weighted_var = 0
            index = 5
            if min_weighted_var is None or min_weighted_var > weighted_var:
                min_weighted_var = weighted_var
                min_index = index
                min_cutting_dim = dim

        first_dim, second_dim = region.global_dim_to_local_dim(min_cutting_dim)
        # TODO: 나중에 find_minimum_variance에서 Region 만들어서 넘겨주면 삭제
        region.samples.sort(key=lambda elem: elem[first_dim][0][second_dim])
        min_left_child = Region(region.state_size, region.action_size)
        min_left_child.samples = region.samples[:min_index]
        min_left_child._train_model()
        min_right_child = Region(region.state_size, region.action_size)
        min_right_child.samples = region.samples[min_index:]
        min_right_child._train_model()

        min_cutting_val = min_left_child.samples[-1][first_dim][0][second_dim]

        region.set_as_non_leaf(min_cutting_dim, min_cutting_val, min_left_child, min_right_child)

    def find_minimum_variance(self, region, dim):
        first_dim, second_dim = region.global_dim_to_local_dim(dim)
        # TODO: 이 구문 때문에 반드시 tuple이 state, action, next_state 순으로 저장되어 있어야 함
        # TODO: 중간 dimension 하드코딩.. [first_dim]!!![0]!!![second_dim]
        region.samples.sort(key=lambda elem: elem[first_dim][0][second_dim])

        samples = self.exemplar_structure(*zip(*region.samples))
        next_s = samples.next_state

        min_weighted_var, min_index = None, None
        # 위에서 정렬했기 때문에 오름차순이 보장되어 있음
        # TODO: 분산이 n > 2 에서만 동작하는데 2개 이상부터 잘라도 괜찮나?
        for i in range(2, len(next_s) - 1):
            left_next_s, right_next_s = torch.stack(next_s[:i]), torch.stack(next_s[i:])
            # TODO: 분산 계산.. 맞는지 모르겠다
            weighted_var = \
                len(left_next_s) * left_next_s.var() + \
                len(right_next_s) * right_next_s.var()

            if min_weighted_var is None or min_weighted_var > weighted_var:
                min_weighted_var = weighted_var
                min_index = i

        # TODO: 현재는 디버깅 쉽게 하기 위해 인덱스만 반환해서 나중에 정렬 한 번 더 하지만
        # TODO: 추후 Region 객체를 만들어서 반환해야 함
        return min_weighted_var, min_index

    def find_region(self, sample, region=None):
        region = region if region else self.region_head
        if region.is_leaf():
            return region

        first_dim, second_dim = region.global_dim_to_local_dim(region.cutting_dim)
        if region.cutting_val < sample[first_dim][0][second_dim]:
            return self.find_region(sample, region.left_child)
        else:
            return self.find_region(sample, region.right_child)

    """
    def em_algorithm(self, region, dim):

        2차원 좌표 집합을 세로로 자를때 (특정 SM(t)[dim]로 자를때)
        잘린 각 클러스터의 σ(S(t+1))의 합이 최소가 되게하는 SM(t)[dim]의 분기값 찾기

        cluster by SM(t)[dim]
        evaluate by σ(S(t+1))
        E: 두 개의 SM(t)[dim]만을 기준으로 가까운 거로 클러스터링 (세로로 자르기)
        M: 각 클러스터 내부의 σ(S(t+1))가 젤 작은 SM(t)[dim] 각각 선택

        :param X:
        :return:

         내 아까운 코드들 ㅠ
        indices_X = region.samples.transpose(dim0=0, dim=1)[dim]

        n_clusters = 2

        samples = self.exemplar_structure(*zip(*region.samples))

        # TODO: 이거 튜플로 묶으면 다시 GPU에서 CPU로 오나?
        # 텐서의 집합에서 고차원 텐서로
        # tuple(tensor, ...) -> tensor()
        s = torch.cat(samples.state).to(self.device)
        a = torch.cat(samples.action).to(self.device)
        s_a = torch.cat((s, a)).to(self.device)
        next_s = torch.cat(samples.next_state).to(self.device)

        # 인덱스 랜덤으로 뽑기
        # SM(t)에서 특정 차원에 대해 중심을 구하므로 a=1
        indices = np.random.choice(a=1, size=n_clusters)

        # 인덱스로 좌표 만들 샘플 가져옴
        current_coord = s_a[indices]  # SM(t), S(t+1)
        # 샘플에서 특정 차원만 꺼내기
        current_coord_X = current_coord.transpose(dim0=0, dim1=1)[dim]

        while True:
            # E: 세로로 자르기 = 그냥 X랑 가까운 거
            # data1 = region.samples
            # data2 = current_coord
            # A = 250, 1, 1 (개수, 1개니깐, SM(t)[dim]=1)
            # B = 1, 2, 1 (개수, 클러스터 2개, SM(t)[dim]=1)
            A = s.transpose(dim0=0, dim1=1)[dim].unsqueeze(dim=1)
            B = current_coord_X.unsqueeze(dim=0)

            dis = (A - B) ** 2.0
            dis = dis.sum(dim=-1).squeeze()

            choice_cluster = torch.argmin(dis, dim=1)
            next_coord_X = current_coord_X.clone()

            for index in range(n_clusters):
                selected = torch.nonzero(choice_cluster == index).squeeze()
                selected_Y = torch.index_select(next_s, dim=0, index=selected)
                next_coord_X =
                next_coord_X[index] =

    """


if __name__ == "__main__":
    region_manager = RegionManager(8, 2)

    state = np.float32([999, 2, 3, 4, 5, 6, 7, 8])
    action = np.float32([9, 10])
    next_state = np.float32([11, 12, 13, 14, 15, 16, 17, 18])
    state = u.t_from_np_to_float32(state)
    action = u.t_from_np_to_float32(action)
    next_state = u.t_from_np_to_float32(next_state)
    sample_1 = region_manager.exemplar_structure(state, action, next_state)

    state = np.float32([21, 22, 23, 24, 25, 26, 27, 28])
    action = np.float32([29, 30])
    next_state = np.float32([31, 32, 33, 34, 35, 36, 37, 38])
    state = u.t_from_np_to_float32(state)
    action = u.t_from_np_to_float32(action)
    next_state = u.t_from_np_to_float32(next_state)
    sample_2 = region_manager.exemplar_structure(state, action, next_state)

    for i in range(100):
        region_manager.add(deepcopy(sample_1))
        region_manager.add(deepcopy(sample_2))

    region = region_manager.find_region(sample_2)
    print(region.expert(torch.unsqueeze(state, dim=0), torch.unsqueeze(action, dim=0)))
