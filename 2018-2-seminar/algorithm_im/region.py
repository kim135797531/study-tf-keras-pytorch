# -*- coding: utf-8 -*-
from collections import deque
from collections import namedtuple
from copy import deepcopy
from itertools import compress

import numpy as np
import torch
import torch.nn as nn
# noinspection PyPep8Naming
import torch.nn.functional as F
import torch.optim as optim

import utils_kdm as u
from utils_kdm import ManageDevice
from utils_kdm.trainer_metadata import TrainerMetadata


ExemplarStructure = namedtuple('ExemplarStructure', ('state', 'action', 'next_state'))


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
        # x = s_a
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        return self.head(x)


class Region(u.TorchSerializable):

    # TODO: 리전별로 객체화 하면 엄청 느릴 것 같은데..
    # TODO: Torch stack으로?
    def __init__(self, state_size, action_size):
        super().__init__()

        self._set_hyper_parameters()
        self.device = TrainerMetadata().device
        self.state_size = state_size
        self.action_size = action_size

        # 전체 차원은 현재 상태 차원 크기 + 액션 차원 크기 + 다음 상태 차원 크기
        self.global_max_dim = self.state_size + self.action_size + self.state_size

        self._is_leaf = True
        self.cutting_dim = None
        self.cutting_val = None
        self.left_child = None
        self.right_child = None

        self.exemplars = list()

        self.transposed_exemplars = list()
        for i in range(self.global_max_dim):
            self.transposed_exemplars.append(list())

        self.expert = Expert(self.state_size, self.action_size).to(self.device)
        self.expert_optimizer = optim.Adam(
            self.expert.parameters(),
            lr=self.learning_rate_expert
        )
        self.loss_queue = deque(maxlen=self.past_time + self.time_window)

        self.register_serializable([
            'self.expert',
            'self.expert_optimizer',
            'self._is_leaf',
            'self.exemplars',
            'self.transposed_exemplars',
            'self.loss_queue',
        ])

    def _set_hyper_parameters(self):
        # TODO: 저장, 로드
        # Expert망 Adam 학습률
        self.learning_rate_expert = 0.01
        # 아래와 같이 하면 (t-40 ~ t-15) 와 (t-25 ~ t) 사이의 에러를 비교하게 된다
        # 논문에서 theta, 얼마나 전의 기록이랑 비교할 것인가
        self.past_time = 15
        # 논문에서 tau, 얼마만큼의 오차 평균을 내서 비교할 것인가
        self.time_window = 25

    def state_dict(self):
        print('save')
        ret = super().state_dict()
        return ret

    def load_state_dict(self, var_state):
        if not var_state['_is_leaf']:
            self.register_serializable([
                'self.cutting_dim',
                'self.cutting_val',
                'self.left_child',
                'self.right_child',
            ])
            self.unregister_serializable([
                'self.expert',
                'self.expert_optimizer',
                'self.exemplars',
                'self.transposed_exemplars',
                'self.loss_queue',
            ])

        super().load_state_dict(var_state)

    def is_leaf(self):
        return self._is_leaf

    def set_as_non_leaf(self, cutting_dim, cutting_val, left_child, right_child):
        self._is_leaf = False
        self.cutting_dim = cutting_dim
        self.cutting_val = cutting_val
        self.left_child = left_child
        self.right_child = right_child
        del self.exemplars
        del self.transposed_exemplars
        del self.expert
        del self.expert_optimizer
        del self.loss_queue
        self.register_serializable([
            'self.cutting_dim',
            'self.cutting_val',
            'self.left_child',
            'self.right_child',
        ])
        self.unregister_serializable([
            'self.expert',
            'self.expert_optimizer',
            'self.exemplars',
            'self.transposed_exemplars',
            'self.loss_queue',
        ])

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
        first_dim, second_dim = -1, -1

        if global_dim < self.state_size:
            first_dim, second_dim = 0, global_dim
        elif self.state_size <= global_dim < self.state_size + self.action_size:
            first_dim, second_dim = 1, global_dim - self.state_size
        elif self.state_size + self.action_size <= global_dim < self.global_max_dim:
            first_dim, second_dim = 2, global_dim - self.state_size - self.action_size

        if first_dim == -1 or second_dim == -1:
            LookupError("region dim 범위 벗어남")

        return first_dim, second_dim

    def _train_model(self, exemplar):
        if isinstance(exemplar, list):
            exemplars = ExemplarStructure(*zip(*exemplar))
            s = torch.stack(exemplars.state).to(self.device)
            a = torch.stack(exemplars.action).to(self.device)
            next_s = torch.stack(exemplars.next_state).to(self.device)
        else:
            s = exemplar.state.unsqueeze(dim=0)
            a = exemplar.action.unsqueeze(dim=0)
            next_s = exemplar.next_state.unsqueeze(dim=0)

        predicted_next_s = self.expert(s, a)

        # 상태 예측기 최적화
        # TODO: 샘플이 최대 250개라면 뉴럴넷보단 SVM이나 베이지안이 낫지 않을까
        self.expert_optimizer.zero_grad()
        state_predictor_loss = nn.MSELoss().to(self.device)  # 배치니까 mean 해줘야 할 듯?
        state_predictor_loss = state_predictor_loss(predicted_next_s, next_s)
        state_predictor_loss.backward()
        self.expert_optimizer.step()

        self.loss_queue.append(state_predictor_loss.item())

    def _add_to_transposed_exemplars(self, exemplar):
        # TODO: 제발 속도 개선 하면서 속도를 더 느려지게 하지 말자 ㅠ
        for i in range(self.global_max_dim):
            first_dim, second_dim = self.global_dim_to_local_dim(i)
            self.transposed_exemplars[i].append(exemplar[first_dim][second_dim])

    def add(self, exemplar):
        self.exemplars.append(exemplar)
        self._add_to_transposed_exemplars(exemplar)
        self._train_model(exemplar)

    def add_all(self, exemplars):
        self.exemplars.extend(exemplars)
        for exemplar in exemplars:
            self._add_to_transposed_exemplars(exemplar)
        self._train_model(exemplars)

    def get_past_error_mean(self):
        if len(self.loss_queue) < self.loss_queue.maxlen:
            # 아직 충분한 샘플이 모이지 않았을 경우
            return 1

        past_error_sum = 0
        for i in range(self.past_time, self.past_time + self.time_window):
            past_error_sum += self.loss_queue[-i]

        return past_error_sum / self.time_window

    def get_current_error_mean(self):
        if len(self.loss_queue) < self.loss_queue.maxlen:
            # 아직 충분한 샘플이 모이지 않았을 경우
            return 1

        current_error_sum = 0
        for i in range(0, self.time_window):
            current_error_sum += self.loss_queue[-i]

        return current_error_sum / self.time_window


class RegionManager(u.TorchSerializable):

    def __init__(self, state_size, action_size):
        super().__init__()

        self._set_hyper_parameters()
        self.device = TrainerMetadata().device
        self.region_head = Region(state_size, action_size)

        self.register_serializable([
            'self.region_head',
        ])

    def _set_hyper_parameters(self):
        self.region_maxlen = 250
        # 분산 계산할 때 각 리전에 최소 2개 이상씩은 있어야 함
        assert(self.region_maxlen >= 4)

    def add(self, exemplar):
        region = self.find_region(exemplar)
        region.add(exemplar)

        if self.met_criterion_1(region):
            self.split_region(region)

    def met_criterion_1(self, region):
        return len(region.exemplars) > self.region_maxlen

    def split_region(self, region):
        min_weighted_var, min_cutting_dim, min_left_indices, min_right_indices = None, None, None, None
        n_dim = region.state_size + region.action_size  # SM(t)

        next_state_batch = ExemplarStructure(*zip(*region.exemplars)).next_state
        next_state_tensor = torch.stack(next_state_batch).to(self.device)

        for dim in range(n_dim):
            # TODO: 여러 dim에 대해 동시에 돌리기
            weighted_var, left_indices, right_indices = self.find_minimum_variance(region, next_state_tensor, dim)

            if min_weighted_var is None or min_weighted_var > weighted_var:
                min_weighted_var, min_cutting_dim, min_left_indices, min_right_indices = \
                    weighted_var, dim, left_indices, right_indices

        min_left_child = Region(region.state_size, region.action_size)
        min_left_child.add_all(list(compress(region.exemplars, min_left_indices)))

        min_right_child = Region(region.state_size, region.action_size)
        min_right_child.add_all(list(compress(region.exemplars, min_right_indices)))

        first_dim, second_dim = region.global_dim_to_local_dim(min_cutting_dim)
        min_cutting_val = min_left_child.exemplars[-1][first_dim][second_dim]

        region.set_as_non_leaf(min_cutting_dim, min_cutting_val, min_left_child, min_right_child)

    def find_minimum_variance(self, region, next_state_tensor, dim):
        min_weighted_var, min_left_indices, min_right_indices = None, None, None

        # region.exemplars 의 현재 차원에 대해 정렬 순서의 index를 가져오기 [3, 20, 13, 1, 4, ... ] = 무조건 251개
        current_dim_vals = region.transposed_exemplars[dim]
        current_dim_vals = torch.stack(current_dim_vals)
        sorted_dim_vals, indices = torch.sort(current_dim_vals)

        # 인덱스 배열에서 현재 나누는 기준까지의 index만 가져오기 [1, 0, 0, 1, 1, ... ] = 무조건 251개
        for cutting_index in range(2, len(indices) - 1):
            left_indices = (indices < cutting_index)
            right_indices = left_indices == 0
            # 해당 인덱스를 갖는 자료들끼리 복사없이 비교?
            weighted_var = cutting_index * next_state_tensor[left_indices].var() + \
                           (len(indices) - cutting_index) * next_state_tensor[right_indices].var()

            if min_weighted_var is None or min_weighted_var > weighted_var:
                min_weighted_var = weighted_var
                min_left_indices = left_indices
                min_right_indices = right_indices

        return min_weighted_var, min_left_indices, min_right_indices

    def find_region(self, sars, region=None):
        # TODO: 재귀에서 루프로 바꾸기 (트리가 엄청 깊음)
        current = region if region else self.region_head
        # s = []
        done = False

        while not done:
            if current.is_leaf():
                break

            first_dim, second_dim = current.global_dim_to_local_dim(current.cutting_dim)
            if current.cutting_val < sars[first_dim][second_dim]:
                current = current.left_child
            else:
                current = current.right_child

        return current

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
        indices_X = region.exemplars.transpose(dim0=0, dim=1)[dim]

        n_clusters = 2

        samples = ExemplarStructure(*zip(*region.exemplars))

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
            # data1 = region.exemplars
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
    sample_1 = ExemplarStructure(state, action, next_state)

    state = np.float32([21, 22, 23, 24, 25, 26, 27, 28])
    action = np.float32([29, 30])
    next_state = np.float32([31, 32, 33, 34, 35, 36, 37, 38])
    state = u.t_from_np_to_float32(state)
    action = u.t_from_np_to_float32(action)
    next_state = u.t_from_np_to_float32(next_state)
    sample_2 = ExemplarStructure(state, action, next_state)

    for i in range(100):
        region_manager.add(deepcopy(sample_1))
        region_manager.add(deepcopy(sample_2))

    region2 = region_manager.find_region(sample_2)
    print(region2.expert(torch.unsqueeze(state, dim=0), torch.unsqueeze(action, dim=0)))
