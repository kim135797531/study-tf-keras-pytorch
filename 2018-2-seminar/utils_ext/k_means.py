# K-means implementation by PyTorch
# Made by https://github.com/overshiki/kmeans_pytorch
# Merged kmeans.py, pairwise.py in original repo

import torch
import numpy as np
from utils_kdm import global_device


'''
calculation of pairwise distance, and return condensed result, i.e. we omit the diagonal and duplicate entries and store everything in a one-dimensional array
'''


def _pairwise_distance(data1, data2=None, device=None):
    r"""
    using broadcast mechanism to calculate pairwise ecludian distance of data
    the input data is N*M matrix, where M is the dimension
    we first expand the N*M matrix into N*1*M matrix A and 1*N*M matrix B
    then a simple elementwise operation of A and B will handle the pairwise operation of points represented by data
    """
    device = device if device else global_device
    if data2 is None:
        data2 = data1

    data1, data2 = data1.to(device), data2.to(device)

    # N*1*M
    A = data1.unsqueeze(dim=1)

    # 1*N*M
    B = data2.unsqueeze(dim=0)

    dis = (A - B) ** 2.0
    # return N*N matrix for pairwise distance
    dis = dis.sum(dim=-1).squeeze()
    return dis


def _group_pairwise(X, groups, device=None, fun=lambda r, c: _pairwise_distance(r, c).cpu()):
    """
    X의 그룹 단위로 거리 계산
    ex) X = {
        x: [1, 2, 3, ...],
        y: [1, 2, 3, ...],
        z: [1, 2, 3, ...]
    }
    x와 x의 거리, x와 y의 거리, x와 z의 거리,
    y와 x의 거리, y와 y의 거리, y와 z의 거리,
    z와 x의 거리, z와 y의 거리, z와 z의 거리를 반환
    :param X:
    :param groups:
    :param device:
    :param fun:
    :return:
    """
    device = device if device else global_device
    group_dict = {}
    for group_index_r, group_r in enumerate(groups):
        for group_index_c, group_c in enumerate(groups):
            R, C = X[group_r].to(device), X[group_c].to(device)
            group_dict[(group_index_r, group_index_c)] = fun(R, C)
    return group_dict


def get_initial_state(X, n_clusters):
    """
    X에서 n개 샘플 추출(해서 중심 좌표로 삼기)

        Args:
          X: n차원 Numpy 배열 (float 가정)
          n_clusters: 클러스터 갯수
        Returns:
          initial_state: 초기 좌표
    """
    _len = len(X)
    indices = np.random.choice(_len, n_clusters)
    initial_state = X[indices]
    return initial_state


def execute(X, n_clusters, device=None, tol=1e-4):
    """lloyd algorithm

        Args:
          X: n차원 Numpy 배열 (float 가정)
          n_clusters: 클러스터 갯수
          device: PyTorch device 오브젝트
          tol: 계산 도중 중심 이동 간격 최소 기대치
        Returns:
          choice_cluster: X가 속한 클러스터 인덱스 (0~n-1)
          initial_state: 초기 좌표
    """
    device = device if device else global_device
    X = torch.from_numpy(X).float().to(device)

    initial_state = get_initial_state(X, n_clusters)

    while True:
        # Expectation
        dis = _pairwise_distance(X, initial_state)

        choice_cluster = torch.argmin(dis, dim=1)

        initial_state_pre = initial_state.clone()

        for index in range(n_clusters):
            selected = torch.nonzero(choice_cluster == index).squeeze()

            selected = torch.index_select(X, 0, selected)
            # Maximization
            initial_state[index] = selected.mean(dim=0)

        center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))

        if center_shift ** 2 < tol:
            break

    return choice_cluster, initial_state


if __name__ == "__main__":
    # Example code
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
    """
    X = [
            [x1, y1],
            [x2, y2],
            ...
        ]
    """
    X = [np.random.randn(1000, 3), np.random.randn(1000, 3) + 5, np.random.randn(1000, 3) + 10]
    X = np.concatenate(X, axis=0)
    choice_cluster, initial_state = execute(X, 3)

    # fig, ax = plt.subplots(projection='3d')
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for i in range(3):
        indices = np.where(choice_cluster == i)[0]
        selected = X[indices]
        ax.plot(selected[:, 0], selected[:, 1], selected[:, 2], '.', label=str(i))

    fig.show()



