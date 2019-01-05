# -*- coding: utf-8 -*-

import torch

from utils_kdm.trainer_metadata import TrainerMetadata


class GAE:
    # Generalized Advantage Estimation (Schulman et al. 2016)

    def __init__(self, gamma=0.99):
        self.device = TrainerMetadata().device
        self.gamma = gamma

    def _flip_0_1(self, batch):
        batch = batch + 1
        batch[batch > 1] = 0
        return batch

    def get_return_advantage(self, r_batch, not_done_batch, v_batch):
        return_batch = torch.zeros_like(r_batch).to(self.device)
        advantage_batch = torch.zeros_like(r_batch).to(self.device)

        running_return = 0
        previous_v = 0
        running_advantage = 0

        # 에피소드가 끝난 (목적 달성한 순간 or 시간 초과) 상태는 계산에 넣지 않는다
        # 끝난 후의 보상은 없으므로
        not_done_batch = self._flip_0_1(not_done_batch)

        for t in reversed(range(0, len(r_batch))):
            running_return = r_batch[t] + self.gamma * running_return * not_done_batch[t]
            running_tderror = r_batch[t] + self.gamma * previous_v * not_done_batch[t] - \
                              v_batch.data[t]
            running_advantage = running_tderror + self.gamma * self.gamma * \
                                running_advantage * not_done_batch[t]

            return_batch[t] = running_return
            previous_v = v_batch.data[t]
            advantage_batch[t] = running_advantage

        advantage_batch = (advantage_batch - advantage_batch.mean()) / advantage_batch.std()

        return return_batch, advantage_batch
