# -*- coding: utf-8 -*-

# RLKR
# https://github.com/reinforcement-learning-kr/pg_travel/blob/master/mujoco/agent/trpo_gae.py


import torch
from utils_kdm.trainer_metadata import TrainerMetadata


# TODO: 논문에서 다시 공부하기
def kl_divergence(new_actor, old_actor, s_batch):
    device = TrainerMetadata().device
    meow, logstd, std = new_actor(s_batch)
    meow_old, logstd_old, std_old = old_actor(s_batch)
    meow_old = meow_old.detach()
    std_old = std_old.detach()
    logstd_old = logstd_old.detach()

    # kl divergence between old policy and new policy : D( pi_old || pi_new )
    # pi_old -> mu0, logstd0, std0 / pi_new -> mu, logstd, std
    # be careful of calculating KL-divergence. It is not symmetric metric
    kl = logstd_old - logstd + (std_old.pow(2) + (meow_old - meow).pow(2)) / \
         (2.0 * std.pow(2)) - 0.5

    return kl.sum(1, keepdim=True)
