# -*- coding: utf-8 -*-
# Idea from https://stackoverflow.com/questions/6602256/python-wrap-all-functions-in-a-library

from gym.wrappers import TimeLimit
from utils_ext.running_state import ZFilter


# ZFilter를 사용하여 state를 정규화한다
# TRPO 논문 저자 레포에도 사용하고 있고,
# OpenAI Baseline의 경우 Mujoco 환경일 때 무조건 ZFilter를 쓰는 듯하다
# 실험해보니 안 써도 학습은 되지만 엄청 느려진다 (TRPO)
class NormalizedMujocoEnv(TimeLimit):

    def __init__(self, env, state_size, clip, max_episode_seconds=None, max_episode_steps=None):
        super().__init__(env, max_episode_seconds, max_episode_steps)
        self.running_state = ZFilter((state_size,), clip=clip)

    def reset(self):
        state = super().reset()
        return self.running_state(state)

    def step(self, action):
        next_state, reward, done, info = super().step(action)
        next_state = self.running_state(next_state)
        return next_state, reward, done, info
