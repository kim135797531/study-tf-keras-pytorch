# -*- coding: utf-8 -*-
# A2C

import sys

import gym
import numpy as np
import torch

import utils_kdm as u
from algorithm_im.im_sm import PredictiveSurpriseMotivation
from algorithm_rl.algo01_a2c import A2C
from algorithm_rl.algo03_ddpg import Transition
from utils_kdm.checkpoint import Checkpoint
from utils_kdm.drawer import Drawer
from utils_kdm.replay_memory import ReplayMemory
from utils_kdm.trainer_metadata import TrainerMetadata


# noinspection PyPep8Naming
class RLAgent(u.TorchSerializable):

    def __init__(self, algorithm_im, algorithm_rl, state_size, action_size, action_range):
        self._set_hyper_parameters()
        self.device = TrainerMetadata().device

        # 기본 설정
        self.state_size, self.action_size = state_size, action_size
        # TODO: 정규화된 입력인지 검사 문구 넣고 range 빼기
        self.action_low, self.action_high = action_range

        self.algorithm_im, self.algorithm_rl = algorithm_im, algorithm_rl
        # TODO: 리플레이 메모리를 DDPG 알고리즘에서 분리해서 저장하는 게 아름다운가?
        # 리플레이 메모리
        self.transition_structure = Transition
        self.memory = ReplayMemory(self.memory_maxlen, self.transition_structure)

    def _set_hyper_parameters(self):
        # 리플레이 메모리 관련
        self.batch_size = 1
        self.memory_maxlen = 1

    def state_dict_impl(self):
        return {
            'algorithm_rl': self.algorithm_rl.state_dict(),
            'algorithm_im': self.algorithm_im.state_dict(),
            'memory': self.memory.state_dict()
        }

    def load_state_dict_impl(self, var_state):
        self.algorithm_rl.load_state_dict(var_state['algorithm_rl'])
        self.algorithm_im.load_state_dict(var_state['algorithm_im'])
        self.memory.load_state_dict(var_state['memory'])

    def get_action(self, state):
        return self.algorithm_rl.get_action(state)

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.push(
            u.t_float32(state),
            u.t_float32(action),
            u.t_float32(reward),
            u.t_float32(next_state),
            u.t_uint8(done)
        )

    def train_model(self, i_episode, step, done):
        # 메모리에서 일정 크기만큼 기억을 불러온다
        # 그 후 기억을 모아 각 변수별로 모은다. (즉, 전치행렬)
        # TODO: random과 zip func 글카에서 하기
        transitions = self.memory.sample(self.batch_size)
        # SARS = State, Action, Reward, next State
        sars_batch = self.transition_structure(*zip(*transitions))

        # TODO: 이거 튜플로 묶으면 다시 GPU에서 CPU로 오나?
        # 텐서의 집합에서 고차원 텐서로
        # tuple(tensor, ...) -> tensor()
        s = torch.cat(sars_batch.state).to(self.device)
        a = torch.cat(sars_batch.action).to(self.device)
        ext_r = torch.cat(sars_batch.reward).to(self.device)
        next_s = torch.cat(sars_batch.next_state).to(self.device)

        USE_INTRINSIC = False

        if USE_INTRINSIC:
            int_r = self.algorithm_im.get_reward(i_episode, step, transitions, s, a, next_s)
            if done:
                TrainerMetadata().log(torch.max(int_r), 'int_reward', 'max')
                TrainerMetadata().log(torch.mean(int_r), 'int_reward', 'mean')
                TrainerMetadata().log(torch.min(int_r), 'int_reward', 'min')
            ext_r = self.algorithm_im.weighted_reward(int_r, ext_r)

        if done:
            TrainerMetadata().log(torch.max(ext_r), 'ext_reward', 'max')
            TrainerMetadata().log(torch.mean(ext_r), 'ext_reward', 'mean')
            TrainerMetadata().log(torch.min(ext_r), 'ext_reward', 'min')

        self.algorithm_rl.train_model(s, a, ext_r, next_s, done)


if __name__ == "__main__":
    #####################
    # 환경 설정
    #####################

    # 0. 일반 설정
    FORCE_CPU = False
    TrainerMetadata().set_device(force_cpu=FORCE_CPU)

    # 1. 시각화 관련 설정
    VISDOM_RESET = True
    # VIZ_ENV_NAME = os.path.basename(os.path.realpath(__file__))
    VIZ_ENV_NAME = '07.a2c_runner'

    # 2. 저장 관련 설정
    VERSION = 1
    IS_LOAD, IS_SAVE, SAVE_INTERVAL = False, True, 400
    SAVE_FULL_PATH = __file__

    # 3. 실험 환경 관련 설정
    GYM_ENV = 'CartPole-v1'
    RENDER = False
    LOG_INTERVAL = 1
    EPISODES = 30000

    #####################
    # 객체 구성
    #####################
    viz = Drawer(reset=VISDOM_RESET, env=VIZ_ENV_NAME)
    checkpoint = Checkpoint(VERSION, IS_SAVE, SAVE_INTERVAL)

    # Agent 생성
    env = gym.make(GYM_ENV)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    action_range = (-1, 1)

    algorithm_im = PredictiveSurpriseMotivation(state_size, action_size)
    algorithm_rl = A2C(state_size, action_size)
    agent = RLAgent(algorithm_im, algorithm_rl, state_size, action_size, action_range)

    # 메타데이터 관리 클래스 설정
    TrainerMetadata().reset(
        viz=viz,
        checkpoint=checkpoint,
        agent=agent,
        force_cpu=FORCE_CPU,
        log_interval=LOG_INTERVAL,
        save_full_path=SAVE_FULL_PATH
    )

    if IS_LOAD:
        TrainerMetadata().load()

    # 최대 에피소드 수만큼 돌린다
    for i_episode in range(TrainerMetadata().current_epoch, EPISODES):
        TrainerMetadata().start_episode()

        agent.algorithm_rl.reset()
        state = env.reset()
        score = u.t_float32(0)

        # 각 에피소드당 환경에 정의된 최대 스텝 수만큼 돌린다
        # 단 그 전에 환경에서 정의된 종료 상태(done)가 나오면 거기서 끝낸다
        for t in range(env.spec.max_episode_steps):
            TrainerMetadata().start_step()

            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)

            reward = reward if not done or score == 499 else -100

            agent.append_sample(state, action, reward, next_state, done)
            agent.train_model(i_episode, t, done)

            score += reward
            state = next_state

            env.render() if RENDER else None
            TrainerMetadata().finish_step()
            if done:
                break

        TrainerMetadata().log(score + 100, 'score')
        TrainerMetadata().finish_episode(i_episode)

        if IS_SAVE:
            TrainerMetadata().save()

        scores = TrainerMetadata().indicators['score']['default_var']
        if np.mean(scores[-10:]) > 490:
            sys.exit()
