# -*- coding: utf-8 -*-
# DDPG
# Intrinsic Motivation based on Oudeyer et al. (2007)

import gym
import torch

import utils_kdm as u
from algorithm_im.im_fm import PredictiveFamiliarityMotivation
from algorithm_im.im_lpm import LearningProgressMotivation
from algorithm_im.im_nm import LearningNoveltyMotivation
from algorithm_im.im_random import RandomMotivation
from algorithm_im.im_sm import PredictiveSurpriseMotivation
from algorithm_rl.algo03_ddpg import DDPG, Transition
from algorithm_rl.algo04_trpo import TRPO
from utils_kdm.checkpoint import Checkpoint
from utils_kdm.drawer import Drawer
from utils_kdm.normalized_mujoco import NormalizedMujocoEnv
from utils_kdm.trainer_metadata import TrainerMetadata


# noinspection PyPep8Naming
class RLAgent(u.TorchSerializable):

    def __init__(self, algorithm_im, algorithm_rl, state_size, action_size, action_range, use_intrinsic=True):
        super().__init__()

        self._set_hyper_parameters()
        self.device = TrainerMetadata().device

        # 기본 설정
        self.state_size, self.action_size = state_size, action_size
        self.action_low, self.action_high = action_range

        self.algorithm_im, self.algorithm_rl = algorithm_im, algorithm_rl
        self.use_intrinsic = use_intrinsic
        if self.use_intrinsic is False:
            self.algorithm_im.intrinsic_reward_ratio = 0

        self.register_serializable([
            'algorithm_im',
            'algorithm_rl',
        ])

    def _set_hyper_parameters(self):
        pass

    def start_epoch(self):
        self.algorithm_rl.actor.eval()
        self.algorithm_rl.critic.eval()
        self.algorithm_rl.reset()

    def finish_epoch(self):
        self.algorithm_rl.actor.train()
        self.algorithm_rl.critic.train()
        self.train_model()

    def append_sample(self, sars, done):
        self.algorithm_rl.append_sample(sars, done)

    def get_action(self, state):
        return self.algorithm_rl.get_action(state)

    def get_weighted_reward(self, i_epoch, current_step, current_sars, current_done):
        current_state, current_action, ext_reward, next_state = current_sars
        TrainerMetadata().log(ext_reward, 'ext_reward', show_only_last=True, compute_maxmin=True)

        int_reward = 0
        if self.use_intrinsic:
            int_reward = self.algorithm_im.get_reward(i_epoch, current_step, current_sars, current_done)
            TrainerMetadata().log(int_reward, 'int_reward', show_only_last=True, compute_maxmin=True)

        if current_done:
            self.algorithm_im.scale_annealing()

        int_ext_reward, weighted_int, weighted_ext = self.algorithm_im.weighted_reward(int_reward, ext_reward)
        TrainerMetadata().log(int_ext_reward, 'int_ext_reward', show_only_last=True, compute_maxmin=True)

        return int_ext_reward

    def train_model(self):
        self.algorithm_rl.train_model()


if __name__ == "__main__":
    #####################
    # 환경 설정
    #####################

    # 0. 일반 설정
    # TODO: 시드 넣기
    # env.seed(args.seed)
    # torch.manual_seed(args.seed)
    FORCE_CPU = False
    TrainerMetadata().set_device(force_cpu=FORCE_CPU)

    # 1. 시각화 관련 설정
    VISDOM_RESET = True
    # VIZ_ENV_NAME = os.path.basename(os.path.realpath(__file__))
    VIZ_ENV_NAME = '32_'

    # 2. 저장 관련 설정
    VERSION = 1
    # TODO: 알고리즘 episode 당 인터벌 계산하는 거 제대로 처리 후 다시 저장 켜기
    IS_LOAD, IS_SAVE, SAVE_INTERVAL = False, False, 422
    SAVE_FULL_PATH = __file__

    # 3. 실험 환경 관련 설정
    GYM_ENV = 'HalfCheetah-v2'
    RENDER = False
    LOG_INTERVAL = 1
    EPOCHS = 100000
    MAX_EPISODES = 30000
    STEPS_PER_EPOCH = 4000  # From OpenAI

    # 4. 알고리즘 설정
    USE_INTRINSIC = False

    #####################
    # 객체 구성
    #####################
    viz = Drawer(reset=VISDOM_RESET, env=VIZ_ENV_NAME)
    checkpoint = Checkpoint(VERSION, IS_SAVE, SAVE_INTERVAL)

    # Agent 생성
    env = gym.make(GYM_ENV)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_range = (min(env.action_space.low), max(env.action_space.high))
    env = NormalizedMujocoEnv(env, state_size, clip=5)

    # Random = 보수 랜덤으로 (지정된 범위 내에서)
    # NM = 예측한 다음 상태와 실제 다음 상태의 오차가 클수록 보상 높음
    # LPM = 각 '지역'별로 나뉜 상태들이 일정 시간에 따라 오차가 줄어들면 보상 높음
    # SM = NM에서 쓰인 예측을 또 다시 예측하는 메타망을 사용해서,
    #      메타망은 오차 작은데 그냥 예측망이 오차 높으면 보상 높음
    # FM = 각 '지역'별로 오차가 작을수록 보상 높음
    # algorithm_im = RandomMotivation(state_size, action_size)
    # algorithm_im = LearningNoveltyMotivation(state_size, action_size)
    # algorithm_im = LearningProgressMotivation(state_size, action_size)
    # algorithm_im = PredictiveSurpriseMotivation(state_size, action_size)
    algorithm_im = PredictiveFamiliarityMotivation(state_size, action_size)

    algorithm_rl = TRPO(state_size, action_size)
    agent = RLAgent(algorithm_im, algorithm_rl,
                    state_size, action_size, action_range,
                    use_intrinsic=USE_INTRINSIC)

    # 메타데이터 관리 클래스 설정
    TrainerMetadata().reset(
        viz=viz,
        checkpoint=checkpoint,
        agent=agent,
        force_cpu=FORCE_CPU,
        log_interval=LOG_INTERVAL,
        save_full_path=SAVE_FULL_PATH,
        visdom_order=[
            'score',
            'critic_loss',
            'KL',
            'real / expected (improve)',
            'int_reward',
            'ext_reward',
            'int_ext_reward',
        ],
        console_log_order=[
            'Epoch',
            'Score',
            'KL_iter',
            'Time',
        ]
    )

    if IS_LOAD:
        TrainerMetadata().load()

    # TODO: i_epoch 변수 만들고 resume 가능하게
    for i_epoch in range(EPOCHS):
        TrainerMetadata().start_episode()
        agent.start_epoch()

        step_in_epoch = 0
        # 최대 에피소드 수만큼 돌린다
        for i_episode in range(0, MAX_EPISODES):
            state = env.reset()
            score = u.t_float32(0)

            # 각 에피소드당 환경에 정의된 최대 스텝 수만큼 돌린다
            # 단 그 전에 환경에서 정의된 종료 상태(done)가 나오면 거기서 끝낸다
            for i_step in range(env.spec.max_episode_steps):
                TrainerMetadata().start_step()

                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)

                sars = (state, action, reward, next_state)
                int_ext_reward = agent.get_weighted_reward(i_epoch, step_in_epoch, sars, done)

                sars = (state, action, int_ext_reward, next_state)
                agent.append_sample(sars, done)

                score += reward
                state = next_state

                env.render() if RENDER else None
                TrainerMetadata().finish_step()
                step_in_epoch += 1
                if done or step_in_epoch == STEPS_PER_EPOCH:
                    break

            TrainerMetadata().log(score, 'score', compute_maxmin=True)
            if step_in_epoch == STEPS_PER_EPOCH:
                break

        agent.finish_epoch()
        TrainerMetadata().finish_episode(i_epoch)

        if IS_SAVE:
            TrainerMetadata().save()
