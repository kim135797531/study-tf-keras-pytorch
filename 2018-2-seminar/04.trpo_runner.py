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
from utils_ext.running_state import ZFilter
from utils_kdm.checkpoint import Checkpoint
from utils_kdm.drawer import Drawer
from utils_kdm.trainer_metadata import TrainerMetadata


# noinspection PyPep8Naming
class RLAgent(u.TorchSerializable):

    def __init__(self, algorithm_im, algorithm_rl, state_size, action_size, action_range, use_intrinsic=True):
        super().__init__()

        self._set_hyper_parameters()
        self.device = TrainerMetadata().device

        # 기본 설정
        self.state_size, self.action_size = state_size, action_size
        # TODO: 정규화된 입력인지 검사 문구 넣고 range 빼기
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

    def get_action(self, state):
        return self.algorithm_rl.get_action(state)

    """
    def train_model(self, i_episode, current_step, current_sars, current_done):
        s, a, ext_reward, next_s = current_sars
        TrainerMetadata().log(ext_reward, 'ext_reward', show_only_last=True, compute_maxmin=True)

        int_reward = 0
        if self.use_intrinsic:
            int_reward = self.algorithm_im.get_reward(i_episode, current_step, current_sars, current_done)
            TrainerMetadata().log(int_reward, 'int_reward', show_only_last=True, compute_maxmin=True)

        if current_done:
            self.algorithm_im.scale_annealing()

        int_ext_reward, weighted_int, weighted_ext = self.algorithm_im.weighted_reward(int_reward, ext_reward)
        TrainerMetadata().log(int_ext_reward, 'int_ext_reward', show_only_last=True, compute_maxmin=True)

        current_sars = (s, a, int_ext_reward, next_s)
        self.algorithm_rl.train_model(current_sars, current_done)
    """
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
    VIZ_ENV_NAME = '999_'

    # 2. 저장 관련 설정
    VERSION = 1
    # TODO: 알고리즘 episode 당 인터벌 계산하는 거 제대로 처리 후 다시 저장 켜기
    IS_LOAD, IS_SAVE, SAVE_INTERVAL = False, False, 422
    SAVE_FULL_PATH = __file__

    # 3. 실험 환경 관련 설정
    GYM_ENV = 'Swimmer-v2'
    RENDER = True
    LOG_INTERVAL = 1
    EPISODES = 30000
    # STEPS_PER_EPOCH = 4000  # From OpenAI
    STEPS_PER_EPOCH = 4000

    # 4. 알고리즘 설정
    USE_INTRINSIC = True

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

    # Random = 보수 랜덤으로 (지정된 범위 내에서)
    # NM = 예측한 다음 상태와 실제 다음 상태의 오차가 클수록 보상 높음
    # LPM = 각 '지역'별로 나뉜 상태들이 일정 시간에 따라 오차가 줄어들면 보상 높음
    # SM = NM에서 쓰인 예측을 또 다시 예측하는 메타망을 사용해서,
    #      메타망은 오차 작은데 그냥 예측망이 오차 높으면 보상 높음
    # FM = 각 '지역'별로 오차가 작을수록 보상 높음
    algorithm_im = RandomMotivation(state_size, action_size)
    # algorithm_im = LearningNoveltyMotivation(state_size, action_size)
    # algorithm_im = LearningProgressMotivation(state_size, action_size)
    # algorithm_im = PredictiveSurpriseMotivation(state_size, action_size)
    # algorithm_im = PredictiveFamiliarityMotivation(state_size, action_size)

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
        save_full_path=SAVE_FULL_PATH
    )

    if IS_LOAD:
        TrainerMetadata().load()

    # TODO: 도대체 running_state란?
    running_state = ZFilter((state_size,), clip=5)

    # TODO: iter 변수 만들고 resume 가능하게
    for iter in range(100000):
        TrainerMetadata().start_episode()
        # TODO: iter부터 시간 측정?
        agent.algorithm_rl.actor.eval()
        agent.algorithm_rl.critic.eval()
        # TODO: TRPO에도 메모리가 필요하구나
        agent.algorithm_rl.memory_clear()

        # TODO: global_step 너무 더러운데..
        global_step = 0
        # 최대 에피소드 수만큼 돌린다
        for i_episode in range(0, EPISODES):

            state = env.reset()
            state = running_state(state)
            score = u.t_float32(0)

            # 각 에피소드당 환경에 정의된 최대 스텝 수만큼 돌린다
            # 단 그 전에 환경에서 정의된 종료 상태(done)가 나오면 거기서 끝낸다
            for t in range(env.spec.max_episode_steps):
                TrainerMetadata().start_step()

                action = agent.get_action(state)
                next_state, reward, done, _ = env.step(action)
                next_state = running_state(next_state)

                TrainerMetadata().log(reward, 'ext_reward', show_only_last=True, compute_maxmin=True)

                if USE_INTRINSIC:
                    int_reward = algorithm_im.get_reward(i_episode, t, (state, action, reward, next_state), done)
                    TrainerMetadata().log(int_reward, 'int_reward', show_only_last=True, compute_maxmin=True)
                else:
                    int_reward = 0

                if done:
                    algorithm_im.scale_annealing()

                int_ext_reward, weighted_int, weighted_ext = algorithm_im.weighted_reward(int_reward, reward)
                TrainerMetadata().log(int_ext_reward, 'int_ext_reward', show_only_last=True, compute_maxmin=True)

                sar = (state, action, int_ext_reward)
                agent.algorithm_rl.append_sample(sar, done)

                score += reward
                state = next_state

                env.render() if RENDER else None
                TrainerMetadata().finish_step()
                global_step += 1
                if done or global_step == STEPS_PER_EPOCH:
                    break

            # TODO: 매 에피소드당 스코어 말고 한 iter 당으로 평균 내기?
            TrainerMetadata().log(score, 'score', compute_maxmin=True)

            if global_step == STEPS_PER_EPOCH:
                break

        agent.algorithm_rl.actor.train()
        agent.algorithm_rl.critic.train()
        agent.train_model()

        TrainerMetadata().finish_episode(iter)

        if IS_SAVE:
            TrainerMetadata().save()
