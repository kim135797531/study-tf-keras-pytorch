# -*- coding: utf-8 -*-
# DDPG
# Intrinsic Motivation based on Oudeyer et al. (2007)

import gym
import torch

import utils_kdm as u
from algorithm_im.im_lpm import LearningProgressMotivation
from algorithm_im.im_nm import NoveltyMotivation
from algorithm_im.im_sm import PredictiveSurpriseMotivation
from algorithm_rl.algo03_ddpg import DDPG, Transition
from utils_kdm.checkpoint import Checkpoint
from utils_kdm.drawer import Drawer
from utils_kdm.trainer_metadata import TrainerMetadata


# noinspection PyPep8Naming
class RLAgent(u.TorchSerializable):

    def __init__(self, algorithm_im, algorithm_rl, state_size, action_size, action_range, use_intrinsic=True):
        self._set_hyper_parameters()
        self.device = TrainerMetadata().device

        # 기본 설정
        self.state_size, self.action_size = state_size, action_size
        # TODO: 정규화된 입력인지 검사 문구 넣고 range 빼기
        self.action_low, self.action_high = action_range

        self.algorithm_im, self.algorithm_rl = algorithm_im, algorithm_rl
        self.use_intrinsic = use_intrinsic

    def _set_hyper_parameters(self):
        pass

    def state_dict_impl(self):
        return {
            'algorithm_rl': self.algorithm_rl.state_dict(),
            'algorithm_im': self.algorithm_im.state_dict()
        }

    def load_state_dict_impl(self, var_state):
        self.algorithm_rl.load_state_dict(var_state['algorithm_rl'])
        self.algorithm_im.load_state_dict(var_state['algorithm_im'])

    def get_action(self, state):
        return self.algorithm_rl.get_action(state)

    def train_model(self, i_episode, current_step, current_sars, current_done):
        s, a, ext_reward, next_s = current_sars
        TrainerMetadata().log(ext_reward, 'ext_reward', show_only_last=True, compute_maxmin=True)

        # TODO: IM 오래 걸리니 일정 단위마다
        INTRINSIC_APPLY_INTERVAL = 1

        int_reward = 0
        if self.use_intrinsic and current_step % INTRINSIC_APPLY_INTERVAL == 0:
            int_reward = self.algorithm_im.get_reward(i_episode, current_step, current_sars, current_done)
            TrainerMetadata().log(int_reward, 'int_reward', show_only_last=True, compute_maxmin=True)

        if current_done:
            self.algorithm_im.scale_annealing()

        int_ext_reward, weighted_int, weighted_ext = self.algorithm_im.weighted_reward(int_reward, ext_reward)
        TrainerMetadata().log(int_ext_reward, 'int_ext_reward', show_only_last=True, compute_maxmin=True)

        current_sars = (s, a, int_ext_reward, next_s)
        self.algorithm_rl.train_model(current_sars, current_done)


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
    VIZ_ENV_NAME = '17_im_(LPM_0.9)(1step)(region250)(annealing99.9%,min0.001)'

    # 2. 저장 관련 설정
    VERSION = 1
    IS_LOAD, IS_SAVE, SAVE_INTERVAL = False, True, 401
    SAVE_FULL_PATH = __file__

    # 3. 실험 환경 관련 설정
    GYM_ENV = 'Swimmer-v2'
    RENDER = True
    LOG_INTERVAL = 1
    EPISODES = 30000

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

    # algorithm_im = NoveltyMotivation(state_size, action_size)
    algorithm_im = LearningProgressMotivation(state_size, action_size)
    # algorithm_im = PredictiveSurpriseMotivation(state_size, action_size)

    algorithm_rl = DDPG(state_size, action_size, action_range)
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

            sars = (state, action, reward, next_state)
            agent.algorithm_rl.append_sample(sars, done)

            if len(agent.algorithm_rl.memory) >= agent.algorithm_rl.train_start:
                agent.train_model(i_episode, t, sars, done)

            score += reward
            state = next_state

            env.render() if RENDER else None
            TrainerMetadata().finish_step()
            if done:
                break

        TrainerMetadata().log(score, 'score')
        TrainerMetadata().log(len(agent.algorithm_rl.memory), 'memory_len')
        TrainerMetadata().finish_episode(i_episode)

        if IS_SAVE:
            TrainerMetadata().save()

        # TODO: 일정 간격마다 노이즈 없이 테스트?
        # if score > env.spec.reward_threshold:
        #     print("Solved! Running reward is now {}".format(score))
        #    break
