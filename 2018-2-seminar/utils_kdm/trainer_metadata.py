# -*- coding: utf-8 -*-

import time
from collections import defaultdict

import torch

from utils_kdm import TorchSerializable
from utils_kdm.manage_device import ManageDevice
from utils_kdm.singleton import Singleton


def _make_indicator_defaultdict():
    return defaultdict(_make_indicator_inner_defaultdict)


def _make_indicator_inner_defaultdict():
    return defaultdict(list)


# noinspection PyMethodParameters
class TrainerMetadata(TorchSerializable, Singleton):

    def __init__(cls):
        super().__init__()

        # 내부 사용 인스턴스
        cls.viz = None
        cls.checkpoint = None
        cls.agent = None

        # 내부 사용 변수
        cls.current_epoch = None
        cls.global_step = None

        cls.indicators = None
        cls._volatile_indicators = None
        cls.best_score = None

        cls.start_time = 0

        # 환경 설정
        cls.log_interval = None
        cls.save_full_path = None

    @property
    def device(cls):
        return ManageDevice().get(call_from='TrainerMetadata')

    def reset(cls,
              viz,
              checkpoint,
              agent,
              force_cpu=False,
              log_interval=1,
              save_full_path=__file__):
        cls.viz = viz
        cls.checkpoint = checkpoint
        cls.agent = agent

        cls.current_epoch = 0
        cls.global_step = 0

        # Indicators는 화면에 표시도 하고 저장/불러오기 할 지표들
        cls.indicators = _make_indicator_defaultdict()
        # Volatile Indicators는 일단 전부 다 저장하는 곳
        # 한 에피소드가 끝나면 indicators로 자료들 옮기고 비우기
        cls._volatile_indicators = _make_indicator_defaultdict()
        cls.best_score = 0

        cls.start_time = 0
        cls.log_interval = log_interval
        cls.save_full_path = save_full_path
        ManageDevice().set(force_cpu, call_from='TrainerMetadata')

    def set_device(cls, force_cpu=False):
        ManageDevice().set(force_cpu, call_from='TrainerMetadata')

    def state_dict_impl(cls):
        return {
            'current_epoch': cls.current_epoch + 1,
            'global_step': cls.global_step,
            'indicators': cls.indicators,
            'best_score': cls.best_score,
            'agent': cls.agent.state_dict()
        }

    def load_state_dict_impl(cls, var_state):
        cls.current_epoch = var_state['current_epoch']
        cls.global_step = var_state['global_step']
        cls.indicators = var_state['indicators']
        cls.best_score = var_state['best_score']
        cls.agent.load_state_dict(var_state['agent'])

    def log(cls, value=0, indicator='default_win', variable='default_var', interval=1, save_only_last=True):
        # TODO: 저장할 때 value의 타입이 Tensor면 .item() 해서 저장하는게 빠르려나?
        if cls.global_step % interval == 0:
            if save_only_last:
                # 일단 들어오는 정보 전부 모으는 곳에 저장
                # 나중에 finish_episode 에서 맨 마지막 정보만 indicators 딕셔너리로 옮기기
                cls._volatile_indicators[indicator][variable].append(value)
            else:
                cls.indicators[indicator][variable].append(value)

    def save(cls):
        # state_dict 구성 속도가 느리므로 필요할 때만 구성
        if cls.checkpoint.is_saving_episode(cls.current_epoch):
            var_state = cls.state_dict()
            is_best = False
            if 'score' in cls.indicators:
                score = cls.indicators['score']['default_var']
                max_score = max(score)
                if max_score > cls.best_score:
                    cls.best_score = max_score
                    is_best = True

            cls.checkpoint.save_checkpoint(cls.save_full_path, var_state, is_best)

    def load(cls):
        full_path = cls.checkpoint.get_best_model_file_name(__file__)
        print("Loading checkpoint '{}'".format(full_path))
        var_state = cls.checkpoint.load_model(full_path=full_path)
        cls.load_state_dict(var_state)

        for indicator_name, variables in cls.indicators.items():
            for variable_name, variable_sequence in variables.items():
                for i_episode in range(0, len(variable_sequence)):
                    cls.viz.draw_line(x=i_episode, y=variable_sequence[i_episode],
                                      win=indicator_name, variable=variable_name)

        print("Loading complete. Resuming from episode: {}".format(cls.current_epoch - 1))
        if 'score' in cls.indicators:
            score = cls.indicators['score']['default_var'][-1]
            print("Score: {:.2f}".format(max(score, default=0)))

    def start_episode(cls):
        cls.start_time = time.time()

    def start_step(cls):
        cls.global_step += 1

    def finish_step(cls):
        pass

    def finish_episode(cls, i_episode):
        cls.current_epoch = i_episode

        for indicator_name, variables in cls._volatile_indicators.items():
            for variable_name, variable_sequence in variables.items():
                cls.indicators[indicator_name][variable_name].append(variable_sequence[-1])
        cls._volatile_indicators.clear()

        if i_episode % cls.log_interval == 0:
            # TODO: score는 그냥 전부 있다고 가정하고 변수화?
            last_score = cls.indicators['score']['default_var'][-1]
            last_score = last_score.item() if isinstance(last_score, torch.Tensor) else last_score
            if len(cls.indicators['memory_len']['default_var']) > 0:
                memory_len = len(cls.agent.memory)
            else:
                memory_len = 0
            print('Episode {}\tScore: {:.2f}\tMem Length: {}\tCompute Time: {:.2f}'.format(
                i_episode, last_score, memory_len, time.time() - cls.start_time))

            for indicator_name, variables in cls.indicators.items():
                if 'memory' in indicator_name:
                    continue
                for variable_name, variable_sequence in variables.items():
                    if len(variable_sequence) > 0:
                        y = variable_sequence[-1]
                        y = y.item() if isinstance(y, torch.Tensor) else y
                        cls.viz.draw_line(x=i_episode, y=y, win=indicator_name, variable=variable_name)
