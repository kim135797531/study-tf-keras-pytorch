# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
import shutil
import torch

from kdm_utils import get_device


class TorchSerializable(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def state_dict_impl(self):
        raise NotImplementedError("Please implement this method.")

    def state_dict(self):
        ret = self.state_dict_impl()
        return ret

    @abstractmethod
    def load_state_dict_impl(self, state_dict):
        raise NotImplementedError("Please implement this method.")

    def load_state_dict(self, state_dict):
        ret = self.load_state_dict_impl(state_dict)
        return ret


class Checkpoint:

    def __init__(self, is_save=True, save_interval=10):
        # TODO: 버전 관리
        self.version = 2
        self.is_save = is_save
        self.save_interval = save_interval
        # TODO: 파일 이름 관리 ㅠㅠ
        self.default_file_name = self._get_current_file_name()

    def _get_current_file_name(self):
        return __file__ if '__file__' in vars() or '__file__' in globals() else 'undefined_name'

    def is_saving_episode(self, current_epoch):
        return self.is_save and current_epoch % self.save_interval == 0

    def get_best_model_file_name(self, file_name=None):
        file_name = self.default_file_name if not file_name else file_name
        return file_name + '.best.pt'

    def save_checkpoint(self, var_state, is_best=False, file_name=None):
        file_name = self.default_file_name if not file_name else file_name
        full_name = "{}.ep{}.pt".format(file_name, str(var_state['current_epoch']))
        var_state['version'] = self.version
        torch.save(var_state, full_name)
        if is_best:
            shutil.copyfile(full_name, self.get_best_model_file_name(file_name))

    def load_model(self, file_name=None, device=None):
        device = device if device else get_device()
        return torch.load(self.get_best_model_file_name(file_name), map_location=device)
        # return torch.load(self.get_best_model_file_name(file_name))

