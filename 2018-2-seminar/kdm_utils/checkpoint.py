# -*- coding: utf-8 -*-

import os
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

    def __init__(self, version, is_save=True, save_interval=10):
        self.version = version
        self.is_save = is_save
        self.save_interval = save_interval

    def is_saving_episode(self, current_epoch):
        return self.is_save and current_epoch % self.save_interval == 0

    def _split_path_base(self, full_path):
        dir_path = os.path.dirname(os.path.realpath(full_path))
        base_name = os.path.basename(os.path.realpath(full_path))
        return dir_path, base_name

    def get_best_model_file_name(self, full_path):
        dir_path, base_name = self._split_path_base(full_path)
        full_path = '{}/saved_model/{}/{}.best.pt'.format(dir_path, self.version, base_name)
        return full_path

    def save_checkpoint(self, full_path, var_state, is_best=False):
        dir_path, base_name = self._split_path_base(full_path)
        full_path = "{}/saved_model/{}/{}.ep{}.pt".format(dir_path, self.version, base_name,
                                                          str(var_state['current_epoch']))
        var_state['version'] = self.version
        torch.save(var_state, full_path)
        if is_best:
            shutil.copyfile(full_path, self.get_best_model_file_name(full_path))

    def load_model(self, full_path=None, device=None):
        device = device if device else get_device()
        return torch.load(full_path, map_location=device)
