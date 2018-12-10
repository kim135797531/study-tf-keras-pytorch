# -*- coding: utf-8 -*-

import random
from collections import namedtuple

from utils_kdm.checkpoint import TorchSerializable


class ReplayMemory(TorchSerializable):
    # TODO: 주석 달기
    def __init__(self, capacity, structure=None):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.structure = structure if structure else self._default_structure()

    def _default_structure(self):
        return namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = self.structure(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, min(len(self.memory), batch_size))

    def __len__(self):
        return len(self.memory)

    def state_dict_impl(self):
        return {
            'capacity': self.capacity,
            'memory': self.memory,
            'position': self.position
        }

    def load_state_dict_impl(self, state_dict):
        self.capacity = state_dict['capacity']
        self.memory = state_dict['memory']
        self.position = state_dict['position']
