# -*- coding: utf-8 -*-

import random
from collections import namedtuple

from utils_kdm import TorchSerializable


class ReplayMemory(TorchSerializable):
    # TODO: 주석 달기
    def __init__(self, capacity, structure=None):
        super().__init__()

        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.structure = structure if structure else self._default_structure()

        self.register_serializable([
            'capacity',
            'memory',
            'position'
        ])

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
