# -*- coding: utf-8 -*-

import numpy as np
from visdom import Visdom


class Drawer:

    def __init__(self, reset=False, env='main'):
        if reset:
            Visdom().delete_env(env=env)

        self.default_step = 0
        self.default_env = env
        self.default_interval = 1
        self.default_name = 'default'
        self.viz = Visdom(env=env)

    def step(self):
        self.default_step += 1

    def draw_line(self, y, x=None, interval=None, env=None, name=None):
        x = x if x or x == 0 else self.default_step
        interval = interval if interval else self.default_interval
        env = env if env else self.default_env
        name = name if name else self.default_name

        if x % interval == 0:
            # Visdom은 numpy array를 입력으로 받음
            x = x if isinstance(x, (np.ndarray, np.generic)) else np.array([x])
            y = y if isinstance(y, (np.ndarray, np.generic)) else np.array([y])

            name = "{}_{}".format(env, name)
            self.viz.line(X=np.array([x]), Y=np.array([y]), name=name, win=name, update='append', opts={'title': name})
