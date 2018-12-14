# -*- coding: utf-8 -*-

import numpy as np
from visdom import Visdom

from utils_kdm.trainer_metadata import TrainerMetadata


class Drawer:

    def __init__(self, reset=False, env='main'):
        if reset:
            Visdom().delete_env(env=env)

        self.default_env = env
        self.default_interval = 1
        self.default_win = 'default'
        self.default_variable = 'default'
        self.viz = Visdom(env=env)

    def draw_line(self, y, x=None, interval=None, env=None, win=None, variable=None):
        x = x if x or x == 0 else TrainerMetadata().global_step
        interval = interval if interval else self.default_interval
        env = env if env else self.default_env
        win = win if win else self.default_win
        variable = variable if variable else self.default_variable

        if x % interval == 0:
            # Visdom은 numpy array를 입력으로 받음
            x = x if isinstance(x, (np.ndarray, np.generic)) else np.array([x])
            y = y if isinstance(y, (np.ndarray, np.generic)) else np.array([y])

            win = "{}...{}".format(env[:6], win)
            self.viz.line(X=np.array([x]), Y=np.array([y]), name=variable, win=win, update='append', opts={'title': win})
