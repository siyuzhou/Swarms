import numpy as np


class Goal:
    def __init__(self, position, priority=1, ndim=3):
        self._ndim = ndim if ndim else 3

        self._position = np.zeros(self._ndim)
        self.position = position

        self.priority = priority

    @property
    def ndim(self):
        return self._ndim

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position[:] = position[:]
