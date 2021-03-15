import numpy as np


class Goal:
    def __init__(self, position, velocity=None, ndim=3):
        self._ndim = ndim if ndim else 3

        self._position = np.zeros(self._ndim)
        self.position = position

        self._velocity = np.zeros(self._ndim)
        if velocity is not None:
            self.velocity = velocity

    @property
    def ndim(self):
        return self._ndim

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, position):
        self._position[:] = position[:]

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, velocity):
        self._velocity[:] = velocity[:]

    def move(self, dt):
        self.position += self.velocity * dt
