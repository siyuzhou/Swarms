import numpy as np


class Obstacle:
    def __init__(self, position, velocity=None, ndim=None):
        """Base class `Obstacle`."""
        self._ndim = ndim if ndim else 3

        self._position = np.zeros(self._ndim)
        self.position = position
        self._velocity = np.zeros(self._ndim)
        if velocity:
            self.velocity = velocity
        self.size = 0

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

    def distance(self, r):
        raise NotImplementedError()

    def direction(self, r):
        """Direction of position `r` relative to obstacle surface"""
        raise NotImplementedError()


class Wall(Obstacle):
    def __init__(self, direction, position, velocity=None, ndim=None):
        """
        A plane in space that repels free agents.

        Parameters:
            position: the position of a point the wall passes.
            direction: the normal direction of the plane wall.
        """
        super().__init__(position, velocity, ndim)

        self._direction = np.array(direction, dtype=float)
        if self._direction.shape != (self._ndim,):
            raise ValueError('direction must be of shape ({},)'.format(self._ndim))
        self._direction /= np.linalg.norm(self._direction)  # Normalization

    def distance(self, r):
        return np.dot(r - self.position, self.direction(r))

    def direction(self, r):
        return self._direction


class Sphere(Obstacle):
    def __init__(self, size, position, velocity=None, ndim=None):
        """
        A sphere in ndim space.
        """
        super().__init__(position, velocity, ndim)
        self.size = size

    def distance(self, r):
        d = np.linalg.norm(r - self.position) - self.size
        if d < 0.1:
            d = 0.1
        return d

    def direction(self, r):
        """Direction of position `r` relative to obstacle surface"""
        d = r - self.position
        return d / np.linalg.norm(d)


class Rectangle(Obstacle):
    def __init__(self, sides, direction, position, velocity=None, ndim=None):
        """
        A generalized rectangle in ndim space.
        """
        super().__init__(position, velocity, ndim)

        if len(sides) != self.ndim:
            raise ValueError('number of side lengths does not match ndim')

        self._direction = np.array(direction, dtype=float)
        if self._direction.shape != (self._ndim,):
            raise ValueError('direction must be of shape ({},)'.format(self._ndim))
        self._direction /= np.linalg.norm(self._direction)  # Normalization
