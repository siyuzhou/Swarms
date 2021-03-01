import numpy as np
from .entity import Entity


class Obstacle(Entity):
    def __init__(self, position, velocity=None, ndim=None):
        """Base class `Obstacle`."""
        super().__init__(position, velocity, ndim=ndim)

        self.size = 0

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
    def __init__(self, sides, position, orientation, velocity=None, ndim=None):
        """
        A generalized rectangle in ndim space.
        """
        super().__init__(position, velocity, ndim)

        if len(sides) != self.ndim:
            raise ValueError('number of side lengths does not match ndim')

        self._orientation = np.array(orientation, dtype=float)
        if self._orientation.shape != (self._ndim,):
            raise ValueError('direction must be of shape ({},)'.format(self._ndim))
        self._orientation /= np.linalg.norm(self._orientation)  # Normalization
