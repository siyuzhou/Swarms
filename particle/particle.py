import numpy as np


class Particle:
    def __init__(self, position, velocity, max_v=None, max_a=None):
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)

        self.position[:] = position[:]
        self.velocity[:] = velocity[:]

        self.max_v = float(max_v) if max_v else None
        self.max_a = float(max_a) if max_a else None

        self._regularize_v()
        self._regularize_a()

    @property
    def speed(self):
        return np.linalg.norm(self.velocity)

    def _regularize_v(self):
        if self.max_v and self.speed > self.max_v:
            self.velocity *= self.max_v / self.speed

    def _regularize_a(self):
        if self.max_a and np.linalg.norm(self.acceleration) > self.max_a:
            self.acceleration *= self.max_a / np.linalg.norm(self.acceleration)

    def move(self, dt):
        raise NotImplementedError()


class ParticleChaser(Particle):
    def __init__(self, position, velocity, max_v=None, max_a=None):
        super().__init__(position, velocity, max_v, max_a)

        self._target = None

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, new_target):
        if new_target:
            if not isinstance(new_target, Particle):
                raise ValueError('new_target must be a Particle')

            self._target = new_target

    def move(self, dt):
        dt = float(dt)

        if self.target:
            displacement = self.target.position - self.position
            self.acceleration[:] = displacement[:]
            self._regularize_a()

        self.velocity += self.acceleration * dt
        self._regularize_v()

        self.position += self.velocity * dt  # + 0.5 * self.acceleration * dt * dt
