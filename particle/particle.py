import numpy as np


class Particle:
    def __init__(self, position, velocity=None, acceleration=None,
                 ndim=None, max_speed=None, max_acceleration=None):
        self._ndim = ndim if ndim else 3

        # Max speed the boid can achieve.
        self.max_speed = float(max_speed) if max_speed else None
        self.max_acceleration = float(max_acceleration) if max_acceleration else None

        self.reset(position, velocity, acceleration)

    def reset(self, position, velocity=None, acceleration=None):
        """Initialize agent's spactial state."""
        self._position = np.zeros(self._ndim)
        self._velocity = np.zeros(self._ndim)
        self._acceleration = np.zeros(self._ndim)

        self._position[:] = position[:]

        if velocity is not None:
            self.velocity = velocity

        if acceleration is not None:
            self.acceleration = acceleration

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
        self._regularize_v()

    @property
    def speed(self):
        return np.linalg.norm(self.velocity)

    @property
    def acceleration(self):
        return self._acceleration

    @acceleration.setter
    def acceleration(self, acceleration):
        self._acceleration[:] = acceleration[:]
        self._regularize_a()

    def _regularize_v(self):
        if self.max_speed and self.speed > self.max_speed:
            self._velocity *= self.max_speed / self.speed
            return True
        return False

    def _regularize_a(self):
        if self.max_acceleration and np.linalg.norm(self.acceleration) > self.max_acceleration:
            self._acceleration *= self.max_acceleration / np.linalg.norm(self.acceleration)
            return True
        return False

    def move(self, dt):
        dt = float(dt)
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt  # + 0.5 * self.acceleration * dt * dt


class ParticleChaser(Particle):
    def __init__(self, position, velocity=None, acceleration=None,
                 ndim=None, max_speed=None, max_acceleration=None):
        super().__init__(position, velocity, acceleration, ndim, max_speed, max_acceleration)

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
            self.acceleration = displacement

        super().move(dt)
