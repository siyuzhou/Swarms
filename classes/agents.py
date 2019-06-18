import numpy as np
from .particle import Particle
from .obstacles import Obstacle


class Agent(Particle):
    def __init__(self, position, velocity=None, acceleration=None,
                 ndim=None, max_speed=None, max_acceleration=None,
                 size=None, vision=None):
        """
        Create a boid with essential attributes.
        `ndim`: dimension of the space it resides in.
        `vision`: the visual range.
        `anticipation`: range of anticipation for its own motion.
        `comfort`: distance the agent wants to keep from other objects.
        `max_speed`: max speed the agent can achieve.
        `max_acceleratoin`: max acceleration the agent can achieve.
        """
        super().__init__(position, velocity, acceleration, ndim, max_speed, max_acceleration)

        self.size = float(size) if size else 0.
        self.vision = float(vision) if vision else np.inf

        self.neighbors = []
        self.obstacles = []

    def can_see(self, other):
        """Whether the boid can see the other."""
        return self.distance(other) < self.vision

    def observe(self, environment):
        """Observe the population and take note of neighbors."""
        self.neighbors = [other for other in environment.population
                          if self.can_see(other) and id(other) != id(self)]
        # To simplify computation, it is assumed that agent is aware of all
        # obstacles including the boundaries. In reality, the agent is only
        # able to see the obstacle when it is in visual range. This doesn't
        # affect agent's behavior, as agent only reacts to obstacles when in
        # proximity, and no early planning by the agent is made.
        self.obstacles = [obstacle for obstacle in environment.obstacles
                          if self.can_see(obstacle)]


class ParticleChaser(Particle):
    """A Particle agent that chases another Particle"""

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


class DynamicalChaser(Agent):
    """An Agent that chases its neighbors."""
    pass


class Boid(Agent):
    """Boid agent"""
    config = {
        "cohesion": 0.2,
        "separation": 2,
        "alignment": 0.2,
        "obstacle_avoidance": 2,
        "goal_steering": 0.5
    }

    def _cohesion(self):
        """Boids try to fly towards the center of neighbors."""
        if not self.neighbors:
            return np.zeros(self._ndim)

        center = np.zeros(self._ndim)
        for neighbor in self.neighbors:
            center += neighbor.position
        center /= len(self.neighbors)

        return center - self.position

    def _seperation(self):
        """Boids try to keep a small distance away from other objects."""
        repel = np.zeros(self._ndim)
        for neighbor in self.neighbors:
            distance = self.distance(neighbor)
            if distance < self.size:
                # Divergence protection.
                if distance < 0.01:
                    distance = 0.01

                repel += (self.position - neighbor.position) / \
                    distance / distance
                # No averaging taken place.
                # When two neighbors are in the same position, a stronger urge
                # to move away is assumed, despite that distancing itself from
                # one neighbor automatically eludes the other.
        return repel

    def _alignment(self):
        """Boids try to match velocity with neighboring boids."""
        # If no neighbors, no change.
        if not self.neighbors:
            return np.zeros(self._ndim)

        avg_velocity = np.zeros(self._ndim)
        for neighbor in self.neighbors:
            avg_velocity += neighbor.velocity
        avg_velocity /= len(self.neighbors)

        return avg_velocity - self.velocity

    def _obstacle_avoidance(self):
        """Boids try to avoid obstacles."""
        # NOTE: Assume there is always enough space between obstacles
        # Find the nearest obstacle in the front.
        min_distance = np.inf
        closest = -1
        for i, obstacle in enumerate(self.obstacles):
            distance = obstacle.distance(self.position)
            if (np.dot(-obstacle.direction(self.position), self.velocity) > 0  # In the front
                    and distance < min_distance):
                closest, min_distance = i, distance

        # No obstacles in front.
        if closest < 0:
            return np.zeros(self.ndim)

        obstacle = self.obstacles[closest]
        # normal distance of obstacle to velocity, note that min_distance is obstacle's distance
        obstacle_direction = -obstacle.direction(self.position)
        sin_theta = np.linalg.norm(
            np.cross(self.direction, obstacle_direction))
        normal_distance = (min_distance + obstacle.size) * \
            sin_theta - obstacle.size
        # Decide if self is on course of collision.
        if normal_distance < self.size:
            # normal direction away from obstacle
            cos_theta = np.sqrt(1 - sin_theta * sin_theta)
            turn_direction = self.direction * cos_theta - obstacle_direction
            turn_direction = turn_direction / np.linalg.norm(turn_direction)
            # Stronger the obstrution, stronger the turn.
            return turn_direction * (self.size - normal_distance)**2 / max(min_distance, self.size)

        # Return 0 if obstacle does not obstruct.
        return np.zeros(self.ndim)

    def _goal_seeking(self, goal):
        """Individual goal of the boid."""
        # As a simple example, suppose the boid would like to go as fast as it
        # can in the current direction when no explicit goal is present.
        if not goal:
            return self.velocity / self.speed

        # The urge to chase the goal is stronger when farther.
        offset = goal.position - self.position
        distance = np.linalg.norm(offset)
        target_speed = self.max_speed * min(1, distance / 20)
        target_velocity = target_speed * offset / distance
        return target_velocity - self.velocity

    def decide(self, goals):
        """Make decision for acceleration."""
        goal_steering = np.zeros(self.ndim)

        for goal in goals:
            goal_steering += self._goal_seeking(goal) * goal.priority

        self._acceleration = (self.config['cohesion'] * self._cohesion() +
                              self.config['separation'] * self._seperation() +
                              self.config['alignment'] * self._alignment() +
                              self.config['obstacle_avoidance'] * self._obstacle_avoidance() +
                              self.config['goal_steering'] * goal_steering)

    @classmethod
    def set_model(cls, config):
        cls.config['cohesion'] = config['cohesion']
        cls.config['separation'] = config['separation']
        cls.config['alignment'] = config['alignment']
        cls.config['obstacle_avoidance'] = config['obstacle_avoidance']
        cls.config['goal_steering'] = config['goal_steering']


class Vicsek(Agent):
    config = {
        'tau': 1.0,
        'A': 1.0,
        'B': 2.0,
        'k': 2.0,
        'kappa': 1.0
    }

    def _interaction(self, other):
        r = self.size + other.size
        d = np.linalg.norm(self.position - other.position)

        if isinstance(other, Obstacle):
            n = other.direction(self.position)
        else:
            n = (self.position - other.position) / d

        repulsion = self.config['A'] * np.exp((r - d) / self.config['B']) * n
        friction = 0
        if r > d:
            repulsion += self.config['k'] * (r - d) * n  # Body force.

            delta_v = other.velocity - self.velocity
            friction += self.config['kappa'] * \
                (r - d) * (delta_v - np.dot(delta_v, n) * n)

        return repulsion + friction

    def _goal_seeking(self, goal):
        """Individual goal of the boid."""
        # As a simple example, suppose the boid would like to go as fast as it
        # can in the current direction when no explicit goal is present.
        if not goal:
            return self.velocity / self.speed

        # The urge to chase the goal is stronger when farther.
        offset = goal.position - self.position
        distance = np.linalg.norm(offset)
        target_speed = self.max_speed * min(1, distance / 20)
        target_velocity = target_speed * offset / distance
        return target_velocity - self.velocity

    def decide(self, goals):
        """Make decision for acceleration."""
        goal_steering = np.zeros(self.ndim)

        for goal in goals:
            goal_steering += self._goal_seeking(goal) * goal.priority

        interactions = 0
        for neighbor in self.neighbors:
            interactions += self._interaction(neighbor)

        for obstacle in self.obstacles:
            interactions += self._interaction(obstacle)

        self._acceleration[:] = interactions[:] + goal_steering[:]

    @classmethod
    def set_model(cls, config):
        cls.config['tau'] = config['tau']
        cls.config['A'] = config['A']
        cls.config['B'] = config['B']
        cls.config['k'] = config['k']
        cls.config['kappa'] = config['kappa']
