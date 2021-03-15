import numpy as np
from .entity import Entity
from .obstacles import Obstacle
from .goals import Goal


class Agent(Entity):
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

        # Goal.
        self.goal = None

        # Perceived neighbors and obstacles.
        self.neighbors = []
        self.obstacles = []

    def distance(self, other):
        """Distance from the other objects."""
        if isinstance(other, Agent):
            return np.linalg.norm(self.position - other.position)
        # If other is not agent, let other tell the distance.
        try:
            return other.distance(self.position)
        except AttributeError:
            raise ValueError(f'cannot determine distance with {type(other)}')

    def set_goal(self, goal):
        if (goal is not None) and not isinstance(goal, Goal):
            raise ValueError("'goal' must be an instance of 'Goal' or None")
        self.goal = goal

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

    def decide(self):
        raise NotImplementedError()


class Chaser(Entity):
    """A Particle agent that chases another Particle"""

    def __init__(self, position, velocity=None, acceleration=None,
                 ndim=None, max_speed=None, max_acceleration=None):
        super().__init__(position, velocity, acceleration, ndim, max_speed, max_acceleration)

        self._targets = []

    @property
    def targets(self):
        return self._targets

    def add_target(self, new_target):
        if new_target:
            if not isinstance(new_target, Entity):
                raise ValueError('new_target must be a Entity')

            self._targets.append(new_target)

    def decide(self):
        displacement = np.zeros(self.ndim)
        for target in self.targets:
            displacement += target.position - self.position

        self.acceleration = displacement


class Boid(Agent):
    """Boid agent"""
    config = {
        "cohesion": 0.2,
        "separation": 2,
        "alignment": 0.2,
        "obstacle_avoidance": 2,
        "goal_steering": 0.5,
        "neighbor_interaction_mode": "avg"
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

    def _separation(self):
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

        # Whether the avg or sum of the influence from neighbors is used.
        if self.config["neighbor_interaction_mode"] == 'avg' and self.neighbors:
            repel /= len(self.neighbors)
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
        v_direction = self.velocity / self.speed
        sin_theta = np.linalg.norm(
            np.cross(v_direction, obstacle_direction))
        normal_distance = (min_distance + obstacle.size) * \
            sin_theta - obstacle.size
        # Decide if self is on course of collision.
        if normal_distance < self.size:
            # normal direction away from obstacle
            cos_theta = np.sqrt(1 - sin_theta * sin_theta)
            turn_direction = v_direction * cos_theta - obstacle_direction
            turn_direction = turn_direction / np.linalg.norm(turn_direction)
            # Stronger the obstrution, stronger the turn.
            return turn_direction * (self.size - normal_distance)**2 / max(min_distance, self.size)

        # Return 0 if obstacle does not obstruct.
        return np.zeros(self.ndim)

    def _goal_seeking(self):
        """Individual goal of the boid."""
        # As a simple example, suppose the boid would like to go as fast as it
        # can in the current direction when no explicit goal is present.
        if not self.goal:
            return self.velocity / self.speed

        # The urge to chase the goal is stronger when farther.
        offset = self.goal.position - self.position
        distance = np.linalg.norm(offset)
        target_speed = self.max_speed * min(1, distance / 20)
        target_velocity = target_speed * offset / distance
        return target_velocity - self.velocity

    def decide(self):
        """Make decision for acceleration."""
        self.acceleration = (self.config['cohesion'] * self._cohesion() +
                             self.config['separation'] * self._separation() +
                             self.config['alignment'] * self._alignment() +
                             self.config['obstacle_avoidance'] * self._obstacle_avoidance() +
                             self.config['goal_steering'] * self._goal_seeking())

    @classmethod
    def set_model(cls, config):
        # Throw out unmatched keys.
        config = {k: v for k, v in config.items() if k in cls.config}
        cls.config.update(config)


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

    def _goal_seeking(self):
        """Individual goal of the boid."""
        # As a simple example, suppose the boid would like to go as fast as it
        # can in the current direction when no explicit goal is present.
        if not self.goal:
            return self.velocity / self.speed

        # The urge to chase the goal is stronger when farther.
        offset = self.goal.position - self.position
        distance = np.linalg.norm(offset)
        target_speed = self.max_speed * min(1, distance / 20)
        target_velocity = target_speed * offset / distance
        return target_velocity - self.velocity

    def decide(self):
        """Make decision for acceleration."""
        goal_steering = np.zeros(self.ndim)

        goal_steering += self._goal_seeking()

        interactions = 0
        for neighbor in self.neighbors:
            interactions += self._interaction(neighbor)

        for obstacle in self.obstacles:
            interactions += self._interaction(obstacle)

        self.acceleration = interactions + goal_steering

    @classmethod
    def set_model(cls, config):
        config = {k: v for k, v in config.items() if k in cls.config}
        cls.config.update(config)
