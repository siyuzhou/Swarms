import numpy as np

from swarms import Environment2D, Sphere, Boid, Goal

ENV_SIZE = 100
NDIM = 2
BOID_SIZE = 3
MAX_ACCELERATION = 5
MAX_SPEED = 10
SPHERE_SIZE = 8


def random_sphere(position1, position2, r):
    """Return an obstacle of radius r randomly placed between position1 and position2"""
    d = position1 - position2
    d_len = np.sqrt(d.dot(d))
    cos = d[0] / d_len
    sin = d[1] / d_len

    # Generat random x and y assuming d is aligned with x axis.
    x = np.random.uniform(2+r, d_len-r)
    y = np.random.uniform(-2*r, 2*r)

    # Rotate the alignment back to the actural d.
    true_x = x * cos + y * sin + position2[0]
    true_y = x * sin - y * cos + position2[1]

    return Sphere(r, [true_x, true_y], ndim=2)


def random_boid(position_bound, max_speed, max_acceleration):
    position = np.random.uniform(-position_bound, position_bound, NDIM)
    velocity = np.random.uniform(-max_speed, max_speed, NDIM)

    agent = Boid(position, velocity, ndim=NDIM, vision=None, size=BOID_SIZE,
                 max_speed=max_speed, max_acceleration=max_acceleration)

    return agent


def _reward_to_goal(distance_to_goal):
    """
    Reward function determined by distance to goal.
    """
    return 0.2 * (1 / (distance_to_goal / ENV_SIZE + 0.1) - 0.25)


def _reward_agent_pair(relative_distance, collision):
    """
    Reward function determined by agent relative distance
    """
    return 0.05 * (0 - relative_distance / ENV_SIZE) * (1 - collision) - 10 * collision


def _reward_to_obstacle(distance_to_obstacle, collision):
    """
    Reward function determined by distance to obstacle.
    """
    return -10 * collision


class BoidSphereEnv2D:
    def __init__(self, num_boids, num_obstacles, num_goals, dt,
                       env_size=ENV_SIZE, ndim=NDIM, 
                       boid_size=BOID_SIZE, sphere_size=SPHERE_SIZE,
                       max_speed=MAX_SPEED, max_acceleration=MAX_ACCELERATION, 
                       config=None):

        self._env = Environment2D((-env_size, env_size, -env_size, env_size))

        self.size = env_size
        self.ndim = ndim

        self.num_agents = num_boids
        self.num_obstacles = num_obstacles
        self.num_goals = num_goals

        self.dt = dt

        self.config = config if config else {}
        self.config.update({
            'boid_size': boid_size,
            'sphere_size': sphere_size,
            'max_speed': max_speed,
            'max_acceleration': max_acceleration
        })

        self._agent_states = np.zeros((num_boids, 2 * ndim))
        self._obstacle_states = np.zeros((num_obstacles, 2 * ndim))
        self._goal_states = np.zeros((num_goals, 2 * ndim))

        self._agent_pair_distances = np.zeros((num_boids, num_boids))
        self._agent_obstacle_distances = np.zeros((num_boids, num_obstacles))
        self._agent_goal_distances = np.zeros((num_boids, num_goals))

        self._agent_pair_collision = np.zeros((num_boids, num_boids))
        self._agent_obstacle_collision = np.zeros((num_boids, num_obstacles))

        self.reset()

    def reset(self):
        self._env.goals.clear()
        for _ in range(self.num_goals):
            goal = Goal(np.random.uniform(-0.4*self.size, 0.4*self.size, self.ndim), ndim=self.ndim)
            self._env.add_goal(goal)

        self._env.population.clear()
        for _ in range(self.num_agents):
            agent = random_boid(0.8 * self.size, self.config['max_speed'], self.config['max_acceleration'])
            self._env.add_agent(agent)

        avg_boids_position = np.mean(
            np.vstack([agent.position for agent in self._env.population]), axis=0)

        avg_goals_position = np.mean(
            np.vstack([goal.position for goal in self._env.goals]), axis=0)

        self._env._obstacles.clear()
        for _ in range(self.num_obstacles):
            sphere = random_sphere(avg_boids_position * 1.5, avg_goals_position * 1.5, self.config['sphere_size'])
            self._env.add_obstacle(sphere)

        self._update_states()
        return self._get_env_state()

    def _update_states(self):
        self._update_agent_states()
        self._update_obstacle_states()
        self._update_goal_states()

        self._update_agent_pair_distances()
        self._update_agent_obstacle_distances()
        self._update_agent_goal_distances()

    def _update_agent_states(self):
        agent_pos = np.vstack([agent.position.copy() for agent in self._env.population])
        agent_vel = np.vstack([agent.velocity.copy() for agent in self._env.population])
        self._agent_states = np.concatenate([agent_pos, agent_vel], -1)

    def _update_obstacle_states(self):
        sphere_pos = np.vstack([sphere.position.copy() for sphere in self._env._obstacles])
        sphere_vel = np.vstack([np.zeros(self.ndim) for _ in self._env._obstacles])
        self._obstacle_states = np.concatenate([sphere_pos, sphere_vel], -1)

    def _update_goal_states(self):
        goal_pos = np.vstack([goal.position.copy() for goal in self._env.goals])
        goal_vel = np.vstack([np.zeros(self.ndim) for _ in self._env.goals])
        self._goal_states = np.concatenate([goal_pos, goal_vel], -1)

    def _update_agent_pair_distances(self):
        agent_positions = self._agent_states[:, :self.ndim]
        agent_positions_from = agent_positions[:, np.newaxis, :]
        agent_positions_to = agent_positions[np.newaxis, :, :]
        agent_pair_displacements = agent_positions_from - agent_positions_to

        agent_pair_distances = np.linalg.norm(agent_pair_displacements, axis=-1)
        self._agent_pair_distances = agent_pair_distances

    def _update_agent_obstacle_distances(self):
        agent_positions = self._agent_states[:, :self.ndim]
        obstacle_positions = self._obstacle_states[:, :self.ndim]
        agent_positions = agent_positions[:, np.newaxis, :]
        obstacle_positions = obstacle_positions[np.newaxis, :, :]
        agent_obstacle_displacements = agent_positions - obstacle_positions

        agent_obstacle_distances = np.linalg.norm(agent_obstacle_displacements, axis=-1)
        self._agent_obstacle_distances = agent_obstacle_distances

    def _update_agent_goal_distances(self):
        agent_positions = self._agent_states[:, :self.ndim]
        goal_positions = self._goal_states[:, :self.ndim]
        agent_positions = agent_positions[:, np.newaxis, :]
        goal_positions = goal_positions[np.newaxis, :, :]
        agent_goal_displacements = agent_positions - goal_positions

        agent_goal_distances = np.linalg.norm(agent_goal_displacements, axis=-1)
        self._agent_goal_distances = agent_goal_distances

    def _get_env_state(self):
        return self._agent_states, self._obstacle_states, self._goal_states

    def step(self, action):
        a = action / self.dt

        assert len(a) == len(self._env.population)
        for agent, acc in zip(self._env.population, a):
            agent.acceleration = acc

        # Cannot use env.update, since it overrides acceleration
        for agent in self._env.population:
            agent.move(self.dt)

        self._update_states()
        self._check_collision()

        next_state = self._get_env_state()
        reward = self._reward()
        done = np.any(self._agent_pair_collision) or np.any(self._agent_obstacle_collision)

        return next_state, reward, done

    def _reward(self):
        agent_pair_reward = _reward_agent_pair(
            self._agent_pair_distances, self._agent_pair_collision)

        agent_obstacle_reward = _reward_to_obstacle(
            self._agent_obstacle_distances, self._agent_obstacle_collision)

        agent_goal_reward = _reward_to_goal(self._agent_goal_distances)

        agent_reward = np.sum(agent_pair_reward, axis=-1) + np.sum(agent_obstacle_reward, axis=-1) + \
            np.sum(agent_goal_reward, axis=-1)

        # obstacle_reward = np.zeros(self.num_obstacles)
        # goal_reward = np.zeros(self.num_goals)

        return agent_reward #, obstacle_reward, goal_reward

    def _check_collision(self):
        """
        Check if there is any collision with
        """
        agent_pair_collision = self._agent_pair_distances < 2 * self.config['boid_size']
        np.fill_diagonal(agent_pair_collision, False)

        agent_obstacle_collision = self._agent_obstacle_distances < self.config['boid_size'] + self.config['sphere_size']

        self._agent_pair_collision = agent_pair_collision
        self._agent_obstacle_collision = agent_obstacle_collision
