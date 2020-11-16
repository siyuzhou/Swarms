from .agents import Boid, Vicsek, ParticleChaser, DynamicalChaser
from .obstacles import Wall, Sphere
from .goals import Goal
from .environments import Environment2D


from gym.envs.registration import register

register(
    id='swarms-v0',
    entry_point='swarms.rl_extensions:BoidSphereEnv2D',
)