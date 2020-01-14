import argparse
import json
import time

import numpy as np

from classes import *


def animate(env, region):
    import matplotlib.pyplot as plt
    from matplotlib import animation

    plt.rcParams['animation.html'] = 'html5'

    def animate(i, scats, env):
        env.update(ARGS.dt)

        boid_positions = [agent.position for agent in env.population
                          if isinstance(agent, Boid)]
        if boid_positions:
            scats[0].set_offsets(boid_positions)

        vicsek_positions = [agent.position for agent in env.population
                            if isinstance(agent, Vicsek)]
        if vicsek_positions:
            scats[1].set_offsets(vicsek_positions)

        return scats[0], scats[1]

    xmin, xmax, ymin, ymax = region

    fig, ax = plt.subplots()
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal')

    scats = [ax.scatter([], [], color='b'),
             ax.scatter([], [], color='m')]

    for goal in env.goals:
        ax.scatter(*goal.position, color='g')
    for obstacle in env.obstacles:
        if not isinstance(obstacle, Wall):
            circle = plt.Circle(obstacle.position,
                                obstacle.size, color='r', fill=False)
            ax.add_patch(circle)

    anim = animation.FuncAnimation(fig, animate,
                                   fargs=(scats, env),
                                   frames=ARGS.steps, interval=20, blit=True)

    anim.save(ARGS.save_name+'.gif', dpi=80, writer='imagemagick')


def main():
    with open(ARGS.config) as f:
        model_config = json.load(f)

    if ARGS.boids > 0:
        Boid.set_model(model_config["boid"])
    if ARGS.vicseks > 0:
        Vicsek.set_model(model_config["vicsek"])

    region = (-100, 100, -100, 100)
    env = Environment2D(region)
    for _ in range(ARGS.boids):
        position = np.random.uniform(10, 80, 2)
        velocity = np.random.uniform(-15, 15, 2)

        agent = Boid(position, velocity, ndim=2, size=3, max_speed=10, max_acceleration=20)

        env.add_agent(agent)

    for _ in range(ARGS.vicseks):
        position = np.random.uniform(10, 80, 2)
        velocity = np.random.uniform(-15, 15, 2)

        agent = Vicsek(position, velocity, ndim=2, size=3, max_speed=10, max_acceleration=20)

        env.add_agent(agent)

    goal = Goal(np.random.uniform(-60, -40, 2), ndim=2)
    env.add_goal(goal)
    # Create a sphere obstacle within in +/- 50 of goal's position.
    for _ in range(ARGS.obstacles):
        sphere = Sphere(8, np.random.uniform(-40, 30, 2), ndim=2)
        env.add_obstacle(sphere)

    animate(env, region)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--boids', type=int, default=10,
                        help='number of boid agents')
    parser.add_argument('--vicseks', type=int, default=0,
                        help='number of vicsek agents')
    parser.add_argument('--obstacles', type=int, default=1,
                        help='number of obstacles')
    parser.add_argument('--steps', type=int, default=200,
                        help='number of simulation steps')
    parser.add_argument('--dt', type=float, default=0.2,
                        help='time resolution')
    parser.add_argument('--config', type=str, default='config/boid_vicsek_default.json',
                        help='path to config file')
    parser.add_argument('--save-name', type=str, default='demo',
                        help='name of the save file')

    ARGS = parser.parse_args()

    main()
