import os
import argparse
import json

import numpy as np

from swarms import *
import utils


def random_obstacle(position1, position2, r):
    """Return an obstacle of radius r randomly placed between position1 and position2"""
    d = position1 - position2
    d_len = np.sqrt(d.dot(d))
    cos = d[0] / d_len
    sin = d[1] / d_len

    # Generat random x and y assuming d is aligned with x axis.
    x = np.random.uniform(2 + r, d_len - r)
    y = np.random.uniform(-2 * r, 2 * r)

    # Rotate the alignment back to the actural d.
    true_x = x * cos + y * sin + position2[0]
    true_y = x * sin - y * cos + position2[1]

    return Sphere(r, [true_x, true_y], ndim=2)


def system_edges(obstacles, boids, vicseks):
    """Edge types of the directed graph representing the influences between
    elements of the system.
        |    |Goal|Obst|Boid|Visc|
        |Goal| 0  | 0  | 1  | 5  |
        |Obst| 0  | 0  | 2  | 6  |
        |Boid| 0  | 0  | 3  | 7  |
        |Visc| 0  | 0  | 4  | 8  |
    """
    # If boids == 0, edges would be same as if vicseks were boids
    if boids == 0:
        boids, vicseks = vicseks, boids

    particles = 1 + obstacles + boids + vicseks
    edges = np.zeros((particles, particles), dtype=int)

    up_to_goal = 1
    up_to_obs = up_to_goal + obstacles
    up_to_boids = up_to_obs + boids

    edges[0, up_to_obs:up_to_boids] = 1  # influence from goal to boid.
    edges[up_to_goal:up_to_obs, up_to_obs:up_to_boids] = 2  # influence from obstacle to boid.
    edges[up_to_obs:up_to_boids, up_to_obs:up_to_boids] = 3  # influence from boid to boid.
    edges[up_to_boids:, up_to_obs:up_to_boids] = 4  # influence from vicsek to boid.

    edges[0, up_to_boids:] = 5  # influence from goal to vicsek.
    edges[up_to_goal:up_to_obs, up_to_boids:] = 6  # influence from obstacle to vicsek.
    edges[up_to_obs:up_to_boids, up_to_boids:] = 7  # influence from obstacle to agent.
    edges[up_to_boids:, up_to_boids:] = 8  # influence from viscek to viscek.

    np.fill_diagonal(edges, 0)
    return edges


def simulation(args, _):
    np.random.seed()

    region = (-100, 100, -100, 100)

    env = Environment2D(region)

    goal = Goal(np.random.uniform(-40, 40, 2), ndim=2)
    env.add_goal(goal)

    for _ in range(args.boids):
        position = np.random.uniform(-80, 80, 2)
        velocity = np.random.uniform(-15, 15, 2)

        agent = Boid(position, velocity, ndim=2, vision=args.vision, size=args.size,
                     max_speed=10, max_acceleration=5)
        agent.set_goal(goal)

        env.add_agent(agent)
    for _ in range(args.vicseks):
        position = np.random.uniform(-80, 80, 2)
        velocity = np.random.uniform(-15, 15, 2)

        agent = Vicsek(position, velocity, ndim=2, vision=args.vision, size=args.size,
                       max_speed=10, max_acceleration=5)
        agent.set_goal(goal)

        env.add_agent(agent)

    # Create a sphere obstacle near segment between avg boids position and goal position.
    avg_boids_position = np.mean(
        np.vstack([agent.position for agent in env.population]), axis=0)

    spheres = []
    for _ in range(args.obstacles):
        sphere = random_obstacle(avg_boids_position, goal.position, 8)
        spheres.append(sphere)
        env.add_obstacle(sphere)

    position_data = []
    velocity_data = []
    time_data = []
    t = 0
    for _ in range(args.steps):
        env.update(args.dt)
        position_data.append([goal.position for goal in env.goals] +
                             [sphere.position for sphere in spheres] +
                             [agent.position.copy() for agent in env.population])
        velocity_data.append([np.zeros(2) for goal in env.goals] +
                             [np.zeros(2) for sphere in spheres] +
                             [agent.velocity.copy() for agent in env.population])
        time_data.append(t)
        t += args.dt

    position_data, velocity_data = np.asarray(position_data), np.asarray(velocity_data)
    timeseries_data = np.concatenate([position_data, velocity_data], axis=-1)

    edge_data = system_edges(args.obstacles, args.boids, args.vicseks)

    return timeseries_data, edge_data, time_data


def main():
    if not os.path.exists(ARGS.save_dir):
        os.makedirs(ARGS.save_dir)

    file_dir = os.path.dirname(os.path.realpath(__file__))
    with open(os.path.join(file_dir, ARGS.config)) as f:
        model_config = json.load(f)

    if ARGS.boids > 0:
        Boid.set_model(model_config["boid"])
    if ARGS.vicseks > 0:
        Vicsek.set_model(model_config["vicsek"])

    timeseries_data_all, edge_data_all, time_data_all = \
        utils.run_simulation(simulation, ARGS, ARGS.instances, ARGS.processes, ARGS.batch_size)

    np.save(os.path.join(ARGS.save_dir,
                         f'{ARGS.prefix}_timeseries{ARGS.suffix}.npy'), timeseries_data_all)
    np.save(os.path.join(ARGS.save_dir, f'{ARGS.prefix}_edge{ARGS.suffix}.npy'), edge_data_all)
    np.save(os.path.join(ARGS.save_dir, f'{ARGS.prefix}_time{ARGS.suffix}.npy'), time_data_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--boids', type=int, default=0,
                        help='number of boid agents')
    parser.add_argument('--vicseks', type=int, default=0,
                        help='number of vicsek agents')
    parser.add_argument('--obstacles', type=int, default=0,
                        help='number of obstacles')
    parser.add_argument('--vision', type=float, default=None,
                        help='vision range to determine range of interaction')
    parser.add_argument('--size', type=float, default=3,
                        help='agent size')
    parser.add_argument('--steps', type=int, default=200,
                        help='number of simulation steps')
    parser.add_argument('--instances', type=int, default=1,
                        help='number of simulation instances')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='time resolution')
    parser.add_argument('--config', type=str, default='config/boid_vicsek_default.json',
                        help='path to config file')
    parser.add_argument('--save-dir', type=str, default='.',
                        help='name of the save directory')
    parser.add_argument('--prefix', type=str, default='',
                        help='prefix for save files')
    parser.add_argument('--suffix', type=str, default='',
                        help='suffix for save files')
    parser.add_argument('--processes', type=int, default=1,
                        help='number of parallel processes')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='number of simulation instances for each process')

    ARGS = parser.parse_args()

    ARGS.save_dir = os.path.expanduser(ARGS.save_dir)

    main()
