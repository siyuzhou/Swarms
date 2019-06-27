import os
import time
import argparse
import numpy as np

from classes import ParticleChaser
import utils


def create_chasers(n, m):
    """
    Create n particle chasers, each with m targets randomly selected with in the group.
    """
    if n < 1:
        raise ValueError('n must be a positive integer')

    if m < 1 or m > n - 1:
        raise ValueError('m must be a positive integer less than n')

    prev = None
    particles = []
    for _ in range(n):
        r = 20
        theta = np.random.rand() * 2 * np.pi
        x, y = r * np.cos(theta), r * np.sin(theta)
        v = np.random.uniform(-2, 2, 2)

        particles.append(ParticleChaser((x, y), v, ndim=2, max_speed=10, max_acceleration=10))

    edges = np.zeros((n, n))
    particle_idxs = np.arange(n)
    for i, p in enumerate(particles):
        for j in np.random.choice(particle_idxs[particle_idxs != i], m, replace=False):
            edges[i, j] = 1  # j is i's target, thus j influences i through edge j->i.
            p.add_target(particles[j])

    return particles, edges


def chasers_edges(n):
    """
    Edges for a list of chaser particles in which each agent chases its predecessor in the list.
    A 1 at Row i, Column j means Particle i influences Particle j. No influence
    is represented by 0.
    """
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        matrix[i, (i+1) % n] = 1

    return matrix


def simulation(_):
    np.random.seed()

    particles, edges = create_chasers(ARGS.num_particles, ARGS.num_targets)

    position_data = []
    velocity_data = []

    for _ in range(ARGS.steps):
        step_position = []
        step_velocity = []
        for p in particles:
            step_position.append(p.position.copy())
            step_velocity.append(p.velocity.copy())

            p.decide()

        for p in particles:
            p.move(ARGS.dt)

        position_data.append(step_position)
        velocity_data.append(step_velocity)

    return position_data, velocity_data, edges


def main():
    if not os.path.exists(ARGS.save_dir):
        os.makedirs(ARGS.save_dir)

    position_data_all, velocity_data_all, edge_data_all = utils.run_simulation(simulation,
                                                                               ARGS.instances,
                                                                               ARGS.processes,
                                                                               ARGS.batch_size)

    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_position.npy'), position_data_all)
    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_velocity.npy'), velocity_data_all)
    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_edge.npy'), edge_data_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-particles', '-n', type=int, default=5,
                        help='number of particles')
    parser.add_argument('--num-targets', '-m', type=int, default=1,
                        help='number of targets for each particle')
    parser.add_argument('--instances', type=int, default=1000,
                        help='number of instances to run')
    parser.add_argument('--steps', type=int, default=50,
                        help='number of time steps per simulation')
    parser.add_argument('--dt', type=float, default=0.3,
                        help='unit time step')
    parser.add_argument('--save-dir', type=str,
                        help='name of the save directory')
    parser.add_argument('--prefix', type=str, default='',
                        help='prefix for save files')
    parser.add_argument('--save-edges', action='store_true', default=False,
                        help='Deprecated. Now edges are always saved.')
    parser.add_argument('--processes', type=int, default=1,
                        help='number of parallel processes')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='number of simulation instances for each process')

    ARGS = parser.parse_args()

    ARGS.save_dir = os.path.expanduser(ARGS.save_dir)

    main()
