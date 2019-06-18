import os
import time
import argparse
import numpy as np

from classes import ParticleChaser
import utils


def create_chasers(n):
    """
    Create n particle chasers. 
    Each particle chases the previous one in the list of particles.
    """
    if n < 1:
        raise ValueError('n must be a positive integer')

    prev = None
    particles = []
    for _ in range(n):
        r = 20
        theta = np.random.rand() * 2 * np.pi
        x, y = r * np.cos(theta), r * np.sin(theta)
        v = np.random.uniform(-2, 2, 2)

        p = ParticleChaser((x, y), v, ndim=2, max_speed=10, max_acceleration=10)

        p.target = prev
        particles.append(p)

        prev = p

    particles[0].target = prev

    return particles


def chasers_edges(n):
    """
    Adjacency matrix of the chaser swarm defined in function `create_chasers(n)`.
    A 1 at Row i, Column j means Particle i influences Particle j. No influence
    is represented by 0.
    """
    matrix = np.zeros((n, n), dtype=int)
    for i in range(n):
        matrix[i, (i+1) % n] = 1

    return matrix


def simulation(_):
    particles = create_chasers(ARGS.num_particles)

    position_data = []
    velocity_data = []

    for _ in range(ARGS.steps):
        step_position = []
        step_velocity = []
        for p in particles:
            step_position.append(p.position.copy())
            step_velocity.append(p.velocity.copy())

            p.move(ARGS.dt)

        position_data.append(step_position)
        velocity_data.append(step_velocity)

    return position_data, velocity_data


def main():
    if not os.path.exists(ARGS.save_dir):
        os.makedirs(ARGS.save_dir)

    position_data_all, velocity_data_all = utils.run_simulation(simulation,
                                                                ARGS.instances, ARGS.processes,
                                                                ARGS.batch_size)

    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_position.npy'), position_data_all)
    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_velocity.npy'), velocity_data_all)

    if ARGS.save_edges:
        edges = chasers_edges(ARGS.num_particles)
        edges_all = np.tile(edges, (ARGS.instances, 1, 1))
        np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_edge.npy'), edges_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-particles', '-n', type=int, default=5,
                        help='number of particles')
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
                        help='turn on to save edges')
    parser.add_argument('--processes', type=int, default=1,
                        help='number of parallel processes')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='number of simulation instances for each process')

    ARGS = parser.parse_args()

    ARGS.save_dir = os.path.expanduser(ARGS.save_dir)
    
    main()
