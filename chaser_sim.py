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

    if m == 'x':
        m = np.random.randint(1, n)
    elif m == 'y':
        pass
    elif int(m) < 1 or int(m) > n - 1:
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
        if m == 'y':
            k = np.random.randint(1, n)
        else:
            k = int(m)

        for j in np.random.choice(particle_idxs[particle_idxs != i], k, replace=False):
            edges[j, i] = 1  # j is i's target, thus j influences i through edge j->i.
            p.add_target(particles[j])

    return particles, edges


# def chasers_edges(n):
#     """
#     Edges for a list of chaser particles in which each agent chases its predecessor in the list.
#     A 1 at Row i, Column j means Particle i influences Particle j. No influence
#     is represented by 0.
#     """
#     matrix = np.zeros((n, n), dtype=int)
#     for i in range(n):
#         matrix[i, (i+1) % n] = 1

#     return matrix


def simulation(_):
    np.random.seed()

    particles, edges = create_chasers(ARGS.num_particles, ARGS.num_targets)

    position_data = []
    velocity_data = []
    time_data = []

    num_skips = 0
    skip = False
    step = 0
    t = 0
    while step < ARGS.steps:
        if skip:
            num_skips += 1
        else:
            num_skips = 0
            
            step_position = []
            step_velocity = []
            for p in particles:
                step_position.append(p.position.copy())
                step_velocity.append(p.velocity.copy())

            position_data.append(step_position)
            velocity_data.append(step_velocity)
            time_data.append(t)

            step += 1

        for p in particles:
            p.decide()

        for p in particles:
            p.move(ARGS.dt)

        t += ARGS.dt

        if ARGS.adaptive and num_skips < ARGS.max_skip:
            for p in particles:
                if np.linalg.norm(p.acceleration) / p.speed * ARGS.dt > ARGS.max_rate:
                    skip = False
                    break
            else:
                skip = True
        else:
            skip = False

    return position_data, velocity_data, edges, time_data


def main():
    if not os.path.exists(ARGS.save_dir):
        os.makedirs(ARGS.save_dir)

    position_data_all, velocity_data_all, edge_data_all, time_data_all = \
        utils.run_simulation(simulation, ARGS.instances, ARGS.processes, ARGS.batch_size)

    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_position.npy'), position_data_all)
    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_velocity.npy'), velocity_data_all)
    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_edge.npy'), edge_data_all)
    np.save(os.path.join(ARGS.save_dir, ARGS.prefix+'_time.npy'), time_data_all)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-particles', '-n', type=int, default=5,
                        help='number of particles')
    parser.add_argument('--num-targets', '-m', type=str, default=1,
                        help="number of targets for each particle\n"
                             "use 'x' for a random number\n"
                             "use 'y' for a different random number for each agent")
    parser.add_argument('--instances', type=int, default=1000,
                        help='number of instances to run')
    parser.add_argument('--steps', type=int, default=50,
                        help='number of time steps per simulation')
    parser.add_argument('--dt', type=float, default=0.3,
                        help='unit time step')
    parser.add_argument('--adaptive', action='store_true', default=False,
                        help='save states only when dv/v exceeds max rate')
    parser.add_argument('--max-rate', type=float,
                        help='max change rate allowed for skip steps')
    parser.add_argument('--max-skip', type=int, default=5,
                        help='max number of steps allowed to skip in adaptive mode')
    parser.add_argument('--save-dir', type=str,
                        help='name of the save directory')
    parser.add_argument('--prefix', type=str, default='',
                        help='prefix for save files')
    parser.add_argument('--processes', type=int, default=1,
                        help='number of parallel processes')
    parser.add_argument('--batch-size', type=int, default=100,
                        help='number of simulation instances for each process')

    ARGS = parser.parse_args()

    ARGS.save_dir = os.path.expanduser(ARGS.save_dir)

    main()
