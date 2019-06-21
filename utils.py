import time
import functools
import multiprocessing


def run_simulation(simulation, instances, processes=1, batch=100, silent=False):
    pool = multiprocessing.Pool(processes=processes)
    position_data_all = []
    velocity_data_all = []
    edge_data_all = []

    remaining_instances = instances

    prev_time = time.time()
    while remaining_instances > 0:
        n = min(remaining_instances, batch)
        data_pool = pool.map(simulation, range(n))

        position_pool, velocity_pool, edge_pool = zip(*data_pool)

        remaining_instances -= n
        if not silent:
            print('Simulation {}/{}... {:.1f}s'.format(instances - remaining_instances,
                                                       instances, time.time()-prev_time))
        prev_time = time.time()

        position_data_all.extend(position_pool)
        velocity_data_all.extend(velocity_pool)
        edge_data_all.extend(edge_pool)

    return position_data_all, velocity_data_all, edge_data_all
