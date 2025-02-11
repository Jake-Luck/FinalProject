import time
from time import perf_counter
from concurrent.futures import ProcessPoolExecutor
import h5py
import numpy as np

import utilities

import algorithms.algorithms_wrapper as algorithms

# todo: Finish this
def test_bruteforce_timings():
    with h5py.File('data/training_data.h5', 'a') as f:
        group = f[utilities.DataGroups.algorithm_performance.value]
        graphs = f[utilities.DataGroups.ordered_graphs.value]
        for i in range(len(graphs)):
            graph = graphs[str(i)]
            num_locations = graph.shape[0]
            num_days = int(graph.attrs[utilities.DataAttributes.days.value])
            np_graph = graph[:]

            perf_start = perf_counter()
            route = algorithms.bruteForce(num_locations, num_days, np_graph)
            perf_end = perf_counter()

            print(f"Bruteforce, {num_locations} locations, {num_days} days:"
                  f" {perf_end - perf_start} seconds. Route: {route}")



def collect_ordered_data():
    for num_locations in range(2, 26):
        num_days = 1
        while num_days < num_locations:
            datum=utilities.generate_test_datum(days=num_days, free_time=1080,
                                                number_of_nodes=num_locations)
            if not isinstance(datum, int):
                utilities.save_test_datum(datum, utilities.DataGroups.ordered_graphs)
                num_days += 1
            if datum == 1:
                print("Waiting 1 minute for api.")
                time.sleep(60)  # Wait 1 minute before trying again
                print("Continuing...")
            if datum == 2:
                print("Waiting 24hrs for api.")
                time.sleep(86400)
                print("Restarting...")

def collect_test_data():
    while True:
        datum = utilities.generate_test_datum()
        if not isinstance(datum, int):
            utilities.save_test_datum(datum, utilities.DataGroups.regular_graphs)
        if datum == 1:
            print("Waiting 1 minute for api.")
            time.sleep(60)  # Wait 1 minute before trying again
            print("Continuing...")
        if datum == 2:
            print("Waiting 24hrs for api.")
            time.sleep(86400)
            print("Restarting...")


if __name__ == '__main__':
    #utilities.reset_database()
    # collect_test_data()
    # with h5py.File('data/training_data.h5', 'r') as f:
    #    graphs = f['graphs']
    #    coordinates = graphs['0'].attrs['coordinates'][:]
    #    graph = graphs['0'][:]

    # utilities.display_coordinates(coordinates)
    # utilities.display_graph(coordinates, graph)
    #graph = np.array([[0,1,4,2],[4,0,1,4],[1,4,0,4],[1,4,4,0]], dtype=np.int32)
    #route = algorithms.bruteForce(4,2, graph)
    test_bruteforce_timings()

