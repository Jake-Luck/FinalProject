import threading
import time
from time import perf_counter
import h5py
import numpy as np

from algorithms import genetic
from algorithms import clustering
from algorithms.routing import Routing

from core import utilities
from core import plotting


# todo: Finish this
def test_bruteforce_timings() -> None:
    with h5py.File('data/training_data.h5', 'a') as f:
        results_group = f[utilities.DataGroups.algorithm_performance.value]
        _graphs = f[utilities.DataGroups.ordered_graphs.value]
        for i in range(len(graphs)):
            _graph = graphs[str(i)]
            _num_locations = _graph.shape[0]
            _num_days = int(_graph.attrs[utilities.DataAttributes.days.value])
            np_graph = _graph[:]

            _routing = Routing()
            perf_start = perf_counter()
            _route = _routing.brute_force(_num_locations, _num_days, np_graph)
            perf_end = perf_counter()

            print(f"Bruteforce, {_num_locations} locations, {_num_days} days:"
                  f" {perf_end - perf_start} seconds. Route: {_route}")

        # Store results


def collect_ordered_data() -> None:
    for num_locations in range(3, 26):
        num_days = 1
        while num_days < num_locations:
            datum = utilities.generate_test_datum()
            if not isinstance(datum, int):
                utilities.save_test_datum(datum,
                                          utilities.DataGroups.regular_graphs)
                num_days += 1
            if datum == 1:
                print("Waiting 1 minute for api.")
                time.sleep(60)  # Wait 1 minute before trying again
                print("Continuing...")
            if datum == 2:
                print("Waiting 24hrs for api.")
                time.sleep(86400)
                print("Restarting...")


def collect_test_data() -> None:
    while True:
        datum = utilities.generate_test_datum()
        if not isinstance(datum, int):
            utilities.save_test_datum(datum,
                                      utilities.DataGroups.regular_graphs)
        if datum == 1:
            print("Waiting 1 minute for api.")
            time.sleep(60)  # Wait 1 minute before trying again
            print("Continuing...")
        if datum == 2:
            print("Waiting 24hrs for api.")
            time.sleep(86400)
            print("Restarting...")


def main():
    data_collection_thread = threading.Thread(target=collect_ordered_data)
    data_collection_thread.start()

    #with h5py.File('data/training_data.h5', 'r') as f:
    #    graphs = f[utilities.DataGroups.ordered_graphs.value]
    #    graph = np.array(graphs['12'], copy=True, dtype=int)
    #    coordinates = graphs['12'].attrs['coordinates'][:]

    num_days = 7
    num_locations = 25
    route_length = 31

    #routing = Routing()
    #route = routing.brute_force(num_locations, num_days, graph)

    #    clustering_algorithm = genetic.GeneticCentroidClustering(1000, 30, 0.9,
    #                                                             0.1, 100, False)
    #    cluster_assignments = clustering_algorithm.find_clusters(
    #        coordinates, graph, num_days,
    #        clustering_algorithm.RoutingMethods.GREEDY)

    #    clustering_algorithm = genetic.GeneticClustering(1000, 30, 0.9, 0.1, 100,
    #                                                     False)
    #    cluster_assignments = clustering_algorithm.find_clusters(
    #        graph, num_locations, num_days, route_length,
    #        clustering_algorithm.RoutingMethods.GREEDY)

    #clustering_algorithm = clustering.KMeans()
    #cluster_assignments = clustering_algorithm.find_clusters(
    #    coordinates, num_days, num_locations)

    #route = clustering_algorithm.find_route_from_cluster_assignments(
    #    cluster_assignments, num_days,
    #    clustering_algorithm.RoutingMethods.BRUTE_FORCE, graph)

    #plotting.display_route(route, coordinates)

    data_collection_thread.join()


if __name__ == '__main__':
    main()