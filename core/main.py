import time
from time import perf_counter
import h5py
import numpy as np

from algorithms.brute_force import brute_force
from algorithms.clustering import (
    cluster_and_solve,
    k_means
)
from algorithms.genetic_clustering import genetic_centroid_clustering
from algorithms.genetic_clustering import genetic_clustering
from algorithms.utilities import (
    ClusteringMethods,
    RoutingMethods,
    evaluate_route
)

from algorithms_oop import genetic

import utilities
from core import plotting


# todo: Finish this
def test_bruteforce_timings() -> None:
    with h5py.File('data/training_data.h5', 'a') as f:
        group = f[utilities.DataGroups.algorithm_performance.value]
        graphs = f[utilities.DataGroups.ordered_graphs.value]
        for i in range(len(graphs)):
            graph = graphs[str(i)]
            num_locations = graph.shape[0]
            num_days = int(graph.attrs[utilities.DataAttributes.days.value])
            np_graph = graph[:]

            perf_start = perf_counter()
            route = brute_force(num_locations, num_days, np_graph)
            perf_end = perf_counter()

            print(f"Bruteforce, {num_locations} locations, {num_days} days:"
                  f" {perf_end - perf_start} seconds. Route: {route}")

        # Store results


def collect_ordered_data() -> None:
    for num_locations in range(3, 26):
        num_days = 1
        while num_days < num_locations:
            datum = utilities.generate_test_datum(days=num_days,
                                                  free_time=1080,
                                                  number_of_nodes=num_locations)
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


if __name__ == '__main__':
    #data_collection_thread = threading.Thread(target=collect_ordered_data)
   # data_collection_thread.start()
    with h5py.File('data/training_data.h5', 'r') as f:
        graphs = f[utilities.DataGroups.ordered_graphs.value]
        graph = np.array(graphs['299'], copy=True, dtype=int)
        coordinates = graphs['299'].attrs['coordinates'][:]

    # GeneticCentroidClustering = genetic.GeneticCentroidClustering(1000, 30, 0.9, 0.1, 100,False)
    # route = GeneticCentroidClustering.find_route(coordinates, graph, 7, RoutingMethods.GREEDY)

    # GeneticClustering = genetic.GeneticClustering(1000, 30, 0.9, 0.1, 100, False)
    # route = GeneticClustering.find_route(graph, 25, 7, 31, RoutingMethods.GREEDY)

    #route = cluster_and_solve(coordinates, graph, 7, ClusteringMethods.K_MEANS, RoutingMethods.GREEDY)

    plotting.display_route(route, coordinates)

    #compare_bruteforce_to_bab()
    #with h5py.File('data/training_data.h5', 'r') as f:
    #    graphs = f[utilities.DataGroups.ordered_graphs.value]
    #    graph = np.array(graphs['1'], copy=True, dtype=int)
    '''
    graph =  np.array([
        [0, 10, 15, 20, 25],  # From start to other locations
        [10, 0, 35, 25, 30],  # From location 1 to others
        [15, 35, 0, 30, 10],  # From location 2 to others
        [20, 25, 30, 0, 15],  # From location 3 to others
        [25, 30, 10, 15, 0]   # From location 4 to others
    ])


    route = optimize_tourist_route(3, 5, graph)
    print(f"Solution: {route}")
    '''

    #utilities.display_coordinates(coordinates)
    #clusters = k_means(coordinates, 7, 25)
    #utilities.display_clusters(coordinates, clusters)
    '''
    _
    with h5py.File('data/training_data.h5', 'r') as f:
        graphs = f[utilities.DataGroups.ordered_graphs.value]
        graph = np.array(graphs['299'], copy=True, dtype=int)
        coordinates = graphs['299'].attrs['coordinates'][:]
        
    route1 = cluster_and_solve(coordinates, graph, 7,
                               ClusteringMethods.K_MEANS,
                               RoutingMethods.GREEDY)

    route2 = genetic_clustering(10000, 50, 0.9, 0.1, graph, 25, 7, 31, RoutingMethods.GREEDY)
    print(f"K-Means evaluation: {evaluate_route(route1, 7, graph)}")
    utilities.display_route(coordinates, route1)
    utilities.display_route(coordinates, route2)
    '''
    # route = brute_force(6, 2, graph)
    #utilities.display_route(coordinates, route)
    #    graphs = f['graphs']
    #    coordinates = graphs['0'].attrs['coordinates'][:]
    #    graph = graphs['0'][:]

    #utilities.generate_test_datum(number_of_nodes=5)

    # utilities.display_coordinates(coordinates)
    # utilities.display_graph(coordinates, graph)
    # graph = np.array([[0,1,4,2],[4,0,1,4],[1,4,0,4],[1,4,4,0]], dtype=np.int32)
    # route = algorithms.bruteForce(4,2, graph)
    # test_bruteforce_timings()

    #data_collection_thread.join()
