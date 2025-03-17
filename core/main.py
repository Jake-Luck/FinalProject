"""
Program launching point
"""
from algorithms.clustering import KMeans
from core.algorithm_shorthands import Shorthands
from core.data_handling import DataHandling

import numpy as np
import threading


def main():
    """
    Launching point of the program, always starts a thread for collecting data.
    """
    data_handler = DataHandling()
    data_collection_thread = threading.Thread(
        target=data_handler.collect_test_data)
    # data_collection_thread.start()

    num_locations = 25
    num_days = 5
    seed = 4
    np.random.seed(seed)
    graph, coordinates = DataHandling.get_random_datum(seed)
    durations = np.random.randint(1, 96, num_locations) * 15
    durations[0] = 0
    graph = graph[:num_locations, :num_locations]
    coordinates = coordinates[:num_locations]

    brute_force_route = Shorthands.brute_force(
        num_locations=9, num_days=3, graph=graph[:9, :9],
        coordinates=coordinates[:9],    durations=durations)
    greedy_route = Shorthands.greedy(
        num_locations, num_days, graph=graph, coordinates=coordinates,
        durations=durations)
    kmeans_greedy_route = Shorthands.k_means(
        num_locations, num_days, graph=graph, coordinates=coordinates,
        durations=durations)
    kmeans_bruteforce_route = Shorthands.k_means(
        num_locations, num_days, graph=graph, coordinates=coordinates,
        durations=durations,
        routing_algorithm=KMeans.RoutingMethods.BRUTE_FORCE)
    genetic_clustering_route = Shorthands.genetic_clustering(
        num_locations, num_days, graph=graph, coordinates=coordinates,
        durations=durations, mutation_probability=0.1,
        generations_per_update=1)
    genetic_centroid_clustering_route = Shorthands.genetic_centroid_clustering(
        num_locations, num_days, graph=graph, coordinates=coordinates,
        durations=durations, mutation_probability=0.1,
        generations_per_update=1)

    # data_collection_thread.join()


if __name__ == '__main__':
    main()
