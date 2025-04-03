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
    data_collection_thread.start()

    data_collection_thread.join()

    return

    num_locations = 6
    num_days = 2
    seed = 0
    np.random.seed(seed)
    graph, coordinates = DataHandling.get_random_datum(seed)
    durations = np.random.randint(1, 96, num_locations) * 15
    durations[0] = 0
    graph = graph[:num_locations, :num_locations]
    coordinates = coordinates[:num_locations]

    Shorthands.gift_wrapping(num_locations, num_days, graph, durations,
                             coordinates)
    Shorthands.brute_force(
        num_locations, num_days, graph[:num_locations, :num_locations],
        durations[:num_locations], coordinates[:num_locations])
    Shorthands.greedy(num_locations, num_days, graph, durations, coordinates)
    Shorthands.k_means(num_locations, num_days, graph, durations, coordinates)
    Shorthands.k_means(num_locations, num_days, graph, durations, coordinates,
                       KMeans.RoutingMethods.BRUTE_FORCE)
    Shorthands.genetic_clustering(
        num_locations, num_days, graph, durations, coordinates)
    Shorthands.genetic_centroid_clustering(
        num_locations, num_days, graph, durations, coordinates)
    Shorthands.genetic_routing(
        num_locations, num_days, graph, durations, coordinates)
    return
    data_collection_thread.join()


if __name__ == '__main__':
    main()
