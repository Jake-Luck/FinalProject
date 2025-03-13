"""
Program launching point
"""
import h5py

from core.algorithm_shorthands import Shorthands
from core.data_handling import DataHandling

import threading


def main():
    """
    Launching point of the program, always starts a thread for collecting data.
    """
    data_handler = DataHandling()
    data_collection_thread = threading.Thread(
        target=data_handler.collect_test_data)
    data_collection_thread.start()

    num_locations = 8
    num_days = 3
    graph, coordinates = DataHandling.get_random_datum()
    brute_force_route = Shorthands.brute_force(
        num_locations, num_days, graph=graph, coordinates=coordinates)
    kmeans_route = Shorthands.k_means(
        num_locations, num_days, graph=graph, coordinates=coordinates)
    genetic_clustering_route = Shorthands.genetic_clustering(
        num_locations, num_days, generations_per_update=0, graph=graph,
        coordinates=coordinates)
    genetic_centroid_clustering_route = Shorthands.genetic_centroid_clustering(
        num_locations, num_days, generations_per_update=0, graph=graph,
        coordinates=coordinates)

    data_collection_thread.join()


if __name__ == '__main__':
    main()
