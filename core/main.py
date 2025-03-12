"""
Program launching point
"""
from core.algorithm_shorthands import Shorthands
from core.data_handling import DataHandling

import threading


def main():
    """
    Launching point of the program, always starts a thread for collecting data.
    """
    data_handler = DataHandling()
    data_collection_thread = threading.Thread(
        target=data_handler.collect_test_data())
    data_collection_thread.start()

    num_locations = 7
    num_days = 3
    Shorthands.brute_force(num_locations, num_days)
    # brute_force_route = Shorthands.brute_force(num_locations, num_days)
    # kmeans_route = Shorthands.k_means(num_locations, num_days)
    # genetic_clustering_route = Shorthands.genetic_clustering(num_locations,
    #                                                          num_days)
    # genetic_centroid_clustering_route = \
    #     Shorthands.genetic_centroid_clustering(num_locations, num_days)

    data_collection_thread.join()


if __name__ == '__main__':
    main()
