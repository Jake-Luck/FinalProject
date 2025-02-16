from typing import Callable, Any

from algorithms.utilities import ClusteringMethods, RoutingMethods
from algorithms.brute_force import brute_force
from algorithms.greedy import greedy

import numpy as np
from numpy import ndarray  # For type hints


def k_means(coordinates: ndarray,
            k: int,
            n: int) -> ndarray:
    """
    Sorts n coordinates into k clusters.

    :param coordinates: Coordinates of each location.
    :param k: Number of clusters.
    :param n: Number of locations.
    :return: A 1D array of shape (n). Represents the chosen clusters.
    """

    def assign_clusters(coordinates: ndarray,
                        means: ndarray,
                        k: int,
                        n: int) -> ndarray:
        """
        Assigns each coordinate a cluster by computing distance from each
        coordinate to each mean and choosing the smallest distance.

        :param coordinates: Coordinates of each location.
        :param means: Coordinates of each cluster's mean.
        :param k: Number of clusters.
        :param n: Number of locations.
        :return: A 1D array of shape (n). Represents the chosen clusters.
        """
        distances = np.linalg.norm(
            coordinates[:, np.newaxis, :2] - means[:, :2], axis=2)
        clusters = np.argmin(distances, axis=1)
        coordinates[:, 2] = clusters
        return clusters

    def compute_means(coordinates: ndarray, k: int) -> ndarray:
        computed_means = np.empty((k, 2))

        for i in range(k):
            cluster_coordinates = coordinates[coordinates[:, 2] == i, :2]
            computed_means[i] = cluster_coordinates.mean(axis=0)
        return computed_means

    means = np.array(coordinates[:k], copy=True)
    coordinates = np.append(coordinates, np.zeros((n, 1)), axis=1)

    previous_clusters = np.zeros(n)
    while True:
        cluster_assignments = assign_clusters(coordinates, means, k, n)

        if (cluster_assignments == previous_clusters).all():
            break
        previous_clusters = np.array(cluster_assignments, copy=True)

        means = compute_means(coordinates, k)

    return cluster_assignments


def cluster_and_solve(coordinates: ndarray,
                      graph: ndarray,
                      num_days: int,
                      clustering_method: ClusteringMethods,
                      routing_method: RoutingMethods) -> ndarray:
    """
    Sorts coordinates into clusters using specified clustering method, then
    creates a route by applying a travelling salesman problem algorithm to each
    cluster. Each cluster's route is added together to form the final route.
    :param coordinates: Coordinates of each location.
    :param graph: Graph represented as an adjacency matrix.
    :param num_days: Number of days in the route (number of clusters).
    :param clustering_method: Clustering method to use.
    :param routing_method: Routing method to use.
    :return:
    """
    centre = np.array(coordinates[0], copy=True)

    clusters = list[ndarray]
    match clustering_method:
        case ClusteringMethods.K_MEANS:
            cluster_coordinates = np.array(coordinates[1:], copy=True)
            num_locations = cluster_coordinates.shape[0]
            clusters = k_means(cluster_coordinates, num_days, num_locations)

    cluster_indexes = list[ndarray]()
    for i in range(num_days):
        indexes_in_cluster = np.where(clusters == i)[0] + 1
        cluster_indexes.append(np.concatenate(([0], indexes_in_cluster)))

    # numpy magic
    # np.ix_([1,2,3], [1,2,3]) returns [[[1],[2],[3]],[1,2,3]]
    # Which can they be used to access graphs
    graphs = [graph[np.ix_(indexes, indexes)] for indexes in cluster_indexes]

    routing_function: Callable[[Any, Any], ndarray]
    route = np.empty(0, dtype=int)
    match routing_method:
        case RoutingMethods.GREEDY:
            routing_function = greedy
        case _:
            routing_function = lambda n, g: brute_force(n, 1, g)
    for sub_graph, cluster_index in zip(graphs, cluster_indexes):
        n = sub_graph.shape[0]
        sub_route = routing_function(n, sub_graph)

        route = np.concatenate((route, cluster_index[sub_route]))
    return route
