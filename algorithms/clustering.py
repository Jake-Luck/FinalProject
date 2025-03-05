from typing import Callable, Any

from sympy import false

from algorithms.utilities import ClusteringMethods, RoutingMethods, \
    find_route_from_cluster_assignments
from algorithms.brute_force import brute_force
from algorithms.greedy import greedy

from core.utilities import display_clusters

import numpy as np
from numpy import ndarray  # For type hints


def k_means(coordinates: ndarray,
            k: int,
            n: int,
            show_stages: bool = False) -> ndarray:
    """
    Sorts n coordinates into k clusters.

    :param coordinates: Coordinates of each location.
    :param k: Number of clusters.
    :param n: Number of locations.
    :return: A 1D array of shape (n). Represents the chosen clusters.
    """
    def assign_clusters(_coordinates: ndarray,
                        _means: ndarray) -> ndarray:
        """
        Assigns each coordinate a cluster by computing distance from each
        coordinate to each mean and choosing the smallest distance.

        :param _coordinates: Coordinates of each location.
        :param _means: Coordinates of each cluster's mean.
        :return: A 1D array of shape (n). Represents the chosen clusters.
        """
        distances = np.linalg.norm(
            _coordinates[:, np.newaxis, :2] - _means[:, :2], axis=2)
        clusters = np.argmin(distances, axis=1)
        _coordinates[:, 2] = clusters
        return clusters

    def compute_means(_coordinates: ndarray,
                      _k: int) -> ndarray:
        """
        Computes the mean coordinate of each cluster.
        :param _coordinates: Coordinates of each cluster, a 2D array with shape
        (num_coordinates, 3). Second dimension is (x, y, assigned_cluster)
        :param _k: The number of clusters/means to compute.
        :return: Returns a list of means, 1D array.
        """
        computed_means = np.empty((_k, 2))

        for i in range(_k):
            _cluster_assignments = _coordinates[_coordinates[:, 2] == i, :2]
            computed_means[i] = _cluster_assignments.mean(axis=0)
        return computed_means

    means = np.array(coordinates[:k], copy=True)
    coordinates = np.append(coordinates, np.zeros((n, 1)), axis=1)

    previous_clusters = np.zeros(n)
    while True:
        cluster_assignments = assign_clusters(coordinates, means)

        if show_stages:
            display_clusters(coordinates, cluster_assignments, k, means)

        if (cluster_assignments == previous_clusters).all():
            break
        previous_clusters = np.array(cluster_assignments, copy=True)

        means = compute_means(coordinates, k)

    return cluster_assignments


def cluster_and_solve(coordinates: ndarray,
                      graph: ndarray,
                      num_days: int,
                      clustering_method: ClusteringMethods,
                      routing_method: RoutingMethods,
                      show_stages: bool = False) -> ndarray:
    """
    Sorts coordinates into clusters using specified clustering method, then
    creates a route by applying a travelling salesman problem algorithm to each
    cluster. Each cluster's route is added together to form the final route.
    :param coordinates: Coordinates of each location.
    :param graph: Graph represented as an adjacency matrix.
    :param num_days: Number of days in the route (number of clusters).
    :param clustering_method: Clustering method to use.
    :param routing_method: Routing method to use.
    :param show_stages: Whether to plot clusters each step
    :return: Returns a 1D np array representing the route found.
    """
    centre = np.array(coordinates[0], copy=True)

    cluster_assignments = ndarray
    match clustering_method:
        case ClusteringMethods.K_MEANS:
            cluster_coordinates = np.array(coordinates[1:], copy=True)
            num_locations = cluster_coordinates.shape[0]
            cluster_assignments = k_means(cluster_coordinates, num_days,
                                          num_locations)

    clusters = list[ndarray]()
    route = find_route_from_cluster_assignments(cluster_assignments, num_days,
                                                routing_method, graph)

    return route
