"""
Provides Clustering base class and most clustering classes.
"""
from algorithms.algorithm import Algorithm
from algorithms.routing import Routing
from core.plotting import Plotting

from enum import Enum
import numpy as np
from numpy import ndarray  # For type hints


class Clustering(Algorithm):
    """
    Clustering algorithm base class, provides routing methods enum as well as
    methods for assigning nodes to centroids and finding routes rome a given
    cluster assignment.
    """
    class RoutingMethods(Enum):
        """
        These are travelling salesmen solvers for use with clustering.
        """
        # Brute force needs lambda for num_days (it has num_days as a
        # parameter) because it can be used without clustering too.
        GREEDY = 0
        'Always chooses the shortest path from any given position.'
        BRUTE_FORCE = 1
        'Compares every possible route to find the best.'

    @staticmethod
    def _assign_nodes_to_centroid(coordinates: ndarray,
                                  centroids: ndarray) -> ndarray:
        """
        Assigns each coordinate a cluster by computing distance from each
        coordinate to each centroid and choosing the smallest distance.
        :param coordinates: Coordinates of each location.
        :param centroids: Coordinates of each cluster's centroid. 2d array of
        shape (num_days, 2)
        :return: Each location's cluster assignment. A 1D array of shape
        (num_locations).
        """
        # Gets a matrix of distances from each location to each centroid
        distances = np.linalg.norm(
            coordinates[:, np.newaxis, :2] - centroids[:, :2], axis=2)

        # For each location (index in distance matrix) gets the index for the
        # centroid with the smallest distance
        clusters = np.argmin(distances, axis=1)
        return clusters

    @staticmethod
    def find_route_from_clusters(cluster_assignments: ndarray,
                                 num_days: int,
                                 routing_method: RoutingMethods,
                                 graph: ndarray,
                                 durations: ndarray) -> ndarray:
        """
        Finds a route from the given cluster assignment using the given routing
        method on each cluster and stitching the routes together.
        :param cluster_assignments: Each location's cluster assignment. A 1D
        array of shape (num_locations).
        :param num_days: Number of days in the route.
        :param routing_method: The routing method to use on each cluster.
        :param graph: The graph input as an adjacency matrix.
        :param durations: Duration spent at each location.
        :return: 1D ndarray representing the found route.
        """
        clusters = list[ndarray]()
        for i in range(num_days):
            indexes_in_cluster = np.where(cluster_assignments == i)[0] + 1
            clusters.append(np.concatenate(([0], indexes_in_cluster)))

        # numpy magic
        # np.ix_([1,2,3], [1,2,3]) returns [[[1],[2],[3]],[1,2,3]]
        # Which can they be used to access graphs
        graphs = [graph[np.ix_(indexes, indexes)] for indexes in clusters]
        durations_each_day = [durations[indexes] for indexes in clusters]

        route = np.empty(0, dtype=int)

        routing = Routing()
        match routing_method:
            case Clustering.RoutingMethods.BRUTE_FORCE:
                routing_function = routing.brute_force
            case Clustering.RoutingMethods.GREEDY:
                routing_function = routing.greedy_routing
            case _:
                print("Invalid routing method, defaulting to greedy")
                routing_function = routing.greedy_routing

        for sub_graph, cluster, sub_durations in zip(graphs, clusters,
                                                     durations_each_day):
            n = sub_graph.shape[0]
            sub_route = routing_function(n, 1, sub_graph,
                                         sub_durations)

            route = np.concatenate((route, cluster[sub_route]))
        return route


class KMeans(Clustering):
    """
    Class for K-Means clustering.
    """
    def __init__(self,
                 show_stages: bool = False,
                 maximum_iterations: int = 100):
        """
        Initialises k-means class with given parameter.
        :param show_stages: Whether to plot each k-means stage.
        """
        self.show_stages = show_stages
        self.maximum_iterations = maximum_iterations

    @staticmethod
    def _compute_means(coordinates: ndarray,
                       num_days: int) -> ndarray:
        """
        Computes the mean coordinate of each cluster.
        :param coordinates: Coordinates of each cluster, a 2D array with shape
        (num_coordinates, 3). Second dimension is (x, y, assigned_cluster)
        :param num_days: The number of clusters/means to compute.
        :return: Returns a list of means, 1D array.
        """
        computed_means = np.empty((num_days, 2))

        for i in range(num_days):
            cluster = coordinates[coordinates[:, 2] == i, :2]
            computed_means[i] = cluster.mean(axis=0)
        return computed_means

    def find_clusters(self,
                      coordinates: ndarray,
                      num_days: int,
                      num_locations: int):
        """
        Sorts n coordinates into k clusters via a K-Means approach.
        :param coordinates: Coordinates of each location.
        :param num_days: Number of clusters.
        :param num_locations: Number of locations.
        :return: Each location's cluster assignment. A 1D array of shape
        (num_locations).
        """
        # Initialises cluster assignments to 0
        cluster_assignments = previous_clusters = np.zeros(num_locations)

        # Adds third axis to coordinates. Used to denote cluster assignment.
        # Excludes starting point, which does not need clustering.
        coordinates = np.append(coordinates[1:],
                                np.zeros((num_locations - 1, 1)),
                                axis=1)

        # Initialises means to random unique coordinates
        chosen_indices = np.random.choice(coordinates.shape[0], num_days,
                                          replace=False)
        means = coordinates[chosen_indices]

        for _ in range(self.maximum_iterations):
            cluster_assignments = self._assign_nodes_to_centroid(coordinates,
                                                                 means)
            coordinates[:, 2] = cluster_assignments

            # Delete this for screenshots
            if self.show_stages:
                centre = coordinates.mean(axis=0)
                Plotting.display_clusters(coordinates, cluster_assignments,
                                          num_days, means, centre)

            if np.array_equal(cluster_assignments, previous_clusters):
                break
            previous_clusters = np.array(cluster_assignments, copy=True)

            means = self._compute_means(coordinates, num_days)

        return cluster_assignments
