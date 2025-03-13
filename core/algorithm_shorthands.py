"""
Provides Shorthands class to easily run different algorithms with
random/default inputs.
"""
from core.data_handling import DataHandling
from core.plotting import Plotting

from algorithms.clustering import KMeans, Clustering
from algorithms.genetic import GeneticClustering, GeneticCentroidClustering
from algorithms.routing import Routing

import h5py
import numpy as np
from numpy import ndarray  # For type hints
import random


# noinspection PyPep8
class Shorthands:
    """
    Provides shorthand methods for running different algorithm with
    random/default inputs. Useful for demonstrating and quickly running
    algorithms.
    """

    @staticmethod
    def _setup_inputs(num_locations: int,
                      graph: ndarray | None = None,
                      coordinates: ndarray | None = None) -> tuple:
        """
        Sets up our inputs so that if either our graph or coordinates aren't
        provided, random ones will be chosen.
        :param graph: The graph input to use if given. If not, a randomly
        chosen graph & coordinates will be used.
        :param coordinates: The coordinates input to use if given. If not, a
        randomly chosen graph & coordinates will be used.
        """
        # If graph provided, but no coordinates, cannot plot
        if graph is not None and coordinates is not None:
            return graph[:num_locations], coordinates[:num_locations]

        # Chooses a random graph from saved data.
        return DataHandling.get_random_datum()

    @staticmethod
    def brute_force(num_locations: int,
                    num_days: int,
                    graph: ndarray | None = None,
                    coordinates: ndarray | None = None,
                    plot: bool = True) -> ndarray:
        """
        Shorthand for performing brute force routing. Unless **both** graph and
        coordinates are provided, random replacements will be chosen instead.
        :param num_locations: The number of locations in the route.
        :param num_days: The number of days in the route.
        :param graph: The graph input as an adjacency matrix.
        :param coordinates: Coordinates of each location in the graph.
        :param plot: Whether to plot the final route.
        :return: Returns a 1D ndarray representing the found route.
        """
        graph, coordinates = Shorthands._setup_inputs(num_locations, graph,
                                                      coordinates)

        route = Routing.brute_force(num_locations, num_days, graph)

        if plot:
            Plotting.display_route(route, coordinates)
        return route

    @staticmethod
    def genetic_clustering(
            num_locations: int,
            num_days: int,
            graph: ndarray | None = None,
            coordinates: ndarray | None = None,
            num_generations: int = 1000,
            population_size: int = 100,
            crossover_probability: float = 0.9,
            mutation_probability: float = 0.1,
            routing_algorithm: Clustering.RoutingMethods =
                    Clustering.RoutingMethods.GREEDY,
            generations_per_update: int | None = 200,
            plot: bool = True,
            seed: int | None = None) -> ndarray:
        """
        Shorthand for performing genetic clustering and performing routing
        using those clusters. Unless **both** graph and coordinates are
        provided, random replacements will be chosen instead.
        :param num_locations: The number of locations in the route.
        :param num_days: The number of days in the route.
        :param graph: The graph input as an adjacency matrix.
        :param coordinates: Coordinates of each location in the graph.
        :param num_generations: Number of generations to run.
        :param population_size: Number of individuals in each population.
        :param crossover_probability: Probability of crossover (0-1).
        :param mutation_probability: Probability of mutation (0-1).
        :param routing_algorithm: The routing algorithm to use with clustering.
        :param generations_per_update: How many generations to run between each
        update.
        :param plot: Whether to plot on each update and the final route.
        :param seed: Specified seed for random number generators.
        :return: Returns a 1D ndarray representing the found route.
        """
        graph, coordinates = Shorthands._setup_inputs(num_locations, graph,
                                                      coordinates)
        route_length = num_locations + num_days - 1

        genetic_algorithm = GeneticClustering(
            num_generations, population_size, crossover_probability,
            mutation_probability, generations_per_update, plot, seed)
        cluster_assignments = genetic_algorithm.find_clusters(
            graph, num_locations, num_days, route_length, routing_algorithm,
            coordinates)

        route = genetic_algorithm.find_route_from_cluster_assignments(
            cluster_assignments, num_days, routing_algorithm, graph)

        if plot:
            Plotting.display_route(route, coordinates)
        return route

    @staticmethod
    def genetic_centroid_clustering(
            num_locations: int,
            num_days: int,
            graph: ndarray | None = None,
            coordinates: ndarray | None = None,
            num_generations: int = 1000,
            population_size: int = 100,
            crossover_probability: float = 0.9,
            mutation_probability: float = 0.1,
            routing_algorithm: Clustering.RoutingMethods =
                    Clustering.RoutingMethods.GREEDY,
            generations_per_update: int | None = 200,
            plot: bool = True,
            seed: int | None = None) -> ndarray:
        """
        Shorthand for performing genetic centroid clustering and performing
        routing using those clusters. Unless **both** graph and coordinates are
        provided, random replacements will be chosen instead.
        :param num_locations: The number of locations in the route.
        :param num_days: The number of days in the route.
        :param graph: The graph input as an adjacency matrix.
        :param coordinates: Coordinates of each location in the graph.
        :param num_generations: Number of generations to run.
        :param population_size: Number of individuals in each population.
        :param crossover_probability: Probability of crossover (0-1).
        :param mutation_probability: Probability of mutation (0-1).
        :param routing_algorithm: The routing algorithm to use with clustering.
        :param generations_per_update: How many generations to run between each
        update.
        :param plot: Whether to plot on each update and the final route.
        :param seed: Specified seed for random number generators.
        :return: Returns a 1D ndarray representing the found route.
        """
        graph, coordinates = Shorthands._setup_inputs(num_locations, graph,
                                                      coordinates)

        genetic_algorithm = GeneticCentroidClustering(
            num_generations, population_size, crossover_probability,
            mutation_probability, generations_per_update, plot, seed)
        cluster_assignments = genetic_algorithm.find_clusters(
            coordinates, graph, num_days, routing_algorithm)

        route = genetic_algorithm.find_route_from_cluster_assignments(
            cluster_assignments, num_days, routing_algorithm, graph)

        if plot:
            Plotting.display_route(route, coordinates)
        return route

    @staticmethod
    def k_means(num_locations: int,
                num_days: int,
                graph: ndarray | None = None,
                coordinates: ndarray | None = None,
                routing_algorithm: Clustering.RoutingMethods =
                        Clustering.RoutingMethods.GREEDY,
                plot: bool = True) -> ndarray:
        """
        Shorthand for performing k-means clustering and performing routing
        using those clusters. Unless **both** graph and coordinates are
        provided, random replacements will be chosen instead.
        :param num_locations: The number of locations in the route.
        :param num_days: The number of days in the route.
        :param graph: The graph input as an adjacency matrix.
        :param coordinates: Coordinates of each location in the graph.
        :param routing_algorithm: The routing algorithm to use with clustering.
        :param plot: Whether to plot each k-means step and the final route.
        :return: Returns a 1D ndarray representing the found route.
        """
        graph, coordinates = Shorthands._setup_inputs(num_locations, graph,
                                                      coordinates)

        kmeans = KMeans(show_stages=True)
        cluster_assignments = kmeans.find_clusters(coordinates, num_days,
                                                   num_locations)
        route = kmeans.find_route_from_cluster_assignments(
            cluster_assignments, num_days, routing_algorithm, graph)

        if plot:
            Plotting.display_route(route, coordinates)
        return route
