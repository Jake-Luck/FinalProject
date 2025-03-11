from core.data_handling import DataAttributes, DataGroups
from core.plotting import Plotting

from algorithms.routing import Routing
from algorithms.clustering import KMeans, Clustering
from algorithms.genetic import GeneticClustering, GeneticCentroidClustering

import h5py
import random

import numpy as np
from numpy import ndarray  # For type hints

class Shorthands:
    """Provides shorthand methods for running different algorithm with
    random/default inputs. Useful for demonstrating and quickly checking something."""
    @staticmethod
    def _setup_inputs(graph: ndarray | None = None,
                      coordinates: ndarray | None = None,
                      plot: bool = True) -> None:
        # If graph provided, but no coordinates, cannot plot
        if graph is not None:
            if coordinates is None:
                plot = False
            return

        """Chooses a random graph from saved data."""
        with h5py.File('data/training_data.h5', 'r') as f:
            graphs = f[DataGroups.graphs]
            chosen_index = random.randrange(len(graphs))
            graph_data = graphs[chosen_index]
            graph = np.array(graph_data, copy=True, dtype=float)
            coordinates = np.array(
                graph_data.attrs[DataAttributes.coordinates],
                copy=True, dtype=float)

    @staticmethod
    def brute_force(num_locations: int,
                    num_days: int,
                    graph: ndarray | None = None,
                    coordinates: ndarray | None = None,
                    plot: bool = True):
        Shorthands._setup_inputs(graph, coordinates, plot)

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
            routing_algorithm: Clustering.RoutingMethods = \
                    Clustering.RoutingMethods.BRUTE_FORCE,
            generations_per_update: int | None = 200,
            plot: bool = True,
            seed: int | None = None,):
        Shorthands._setup_inputs(graph, coordinates, plot)
        route_length = num_locations+num_days-1

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
            routing_algorithm: Clustering.RoutingMethods = \
                    Clustering.RoutingMethods.BRUTE_FORCE,
            generations_per_update: int | None = 200,
            plot: bool = True,
            seed: int | None = None,):
        Shorthands._setup_inputs(graph, coordinates, plot)
        route_length = num_locations+num_days-1

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
                routing_algorithm: Clustering.RoutingMethods = \
                        Clustering.RoutingMethods.BRUTE_FORCE,
                plot: bool = True,):
        Shorthands._setup_inputs(graph, coordinates, plot)

        kmeans = KMeans(show_stages=True)
        cluster_assignments = kmeans.find_clusters(coordinates, num_days,
                                                   num_locations)
        route = kmeans.find_route_from_cluster_assignments(
            cluster_assignments, num_days, routing_algorithm, graph)

        if plot:
            Plotting.display_route(route, coordinates)
        return route
