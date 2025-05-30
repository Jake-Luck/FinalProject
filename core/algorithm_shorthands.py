"""
Provides Shorthands class to easily run different algorithms with
random/default inputs.
"""
import numpy as np
from time import process_time

from core.data_handling import DataHandling
from core.plotting import Plotting

from algorithms.clustering import (
    KMeans, Clustering, GeneticClustering, GeneticCentroidClustering)
from algorithms.routing import Routing, GeneticRouting

from numpy import ndarray  # For type hints


class Shorthands:
    """
    Provides shorthand methods for running different algorithm with
    random/default inputs. Useful for demonstrating and quickly running
    algorithms.
    """
    @staticmethod
    def brute_force(num_locations: int,
                    num_days: int,
                    graph: ndarray | None = None,
                    durations: ndarray | None = None,
                    coordinates: ndarray | None = None,
                    plot: bool = True) -> ndarray:
        """
        Shorthand for performing brute force routing. Unless **both** graph and
        coordinates are provided, random replacements will be chosen instead.
        :param num_locations: The number of locations in the route.
        :param num_days: The number of days in the route.
        :param graph: The graph input as an adjacency matrix.
        :param durations: Duration spent at each location.
        :param coordinates: Coordinates of each location in the graph.
        :param plot: Whether to plot the final route.
        :return: Returns a 1D ndarray representing the found route.
        """
        graph, coordinates, durations = Shorthands._setup_inputs(
            num_locations, graph, durations, coordinates)

        time_start = process_time()
        route = Routing.brute_force(num_locations, num_days, graph, durations)
        time_end = process_time()
        print(f"Brute Force time: {time_end - time_start} seconds")

        evaluation, std_deviation, evaluation_per_day = Routing.evaluate_route(
            route, num_days, graph, durations)
        title = f"Bruteforce: {evaluation}, σ={std_deviation}"

        centre = coordinates.mean(axis=0)
        if plot:
            Plotting.display_route(route, coordinates, centre, title,
                                   evaluation_per_day, durations)
        return route

    @staticmethod
    def genetic_clustering(
            num_locations: int,
            num_days: int,
            graph: ndarray | None = None,
            durations: ndarray | None = None,
            coordinates: ndarray | None = None,
            num_generations: int = 1000,
            population_size: int = 100,
            crossover_probability: float = 0.9,
            mutation_probability: float = 0.1,
            routing_algorithm: Clustering.RoutingMethods =
                    Clustering.RoutingMethods.GREEDY,
            generations_per_update: int | None = 200,
            plot: bool = True,
            seed: int | None = None,
            plot_stages: bool = False) -> ndarray:
        """
        Shorthand for performing genetic clustering and performing routing
        using those clusters. Unless **both** graph and coordinates are
        provided, random replacements will be chosen instead.
        :param num_locations: The number of locations in the route.
        :param num_days: The number of days in the route.
        :param graph: The graph input as an adjacency matrix.
        :param durations: Duration spent at each location.
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
        :param plot_stages: Whether to plot the clusters.
        :return: Returns a 1D ndarray representing the found route.
        """
        graph, coordinates, durations = Shorthands._setup_inputs(
            num_locations, graph, durations, coordinates)

        genetic_algorithm = GeneticClustering(
            num_generations, population_size, crossover_probability,
            mutation_probability, generations_per_update, plot, seed,
            plot_stages)

        time_start = process_time()
        cluster_assignments = genetic_algorithm.find_clusters(
            graph, durations, num_locations, num_days, routing_algorithm,
            coordinates)
        time_end = process_time()
        print(f"Genetic Clustering time: {time_end - time_start} seconds")

        route = genetic_algorithm.find_route_from_clusters(
            cluster_assignments, num_days, routing_algorithm, graph, durations,
            coordinates)

        evaluation, std_deviation, evaluation_per_day = Routing.evaluate_route(
            route, num_days, graph, durations)

        match routing_algorithm:
            case Clustering.RoutingMethods.GREEDY:
                routing_string = " + Greedy"
            case Clustering.RoutingMethods.BRUTE_FORCE:
                routing_string = " + Brute Force"
            case _:
                routing_string = " + Unknown Routing"
        title = (f"Genetic Clustering{routing_string}: {evaluation}, "
                 f"σ={std_deviation}")

        centre = coordinates.mean(axis=0)

        if plot:
            Plotting.display_route(route, coordinates, centre, title,
                                   evaluation_per_day, durations)
        return route

    @staticmethod
    def genetic_centroid_clustering(
            num_locations: int,
            num_days: int,
            graph: ndarray | None = None,
            durations: ndarray | None = None,
            coordinates: ndarray | None = None,
            num_generations: int = 1000,
            population_size: int = 100,
            crossover_probability: float = 0.9,
            mutation_probability: float = 0.1,
            routing_algorithm: Clustering.RoutingMethods =
                    Clustering.RoutingMethods.GREEDY,
            generations_per_update: int | None = 200,
            plot: bool = True,
            seed: int | None = None,
            plot_stages: bool = False) -> ndarray:
        """
        Shorthand for performing genetic centroid clustering and performing
        routing using those clusters. Unless **both** graph and coordinates are
        provided, random replacements will be chosen instead.
        :param num_locations: The number of locations in the route.
        :param num_days: The number of days in the route.
        :param graph: The graph input as an adjacency matrix.
        :param durations: Duration spent at each location.
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
        :param plot_stages: Whether to plot the clusters.
        :return: Returns a 1D ndarray representing the found route.
        """
        graph, coordinates, durations = Shorthands._setup_inputs(
            num_locations, graph, durations, coordinates)

        genetic_algorithm = GeneticCentroidClustering(
            num_generations, population_size, crossover_probability,
            mutation_probability, generations_per_update, plot, seed,
            plot_stages)

        time_start = process_time()
        cluster_assignments = genetic_algorithm.find_clusters(
            coordinates, graph, durations, num_days, routing_algorithm)
        time_end = process_time()
        print(f"Genetic Centroid time: {time_end - time_start} seconds")

        route = genetic_algorithm.find_route_from_clusters(
            cluster_assignments, num_days, routing_algorithm, graph, durations,
            coordinates)

        evaluation, std_deviation, evaluation_per_day = Routing.evaluate_route(
            route, num_days, graph, durations)

        match routing_algorithm:
            case Clustering.RoutingMethods.GREEDY:
                routing_string = " + Greedy"
            case Clustering.RoutingMethods.BRUTE_FORCE:
                routing_string = " + Brute Force"
            case _:
                routing_string = " + Unknown Routing"
        title = (f"Genetic Centroid Clustering{routing_string}: {evaluation}"
                 f", σ={std_deviation}")

        centre = coordinates.mean(axis=0)
        if plot:
            Plotting.display_route(route, coordinates, centre, title,
                                   evaluation_per_day, durations)
        return route

    @staticmethod
    def genetic_routing(
            num_locations: int,
            num_days: int,
            graph: ndarray | None = None,
            durations: ndarray | None = None,
            coordinates: ndarray | None = None,
            num_generations: int = 1000,
            population_size: int = 100,
            crossover_probability: float = 0.9,
            mutation_probability: float = 0.1,
            generations_per_update: int | None = 200,
            plot: bool = True,
            seed: int | None = None) -> ndarray:
        graph, coordinates, durations = Shorthands._setup_inputs(
            num_locations, graph, durations, coordinates)

        genetic_algorithm = GeneticRouting(
            num_generations, population_size, crossover_probability,
            mutation_probability, generations_per_update, plot, seed)

        time_start = process_time()
        route = genetic_algorithm.find_route(num_locations, num_days,
            graph, durations, coordinates)
        time_end = process_time()
        print(f"Genetic routing time: {time_end - time_start} seconds")

        evaluation, std_deviation, evaluation_per_day = Routing.evaluate_route(
            route, num_days, graph, durations)

        title = f"Genetic Routing: {evaluation}, σ={std_deviation}"

        centre = coordinates.mean(axis=0)
        if plot:
            Plotting.display_route(route, coordinates, centre, title,
                                   evaluation_per_day, durations)
        return route

    @staticmethod
    def gift_wrapping(num_locations: int,
                      num_days: int,
                      graph: ndarray | None = None,
                      durations: ndarray | None = None,
                      coordinates: ndarray | None = None,
                      plot: bool = True) -> ndarray:
        """
        Shorthand for performing gift wrapping routing. Unless **both** graph
        and coordinates are provided, random replacements will be chosen
        instead.
        :param num_locations: The number of locations in the route.
        :param num_days: The number of days in the route.
        :param graph: The graph input as an adjacency matrix.
        :param durations: Duration spent at each location.
        :param coordinates: Coordinates of each location in the graph.
        :param plot: Whether to plot the final route.
        :return: Returns a 1D ndarray representing the found route.
        """
        graph, coordinates, durations = Shorthands._setup_inputs(
            num_locations, graph, durations, coordinates)

        time_start = process_time()
        route = Routing.gift_wrapping(
            num_locations, num_days, coordinates, graph, durations)
        time_end = process_time()
        print(f"Giftwrapping time: {time_end - time_start} seconds")


        evaluation, std_deviation, evaluation_per_day = Routing.evaluate_route(
            route, num_days, graph, durations)
        title = f"Gift Wrapping: {evaluation}, σ={std_deviation}"

        centre = coordinates.mean(axis=0)
        if plot:
            Plotting.display_route(route, coordinates, centre, title,
                                   evaluation_per_day, durations)
        return route

    @staticmethod
    def greedy(num_locations: int,
               num_days: int,
               graph: ndarray | None = None,
               durations: ndarray | None = None,
               coordinates: ndarray | None = None,
               plot: bool = True) -> ndarray:
        """
        Shorthand for performing greedy routing. Unless **both** graph and
        coordinates are provided, random replacements will be chosen instead.
        :param num_locations: The number of locations in the route.
        :param num_days: The number of days in the route.
        :param graph: The graph input as an adjacency matrix.
        :param durations: Duration spent at each location.
        :param coordinates: Coordinates of each location in the graph.
        :param plot: Whether to plot the final route.
        :return: Returns a 1D ndarray representing the found route.
        """
        graph, coordinates, durations = Shorthands._setup_inputs(
            num_locations, graph, durations, coordinates)

        time_start = process_time()
        route = Routing.greedy_routing(
            num_locations, num_days, graph, durations)
        time_end = process_time()
        print(f"Greedy routing time: {time_end - time_start} seconds")

        evaluation, std_deviation, evaluation_per_day = Routing.evaluate_route(
            route, num_days, graph, durations)
        title = f"Greedy: {evaluation}, σ={std_deviation}"

        centre = coordinates.mean(axis=0)
        if plot:
            Plotting.display_route(route, coordinates, centre, title,
                                   evaluation_per_day, durations)
        return route

    @staticmethod
    def greedy_insertion(num_locations: int,
                         num_days: int,
                         graph: ndarray | None = None,
                         durations: ndarray | None = None,
                         coordinates: ndarray | None = None,
                         plot: bool = True) -> ndarray:
        graph, coordinates, durations = Shorthands._setup_inputs(
            num_locations, graph, durations, coordinates)

        location_set = list()
        for _ in range(num_days):  # -1 because 0 added to end of routes
            location_set.append(0)
        for i in range(1, num_locations):
            location_set.append(i)
        time_start = process_time()
        route = Routing.greedy_insertion(np.array([]), np.array(location_set), graph, durations)
        time_end = process_time()
        print(f"Greedy insertion time: {time_end - time_start} seconds")

        evaluation, std_deviation, evaluation_per_day = Routing.evaluate_route(
            route, num_days, graph, durations)
        title = f"Greedy Insertion: {evaluation}, σ={std_deviation}"

        centre = coordinates.mean(axis=0)
        if plot:
            Plotting.display_route(route, coordinates, centre, title,
                                   evaluation_per_day, durations)
        return route

    @staticmethod
    def k_means(num_locations: int,
                num_days: int,
                graph: ndarray | None = None,
                durations: ndarray | None = None,
                coordinates: ndarray | None = None,
                routing_algorithm: Clustering.RoutingMethods =
                        Clustering.RoutingMethods.GREEDY,
                plot: bool = True,
                show_stages: bool = False,
                seed: int | None = None) -> ndarray:
        """
        Shorthand for performing k-means clustering and performing routing
        using those clusters. Unless **both** graph and coordinates are
        provided, random replacements will be chosen instead.
        :param num_locations: The number of locations in the route.
        :param num_days: The number of days in the route.
        :param graph: The graph input as an adjacency matrix.
        :param durations: Duration spent at each location.
        :param coordinates: Coordinates of each location in the graph.
        :param routing_algorithm: The routing algorithm to use with clustering.
        :param plot: Whether to plot each k-means step and the final route.
        :param show_stages: Whether to show each kmeans stages.
        :return: Returns a 1D ndarray representing the found route.
        """
        graph, coordinates, durations = Shorthands._setup_inputs(
            num_locations, graph, durations, coordinates)

        kmeans = KMeans(show_stages=show_stages, random_seed=seed)

        time_start = process_time()
        cluster_assignments = kmeans.find_clusters(coordinates, num_days,
                                                   num_locations)
        time_end = process_time()
        print(f"K-Means time: {time_end - time_start} seconds")

        route = kmeans.find_route_from_clusters(
            cluster_assignments, num_days, routing_algorithm, graph, durations,
            coordinates)

        evaluation, std_deviation, evaluation_per_day = Routing.evaluate_route(
            route, num_days, graph, durations)

        match routing_algorithm:
            case Clustering.RoutingMethods.GREEDY:
                routing_string = " + Greedy"
            case Clustering.RoutingMethods.BRUTE_FORCE:
                routing_string = " + Brute Force"
            case _:
                routing_string = " + Unknown Routing"
        title = f"K-Means{routing_string}: {evaluation}, σ={std_deviation}"

        centre = coordinates.mean(axis=0)

        if plot:
            Plotting.display_route(route, coordinates, centre, title,
                                   evaluation_per_day, durations)
        else:
            print(title)
        return route

    @staticmethod
    def _setup_inputs(num_locations: int,
                      graph: ndarray | None = None,
                      durations: ndarray | None = None,
                      coordinates: ndarray | None = None) -> tuple:
        """
        Sets up our inputs so that if either our graph or coordinates aren't
        provided, random ones will be chosen.
        :param graph: The graph input to use if given. If not, a randomly
        chosen graph & coordinates will be used.
        :param durations: Duration spent at each location. If not given, random
        durations will be generated.
        :param coordinates: The coordinates input to use if given. If not, a
        randomly chosen graph & coordinates will be used.
        """
        # If graph provided, but no coordinates, cannot plot
        if durations is None:
            durations = np.random.randint(1, 96, num_locations) * 15
            durations[0] = 0

        if graph is not None and coordinates is not None:
            return graph[:num_locations], coordinates[:num_locations], durations

        # Chooses a random graph from saved data.
        return DataHandling.get_random_datum()
