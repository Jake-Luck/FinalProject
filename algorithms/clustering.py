"""
Provides Clustering base class and most clustering classes.
"""
from algorithms.algorithm import Algorithm
from algorithms.genetic import Genetic
from algorithms.routing import Routing
from algorithms.routing import GeneticRouting
from core.plotting import Plotting

from enum import Enum
import numpy as np
from numpy import ndarray  # For type hints
import random


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
        GENETIC = 2
        'Uses a genetic algorithm to find the best route.'
        CONVEX_HULL = 3
        'Uses the convex hull of the points to find the best route.'

    @staticmethod
    def find_route_from_clusters(cluster_assignments: ndarray,
                                 num_days: int,
                                 routing_method: RoutingMethods,
                                 graph: ndarray,
                                 durations: ndarray,
                                 coordinates: ndarray) -> ndarray:
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
        # uses a list of each locations cluster assignment to form a list of
        # the index for each cluster's locations.
        clusters = list[ndarray]()
        for i in range(num_days):
            indexes_in_cluster = np.where(cluster_assignments == i)[0] + 1
            clusters.append(np.concatenate(([0], indexes_in_cluster)))

        routing = Routing()
        match routing_method:
            case Clustering.RoutingMethods.BRUTE_FORCE:
                routing_function = routing.brute_force
            case Clustering.RoutingMethods.GREEDY:
                routing_function = routing.greedy_routing
            case Clustering.RoutingMethods.GENETIC:
                genetic_routing = GeneticRouting(50, 20, 0.9, 0.1, None, False)
                routing_function = genetic_routing.find_route
            case Clustering.RoutingMethods.CONVEX_HULL:
                routing_function = routing.gift_wrapping
            case _:
                print("Invalid routing method, defaulting to greedy")
                routing_function = routing.greedy_routing

        # numpy magic
        # np.ix_([1,2,3], [1,2,3]) returns [[[1],[2],[3]],[1,2,3]]
        # Which can then be used to easily form subgraphs for each cluster.
        graphs = [graph[np.ix_(indexes, indexes)] for indexes in clusters]

        coordinates_each_day = [coordinates[indexes] for indexes in clusters]
        durations_each_day = [durations[indexes] for indexes in clusters]

        route = np.empty(0, dtype=int)
        for sub_graph, cluster, sub_coordinates, sub_durations in \
                zip(graphs, clusters, coordinates_each_day, durations_each_day):
            num_locations = sub_graph.shape[0]
            num_days = 1
            if routing_method == Clustering.RoutingMethods.CONVEX_HULL:
                sub_route = routing_function(
                    num_locations, num_days, sub_coordinates, sub_graph,
                    sub_durations)
            else:
                sub_route = routing_function(num_locations, num_days, sub_graph,
                                             sub_durations)

            route = np.concatenate((route, cluster[sub_route]))
        return route

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


class GeneticClustering(Genetic, Clustering):
    """
    Class for genetic clustering. Genome contains the assignment of each
    location to a cluster.
    """
    def __init__(self,
                 num_generations: int,
                 population_size: int,
                 crossover_probability: float,
                 mutation_probability: float,
                 generations_per_update: int | None = 1,
                 plotting: bool = True,
                 random_seed: int | None = None,
                 plot_clusters: bool = True):
        """
        Initialises genetic clustering with given parameters.
        :param num_generations: Number of generations to run.
        :param population_size: Number of individuals in each population.
        :param crossover_probability: Probability of crossover (0-1).
        :param mutation_probability: Probability of mutation (0-1).
        :param generations_per_update: Number of generations between each
        progress update. If None or less than 0, no updates given.
        :param plotting: Whether to display plots on each update.
        :param random_seed: Specified seed for random number generators.
        """
        super().__init__(num_generations, population_size,
                         crossover_probability, mutation_probability,
                         generations_per_update, plotting, random_seed)
        self.plot_stages = plot_clusters

    def find_clusters(self,
                      graph: ndarray,
                      durations: ndarray,
                      num_locations: int,
                      num_days: int,
                      routing_method: Clustering.RoutingMethods,
                      coordinates: ndarray | None = None) -> ndarray:
        """
        Sorts `num_locations` coordinates into `num_days` clusters via a
        genetic algorithm approach. Genome contains the assignment of each
        location to a cluster.
        :param graph: The graph input as an adjacency matrix.
        :param durations: Duration spent at each location.
        :param num_locations: The number of locations in the route.
        :param num_days: The number of days in the route.
        :param routing_method: The routing method to use on each cluster.
        :param coordinates: Coordinates of each location.
        :return: Each location's cluster assignment. A 1D array of shape
        (num_locations).
        """
        if coordinates is None and self.plotting is True:
            print("No coordinates provided, setting plotting to False")
            self.plotting = False

        evaluations = np.empty(self.population_size)
        evaluations[:] = float('inf')

        # Assign random clusters to each location
        population = np.random.randint(
            num_days, size=(self.population_size, num_locations - 1))

        # Assign these before loop just in case num_generations is 0 and these
        # are used before initialisation
        index1 = 0
        parent1 = population[index1]

        if self.plotting:
            best_evaluation_per_generation = np.empty(self.num_generations)

        for generation_number in range(self.num_generations):
            evaluations = self._evaluate_population(population, num_days,
                                                    routing_method, graph,
                                                    durations, coordinates)

            # get the indices of the lowest (and best) evaluations
            index1, index2 = np.argpartition(evaluations, 2)[:2]
            parent1 = population[index1]
            parent2 = population[index2]

            if self.generations_per_update is not None \
                    and generation_number % self.generations_per_update == 0:
                progress = 100 * generation_number/self.num_generations
                print(f"Cluster evolution {int(progress)}% complete: "
                      f"{generation_number}/{self.num_generations} completed. "
                      f"Best evaluation: {evaluations[index1]}")

                if self.plot_stages:
                    route = self.find_route_from_clusters(
                        population[0], num_days, routing_method, graph,
                        durations)
                    Plotting.display_route(route, coordinates,
                                           title=f"Genetic Clustering, best "
                                                 f"route from generation "
                                                 f"{generation_number}",
                                           durations=durations,)

            if float('inf') in (evaluations[index1], evaluations[index2]):
                population = self._generate_random_centroids(num_days, centre)
                if evaluations[index1] != float('inf'):
                    population[0] = parent1
                if evaluations[index2] != float('inf'):
                    population[0] = parent2
                continue

            # Maintain best two individuals across generations.
            population[0] = parent1
            population[1] = parent2

            # Generate rest of new generation
            for i in range(2, self.population_size):
                use_crossover = random.random() < self.crossover_probability

                if not use_crossover:
                    # Generate random individual (increases genetic diversity)
                    population[i] = np.random.randint(num_days,
                                                      size=num_locations - 1)
                    continue

                population[i] = self._crossover(parent1, parent2)

                for j in range(population[i].shape[0]):
                    mutate = random.random() < self.mutation_probability
                    if mutate:
                        population[i, j] = random.randrange(num_days)

            if self.plotting:
                best_evaluation_per_generation[generation_number] = \
                    evaluations[index1]

        print(f"Cluster evolution complete. "
              f"Best evaluation: {evaluations[index1]}")
        if self.plotting:
            x_axis = np.arange(1, self.num_generations + 1, dtype=float)
            Plotting.plot_line_graph(x_axis, best_evaluation_per_generation,
                                     "Best evaluation",
                                     "Generation number",
                                     "Best evaluation per generation",)

        return parent1

    def _crossover(self,
                   parent1: ndarray,
                   parent2: ndarray) -> ndarray:
        """
        Performs crossover between two given individuals. For each location,
        chooses a cluster from either parent.
        :param parent1: First parent's genome.
        :param parent2: Second parent's genome.
        :return: Offspring of each parent.
        """
        parent1 = self._relabel_individuals_clusters(parent1)
        parent2 = self._relabel_individuals_clusters(parent2)

        crossover_mask = np.random.randint(0, 2, size=len(parent1))
        offspring = np.where(crossover_mask == 0, parent1, parent2)
        return offspring

    def _evaluate_population(self,
                             population: ndarray,
                             num_days: int,
                             routing_method: Clustering.RoutingMethods,
                             graph: ndarray,
                             durations: ndarray,
                             coordinates: ndarray) -> ndarray:
        """
        Evaluates the fitness of a given population. This fitness is calculated
        by evaluating the route that's found from passing each individual's
        clusters to the given routing method.
        :param population: Population to evaluate.
        :param num_days: The number of days in the route.
        :param routing_method: The routing method to use on each cluster.
        :param graph: The graph input as an adjacency matrix.
        :param durations: Duration spent at each location.
        :return: 1D ndarray representing each individual's fitness.
        """
        evaluations = np.zeros(self.population_size)
        for individual in range(self.population_size):
            route = self.find_route_from_clusters(
                population[individual], num_days, routing_method, graph,
                durations, coordinates)

            evaluations[individual], _, _ = self.evaluate_route(
                route, num_days, graph, durations)
        return evaluations

    @staticmethod
    def _relabel_individuals_clusters(individual: ndarray) -> ndarray:
        """
        Relabels cluster assignments to be in order of appearance. This
        ensures consistency between parent1 and parent2
        :param individual: The individual to relabel.
        :return: The individual with relabelled clusters.
        """
        # Get unique values and the first index they're used
        unique_vals, first_indices = np.unique(individual,
                                               return_index=True)

        sorted_unique_vals = unique_vals[np.argsort(first_indices)]
        mapping = {unique: sorted for unique, sorted in
                   zip(unique_vals, sorted_unique_vals)}

        relabelled_individual = np.vectorize(mapping.get)(individual)
        return relabelled_individual


class GeneticCentroidClustering(Genetic, Clustering):
    """
    Class for genetic centroid clustering. Genome contains the coordinates of
    the centroid for each class.
    """
    def __init__(self,
                 num_generations: int,
                 population_size: int,
                 crossover_probability: float,
                 mutation_probability: float,
                 generations_per_update: int | None = 1,
                 plotting: bool = True,
                 random_seed: int | None = None):
        """
        Initialises genetic centroid clustering with given parameters.
        :param num_generations: Number of generations to run.
        :param population_size: Number of individuals in each population.
        :param crossover_probability: Probability of crossover (0-1).
        :param mutation_probability: Probability of mutation (0-1).
        :param generations_per_update: Number of generations between each
        progress update. If None or less than 0, no updates given.
        :param plotting: Whether to display plots on each update.
        :param random_seed: Specified seed for random number generators.
        """
        super().__init__(num_generations, population_size,
                         crossover_probability, mutation_probability,
                         generations_per_update, plotting, random_seed)

    def find_clusters(self,
                      coordinates: ndarray,
                      graph: ndarray,
                      durations: ndarray,
                      num_days: int,
                      routing_method: Clustering.RoutingMethods) -> ndarray:
        """
        Sorts `num_locations` coordinates into `num_days` clusters via a genetic
        algorithm approach.  Genome contains the coordinates of the centroid for
        each class.
        :param coordinates: Coordinates of each location.
        :param graph: The graph input as an adjacency matrix.
        :param durations: Duration spent at each location.
        :param num_days: The number of days in the route.
        :param routing_method: The routing method to use on each cluster.
        :return: Each location's cluster assignment. A 1D array of shape
        (num_locations).
        """
        evaluations = np.empty(self.population_size)
        evaluations[:] = float('inf')

        lon_range = (coordinates[:, 0].min(), coordinates[:, 0].max())
        lat_range = (coordinates[:, 1].min(), coordinates[:, 1].max())

        # Initialise coordinates
        population = self._generate_random_centroids(num_days, lat_range,
                                                     lon_range)
        cluster_coordinates = np.array(coordinates[1:], copy=True)

        # Assign these before loop just in case num_generations is 0 and these
        # are used before initialisation
        parent1 = np.empty_like(num_days)
        index1 = 0

        for generation_number in range(self.num_generations):
            evaluations = self._evaluate_population(
                cluster_coordinates, population, num_days, routing_method,
                graph, durations)

            index1, index2 = np.argpartition(evaluations, 2)[:2]
            parent1 = population[index1]
            parent2 = population[index2]

            if (self.generations_per_update is not None
                    and generation_number % self.generations_per_update == 0):
                progress = 100 * generation_number/self.num_generations
                print(f"Centroid evolution {int(progress)}% complete: "
                      f"{generation_number}/{self.num_generations} completed. "
                      f"Best evaluation: {evaluations[index1]}")

                if self.plotting:
                    cluster_assignments = self._assign_nodes_to_centroid(
                        cluster_coordinates, parent1)

                    centre = coordinates.mean(axis=0)
                    Plotting.display_clusters(
                        cluster_coordinates, cluster_assignments, num_days,
                        parent1, centre)

            if float('inf') in (evaluations[index1], evaluations[index2]):
                population = self._generate_random_centroids(num_days, centre)
                if evaluations[index1] != float('inf'):
                    population[0] = parent1
                if evaluations[index2] != float('inf'):
                    population[0] = parent2
                continue

            # Crossover
            population[0] = parent1
            population[1] = parent2

            for individual in range(2, self.population_size):
                use_crossover = random.random() < self.crossover_probability

                # Generate random individual (increases genetic diversity)
                if not use_crossover:
                    new_lon = np.random.uniform(lon_range[0], lon_range[1],
                                                num_days)
                    new_lat = np.random.uniform(lat_range[0], lat_range[1],
                                                num_days)
                    population[individual] = np.vstack((new_lon, new_lat))
                    continue

                population[individual] = self._crossover(parent1, parent2)

                for cluster in range(num_days):
                    mutate = random.random() < self.mutation_probability
                    if mutate:
                        new_lon = np.random.uniform(lon_range[0], lon_range[1],
                                                    num_days)
                        new_lat = np.random.uniform(lat_range[0], lat_range[1],
                                                    num_days)
                        population[individual, cluster] = np.concatenate(
                            (new_lon, new_lat))

        print(f"Centroid evolution complete. "
              f"Best evaluation: {evaluations[index1]}")
        cluster_assignments = self._assign_nodes_to_centroid(
            cluster_coordinates, parent1)

        return cluster_assignments

    def _crossover(self,
                   parent1: ndarray,
                   parent2: ndarray,) -> ndarray:
        """
        Performs crossover between two given individuals. For each location,
        chooses a cluster from either parent.
        :param parent1: First parent's genome.
        :param parent2: Second parent's genome.
        :return: Offspring of each parent.
        """
        offspring = np.empty_like(parent2)
        reordered_parent2 = np.empty_like(parent2)

        # Get distances between parent1 & parent2 centroids
        distances = np.linalg.norm(parent1[:, np.newaxis] - parent2, axis=2)

        num_days = parent1.shape[0]

        # Reorder parent2 so clusters are similar to parent1
        for i in range(num_days):
            best_match = np.argmin(distances[i])
            reordered_parent2[i] = parent2[best_match]

            distances[:, best_match] = np.inf

        # Create centroids in-between parents'
        for i in range(num_days):
            weight = random.random()
            offspring[i] = weight * parent1[i] + (1-weight) * parent2[i]
        return offspring

    def _evaluate_population(self,
                             coordinates: ndarray,
                             population: ndarray,
                             num_days: int,
                             routing_method: Clustering.RoutingMethods,
                             graph: ndarray,
                             durations: ndarray) -> ndarray:
        """
        Evaluates the fitness of a given population. This fitness is calculated
        by evaluating the route that's found from passing each individual's
        clusters to the given routing method.
        :param coordinates: Coordinates of each location.
        :param population: Population to evaluate.
        :param num_days: The number of days in the route.
        :param routing_method: The routing method to use on each cluster.
        :param graph: The graph input as an adjacency matrix.
        :param durations: Duration spent at each location.
        :return: 1D ndarray representing each individual's fitness.
        """
        evaluations = np.zeros(self.population_size)
        for individual in range(self.population_size):
            clusters = self._assign_nodes_to_centroid(coordinates,
                                                      population[individual])

            route = self.find_route_from_clusters(
                clusters, num_days, routing_method, graph,
                durations, coordinates)

            evaluations[individual] = self.evaluate_route(route, num_days,
                                                          graph, durations)
        return evaluations

    def _generate_random_centroids(self,
                                   num_days: int,
                                   lon_range: (float, float),
                                   lat_range: (float, float)) -> ndarray:
        """
        Generates random centroids for each individual in the population.
        :param num_days: The number of days in the route.
        :param centre: The centre of the coordinates.
        :return: A population of random centroids.
        """
        centroid_x_coordinates = np.random.uniform(lon_range[0],
                                                   lon_range[1],
                                                   size=(self.population_size,
                                                         num_days))
        centroid_y_coordinates = np.random.uniform(lat_range[0],
                                                   lat_range[1],
                                                   size=(self.population_size,
                                                         num_days))

        return np.dstack((centroid_x_coordinates, centroid_y_coordinates))


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
