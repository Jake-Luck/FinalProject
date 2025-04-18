"""
Provides Genetic base class and those that inherit it.
"""
from algorithms.algorithm import Algorithm
from algorithms.clustering import Clustering
from core.plotting import Plotting

from abc import abstractmethod
import numpy as np
from numpy import ndarray  # For type hints
import random


class Genetic(Algorithm):
    """
    Base class for genetic algorithms, includes abstract method for crossover.
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
        Initialises genetic class with given parameters.
        :param num_generations: Number of generations to run.
        :param population_size: Number of individuals in each population.
        :param crossover_probability: Probability of crossover (0-1).
        :param mutation_probability: Probability of mutation (0-1).
        :param generations_per_update: Number of generations between each
        progress update. If None or less than 0, no updates given.
        :param plotting: Whether to display plots on each update.
        :param random_seed: Specified seed for random number generators.
        """
        if num_generations < 1:
            raise ValueError("num_generations must be at least 1.")
        if population_size < 3:
            raise ValueError("population_size must be at least 3.")

        self.num_generations = num_generations
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.plotting = plotting

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed = random_seed

        if (not isinstance(generations_per_update, int)
                or generations_per_update < 1):
            generations_per_update = None
        self.generations_per_update = generations_per_update

    @abstractmethod
    def _crossover(self,
                   parent1: ndarray,
                   parent2: ndarray) -> ndarray:
        """
        Create offspring by performing crossover on two given individuals.

        Abstract method to be implemented by subclasses.
        :param parent1: First parent's genome.
        :param parent2: Second parent's genome.
        :return: Offspring of each parent.
        """
        pass


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
                 random_seed: int | None = None):
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

        for generation_number in range(self.num_generations):
            evaluations = self._evaluate_population(population, num_days,
                                                    routing_method, graph,
                                                    durations)

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

                if self.plotting:
                    Plotting.display_clusters(coordinates, parent1, num_days)

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

        print(f"Cluster evolution complete. "
              f"Best evaluation: {evaluations[index1]}")
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
        parent2 = relabel_individuals_clusters(parent2)

        crossover_mask = np.random.randint(0, 2, size=len(parent1))
        offspring = np.where(crossover_mask == 0, parent1, parent2)

        return offspring

    def _evaluate_population(self,
                             population: ndarray,
                             num_days: int,
                             routing_method: Clustering.RoutingMethods,
                             graph: ndarray,
                             durations: ndarray) -> ndarray:
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
                durations)
            evaluations[individual], _, _ = self.evaluate_route(
                route, num_days, graph, durations)
        return evaluations

        @staticmethod
        def _relabel_individuals_clusters(individual: ndarray) -> ndarray:
            """
            Renames cluster assignments to be in order of appearance. This
            ensures consistency between parent1 and parent2
            :param individual: The individual to rename.
            :return: The individual with renamed clusters.
            """
            # Get unique values and the first index they're used
            unique_vals, first_indices = np.unique(individual,
                                                   return_index=True)

            sorted_unique_vals = unique_vals[np.argsort(first_indices)]
            mapping = {unique: sorted for unique, sorted in
                       zip(unique_vals, sorted_unique_vals)}

            renamed_individual = np.vectorize(mapping.get)(individual)
            return renamed_individual


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

        # Randomly Assign Centroids
        centre = coordinates.mean(axis=0)
        population = self._generate_random_centroids(num_days, centre)
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
                    population[individual] = np.random.uniform(centre[1] - 0.1,
                                                               centre[1] + 0.1,
                                                               (num_days, 2))
                    continue

                population[individual] = self._crossover(parent1, parent2)

                for cluster in range(num_days):
                    mutate = random.random() < self.mutation_probability
                    if not mutate:
                        continue
                    mutation = np.random.uniform(-0.01, 0.01, 2)
                    population[individual, cluster] += mutation

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
                clusters, num_days, routing_method, graph, durations)

            evaluations[individual], _, _ = self.evaluate_route(
                route, num_days, graph, durations)
        return evaluations

    def _generate_random_centroids(self,
                                   num_days: int,
                                   centre: ndarray):
        """
        Generates random centroids for each individual in the population.
        :param num_days: The number of days in the route.
        :param centre: The centre of the coordinates.
        :return: A population of random centroids.
        """
        centroid_x_coordinates = np.random.uniform(centre[0] - 0.1,
                                                   centre[0] + 0.1,
                                                   size=(self.population_size,
                                                         num_days))
        centroid_y_coordinates = np.random.uniform(centre[1] - 0.1,
                                                   centre[1] + 0.1,
                                                   size=(self.population_size,
                                                         num_days))

        return np.dstack((centroid_x_coordinates, centroid_y_coordinates))


class GeneticRouting(Genetic):
    """
    Class for genetic routing. Genome contains the order of locations in the
    route.
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
        Initialises genetic routing with given parameters.
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

    def find_route(self,
                   graph: ndarray,
                   durations: ndarray,
                   num_locations: int,
                   num_days: int,
                   coordinates: ndarray | None = None) -> ndarray:
        """
        Generates a route that visits all locations using a genetic algorithm
        approach. Genome contains the order of locations in the route.
        :param graph: The graph input as an adjacency matrix.
        :param durations: Duration spent at each location.
        :param num_locations: The number of locations in the route.
        :param num_days: The number of days in the route.
        :param coordinates: Coordinates of each location. Used for plotting.
        :return: A route that visits all locations.
        """

        if coordinates is None and self.plotting is True:
            print("No coordinates provided, setting plotting to False")
            self.plotting = False

        evaluations = np.empty(self.population_size)
        evaluations[:] = float('inf')

        route_length = num_locations + num_days - 1

        n_routes = 1
        for i in range(2, route_length):
            n_routes *= i
        # Assign random routes to each location
        population = self._generate_random_routes(num_locations, num_days,
                                                  n_routes, route_length)

        # Assign these before loop just in case num_generations is 0 and these
        # are used before initialisation
        index1 = 0
        parent1 = population[index1]

        for generation_number in range(self.num_generations):
            evaluations = self._evaluate_population(population, num_days, graph,
                                                    durations)

            index1, index2 = np.argpartition(evaluations, 2)[:2]
            parent1 = population[index1]
            parent2 = population[index2]

            if self.generations_per_update is not None \
                    and generation_number % self.generations_per_update == 0:
                progress = 100 * generation_number/self.num_generations
                print(f"Route evolution {int(progress)}% complete: "
                      f"{generation_number}/{self.num_generations} completed. "
                      f"Best evaluation: {evaluations[index1]}")

                if self.plotting:
                    Plotting.display_route(coordinates, parent1)

            # Crossover
            population[0] = parent1
            population[1] = parent2

            for i in range(2, self.population_size):
                use_crossover = random.random() < self.crossover_probability

                # Generate random individual (increases genetic diversity)
                if not use_crossover:
                    route_number = random.randint(0, n_routes)
                    population[i] = self.generate_route(route_number, n_routes,
                                                        num_locations, num_days,
                                                        route_length)
                    continue

                population[i] = self._crossover(parent1, parent2)

                mutate = random.random() < self.mutation_probability
                if not mutate:
                    continue
                # swap two elements
                mutation_index1, mutation_index2 = random.sample(
                    range(num_locations), 2)
                temp = population[i, mutation_index1]
                population[i, mutation_index1] = population[i, mutation_index2]
                population[i, mutation_index2] = temp

        print(f"Route evolution complete. "
              f"Best evaluation: {evaluations[index1]}")
        return parent1

    def _crossover(self,
                   parent1: ndarray,
                   parent2: ndarray) -> ndarray:
        """
        Performs crossover between two given individuals.
        :param parent1: First parent's genome.
        :param parent2: Second parent's genome.
        :return: Offspring of each parent.
        """

        offspring = np.empty_like(parent1)
        offspring[:] = -1

        crossover_point = random.randint(0, len(parent1)-1)

        # Adds parent1 to offspring up to crossover point
        offspring[:crossover_point] = parent1[:crossover_point]

        days_left = (np.count_nonzero(parent1 == 0) -
                     np.count_nonzero(offspring == 0)) - 1

        # Adds remaining cities in order that they appear in parent2
        offspring_index = crossover_point
        for i in range(0, parent2.shape[0]-1):
            if parent2[i] == 0 and days_left > 0 or parent2[i] not in offspring:
                offspring[offspring_index] = parent2[i]
                offspring_index += 1
                if parent2[i] == 0:
                    days_left -= 1

        # Return to start at end of route
        offspring[offspring.shape[0]-1] = 0
        return offspring

    def _evaluate_population(self,
                             population: ndarray,
                             num_days: int,
                             graph: ndarray,
                             durations: ndarray) -> ndarray:
        """
        Evaluates the fitness of a given population.
        :param population: Population to evaluate.
        :param num_days: The number of days in the route.
        :param graph: The graph input as an adjacency matrix.
        :param durations: Duration spent at each location.
        :return: 1D ndarray representing each individual's fitness.
        """
        evaluations = np.zeros(self.population_size)
        for individual in range(self.population_size):
            evaluations[individual], _, _ = self.evaluate_route(
                population[individual], num_days, graph, durations)
        return evaluations

    def _generate_random_routes(self,
                                num_locations: int,
                                num_days: int,
                                n_routes: int,
                                route_length: int) -> ndarray:
        """
        Generates a random route for each individual in the population.
        :param num_locations: The number of locations in the route.
        :param num_days: The number of days in the route.
        :param n_routes: Total number of possible routes
        :return: A population of random routes.
        """
        population = np.empty(shape=(self.population_size, route_length),
                              dtype=int)
        for i in range(self.population_size):
            route_number = random.randint(0, n_routes)
            population[i] = self.generate_route(route_number, n_routes,
                                                num_locations, num_days,
                                                route_length)
        return population

