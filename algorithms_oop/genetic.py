import random
from abc import abstractmethod

import numpy as np
from numpy import ndarray  # For type hints

from algorithms_oop.algorithm import Algorithm
from algorithms_oop.clustering import Clustering

from core.plotting import display_clusters


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
        Initialises genetic algorithm with given parameters.
        :param num_generations: Number of generations to run.
        :param population_size: Number of individuals in each population.
        :param crossover_probability: Probability of crossover (0-1).
        :param mutation_probability: Probability of mutation (0-1).
        :param generations_per_update: Number of generations between each
        progress update. If None or less than 0, no updates given.
        :param plotting: Whether to display plots on each update.
        :param random_seed: Specified seed for random number generators.
        """
        self.num_generations = num_generations
        self.population_size = population_size
        self.crossover_probability = crossover_probability
        self.mutation_probability = mutation_probability
        self.plotting = plotting

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

        if not isinstance(generations_per_update, int) or generations_per_update < 1:
            generations_per_update = None
        self.generations_per_update = generations_per_update

        evaluations = np.empty(population_size)
        evaluations[:] = float('inf')
        self.evaluations = evaluations

    @abstractmethod
    def _crossover(self,
                  parent1: ndarray,
                  parent2: ndarray) -> ndarray:
        """
        Create offspring by performing crossover on two given individuals.

        Abstract method to be implemented by subclasses.
        :param parent1: First parent's genetic representation.
        :param parent2: Second parent's genetic representation.
        """
        pass


# Pretty bad
class GeneticClustering(Genetic, Clustering):
    """

    """
    def __init__(self,
                 num_generations: int,
                 population_size: int,
                 crossover_probability: float,
                 mutation_probability: float,
                 generations_per_update: int | None = 1,
                 plotting: bool = True,
                 random_seed: int | None = None):
        super().__init__(num_generations, population_size,
                         crossover_probability, mutation_probability,
                         generations_per_update, plotting, random_seed)

    def _evaluate_population(self,
                             population: ndarray,
                             num_days: int,
                             routing_method: Clustering.RoutingMethods,
                             graph: ndarray):
        for individual in range(self.population_size):
            route = self.find_route_from_cluster_assignments(
                population[individual], num_days, routing_method, graph
            )
            self.evaluations[individual] = self.evaluate_route(route, num_days,
                                                               graph)

    def _crossover(self,
                  parent1: ndarray,
                  parent2: ndarray) -> ndarray:
        # Keeps value if parents the same, otherwise sets to -1
        offspring = np.where(parent1 == parent2, parent1, -1)

        # For each conflicting assignment, choose from parent randomly.
        conflicts = np.argwhere(offspring == -1)
        for conflict in conflicts:
            chosen_parent = random.randrange(2)
            offspring[conflict] = parent1[conflict] if chosen_parent == 0 \
                                                    else parent2[conflict]
        return offspring

    def find_clusters(self,
                   graph: ndarray,
                   num_locations: int,
                   num_days: int,
                   route_length: int,
                   routing_method: Clustering.RoutingMethods,
                   coordinates: ndarray | None = None) -> ndarray:
        if coordinates is None:
            print("No coordinates provided, setting plotting to False")
            self.plotting = False

        n_routes = 1
        for i in range(2, route_length):
            n_routes *= i

        # Assign random clusters to each location
        population = np.random.randint(num_days,
                                       size=(self.population_size,
                                             num_locations - 1))

        # Assign these before loop just in case num_generations is 0 and these are
        # used before initialisation
        index1 = 0

        for generation_number in range(self.num_generations):
            self._evaluate_population(population, num_days, routing_method,
                                      graph)

            index1, index2 = np.argpartition(self.evaluations, 2)[:2]
            parent1 = population[index1]
            parent2 = population[index2]

            if self.generations_per_update is not None \
                    and generation_number % self.generations_per_update == 0:
                print(f"Generation {generation_number} completed, best "
                      f"evaluation: {self.evaluations[index1]}")

                if self.plotting:
                    display_clusters(coordinates, parent1, num_days)


            # Crossover
            population[0] = parent1
            population[1] = parent2

            for i in range(2, self.population_size):
                use_crossover = random.random() < self.crossover_probability

                # Generate random individual (increases genetic diversity)
                if not use_crossover:
                    population[i] = np.random.randint(num_days,
                                                      size=num_locations - 1)
                    continue

                population[i] = self._crossover(parent1, parent2)

                for j in range(num_locations - 1):
                    mutate = random.random() < self.mutation_probability
                    if mutate:
                        population[i][j] = random.randrange(num_days)

        print(f"Evolution completed, best evaluation: {self.evaluations[index1]}")
        return parent1

class GeneticCentroidClustering(Genetic, Clustering):
    def __init__(self,
                 num_generations: int,
                 population_size: int,
                 crossover_probability: float,
                 mutation_probability: float,
                 generations_per_update: int | None = 1,
                 plotting: bool = True,
                 random_seed: int | None = None):
        super().__init__(num_generations, population_size,
                         crossover_probability, mutation_probability,
                         generations_per_update, plotting, random_seed)

    def _crossover(self,
                   parent1: ndarray,
                   parent2: ndarray,) -> ndarray:
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
                             graph: ndarray):
        for individual in range(self.population_size):
            clusters = self._assign_nodes_to_centroid(coordinates,
                                                     population[individual])

            route = self.find_route_from_cluster_assignments(clusters, num_days,
                                                             routing_method, graph)

            self.evaluations[individual] = self.evaluate_route(route, num_days,
                                                               graph)

    def find_clusters(self,
                   coordinates: ndarray,
                   graph: ndarray,
                   num_days: int,
                   routing_method: Clustering.RoutingMethods) -> ndarray:
        # Randomly Assign Centroids
        centre = coordinates[0]
        centroid_x_coordinates = np.random.uniform(centre[0] -0.1, centre[0] + 0.1,
                                          (self.population_size, num_days))
        centroid_y_coordinates = np.random.uniform(centre[1] -0.1, centre[1] + 0.1,
                                      (self.population_size, num_days))

        population = np.dstack((centroid_x_coordinates, centroid_y_coordinates))
        cluster_coordinates = np.array(coordinates[1:], copy=True)

        # Assign these before loop just in case num_generations is 0 and these are
        # used before initialisation
        parent1 = np.empty_like(num_days)
        index1 = 0

        for generation_number in range(self.num_generations):
            self._evaluate_population(cluster_coordinates, population, num_days, routing_method, graph)

            index1, index2 = np.argpartition(self.evaluations, 2)[:2]
            parent1 = population[index1]
            parent2 = population[index2]

            if (self.generations_per_update is not None
                    and generation_number % self.generations_per_update == 0):
                print(f"Generation {generation_number} completed, best "
                      f"evaluation: {self.evaluations[index1]}")

                if self.plotting:
                    cluster_assignments = self._assign_nodes_to_centroid(cluster_coordinates, parent1)
                    display_clusters(cluster_coordinates, cluster_assignments, num_days, parent1)

            # Crossover
            population[0] = parent1
            population[1] = parent2

            for individual in range(2, self.population_size):
                use_crossover = random.random() < self.crossover_probability

                # Generate random individual (increases genetic diversity)
                if not use_crossover:
                    np.random.uniform(centre[1] - 0.1, centre[1] + 0.1,
                                      (num_days, 2))
                    continue

                population[individual] = self._crossover(parent1, parent2)

                for cluster in range(num_days):
                    mutate = random.random() < self.mutation_probability
                    if mutate:
                        mutation = np.random.uniform(-0.01, 0.01, 2)
                        # todo: change array indexing throughout project to below format:
                        population[individual, cluster] += mutation

        print(f"Evolution completed, best evaluation: {self.evaluations[index1]}")
        cluster_assignments = self._assign_nodes_to_centroid(
            cluster_coordinates, parent1)

        return cluster_assignments
