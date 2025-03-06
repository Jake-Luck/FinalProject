import random

import numpy as np
from numpy import ndarray  # For type hints

from core.plotting import display_clusters
from .utilities import RoutingMethods, evaluate_route, \
    find_route_from_cluster_assignments


def assign_nodes_to_centroid(_coordinates: ndarray,
                             _centroids: ndarray) -> ndarray:
    """
    Assigns each coordinate a cluster by computing distance from each
    coordinate to each centroid and choosing the smallest distance.

    :param _coordinates: Coordinates of each location.
    :param _centroids: Coordinates of each cluster's centroid.
    :return: A 1D array of shape (n). Represents the chosen clusters.
    """
    distances = np.linalg.norm(
        _coordinates[:, np.newaxis, :2] - _centroids[:, :2], axis=2)
    _clusters = np.argmin(distances, axis=1)
    return _clusters


def genetic_centroid_clustering(num_generations: int,
                                population_size: int,
                                crossover_probability: float,
                                mutation_probability: float,
                                coordinates: ndarray,
                                graph: ndarray,
                                num_days: int,
                                route_length: int,
                                routing_method: RoutingMethods,
                                generations_per_plot: int,
                                random_seed: int | None = None,) -> ndarray:
    def crossover(_parent1: ndarray,
                  _parent2: ndarray,
                  _num_days: int) -> ndarray:
        offspring = np.empty_like(_parent2)
        reordered_parent2 = np.empty_like(_parent2)

        # Get distances between parent1 & parent2 centroids
        distances = np.linalg.norm(_parent1[:, np.newaxis] - _parent2,
                                         axis=2)

        # Reorder parent2 so clusters are similar to parent1
        for i in range(_num_days):
            best_match = np.argmin(distances[i])
            reordered_parent2[i] = _parent2[best_match]

            distances[:, best_match] = np.inf

        # Create centroids in-between parents'
        for i in range(_num_days):
            weight = random.random()
            offspring[i] = weight * _parent1[i] + (1-weight) * _parent2[i]

        return offspring
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    evaluations = np.empty(population_size)
    evaluations[:] = float('inf')

    # Randomly Assign Centroids
    centre = coordinates[0]
    centroid_x_coordinates = np.random.uniform(centre[0] -0.1, centre[0] + 0.1,
                                      (population_size, num_days))
    centroid_y_coordinates = np.random.uniform(centre[1] -0.1, centre[1] + 0.1,
                                      (population_size, num_days))

    population = np.dstack((centroid_x_coordinates, centroid_y_coordinates))
    cluster_coordinates = np.array(coordinates[1:], copy=True)

    # Assign these before loop just in case num_generations is 0 and these are
    # used before initialisation
    clusters = np.empty(population_size)
    index1 = 0

    for generation_number in range(num_generations):
        for individual in range(population_size):
            clusters = assign_nodes_to_centroid(cluster_coordinates,
                                                population[individual])

            route = find_route_from_cluster_assignments(clusters, num_days,
                                                        routing_method, graph)

            evaluations[individual] = evaluate_route(route, num_days, graph)

        if generation_number % generations_per_plot == 0:
            display_clusters(coordinates, clusters, num_days, population[0])

        index1, index2 = np.argpartition(evaluations, 2)[:2]
        parent1 = population[index1]
        parent2 = population[index2]

        # Crossover
        population[0] = parent1
        population[1] = parent2

        for individual in range(2, population_size):
            use_crossover = random.random() < crossover_probability

            # Generate random individual (increases genetic diversity)
            if not use_crossover:
                np.random.uniform(centre[1] -0.1, centre[1] + 0.1,
                                  (num_days, 2))
                continue

            population[individual] = crossover(parent1, parent2, num_days)

            for cluster in range(num_days):
                mutate = random.random() < mutation_probability
                if mutate:
                    mutation = np.random.uniform(-0.01, 0.01, 2)
# todo: change array indexing throughout project to below format:
                    population[individual, cluster] += mutation
        print(f"Generation {generation_number} completed, best evaluation: "
              f"{evaluations[index1]}")

    print(f"Evolution completed, best evaluation: {evaluations[index1]}")
    clusters = assign_nodes_to_centroid(cluster_coordinates,
                                        population[index1])

    route = find_route_from_cluster_assignments(clusters, num_days,
                                                routing_method, graph)
    return route

def genetic_clustering(num_generations: int,
                       population_size: int,
                       crossover_probability: float,
                       mutation_probability: float,
                       graph: ndarray,
                       num_locations: int,
                       num_days: int,
                       route_length: int,
                       routing_method: RoutingMethods,
                       random_seed: int | None = None) -> ndarray:
    """
    # todo: fill in this docstring
    :param num_generations:
    :param population_size:
    :param crossover_probability:
    :param mutation_probability:
    :param graph:
    :param num_locations:
    :param num_days:
    :param route_length:
    :param routing_method:
    :param random_seed:
    :return:
    """
    def crossover(_parent1: ndarray,
                  _parent2: ndarray) -> ndarray:
        """
        # todo: fill in this docstring
        :param _parent1:
        :param _parent2:
        :return:
        """
        # Keeps value if parents the same, otherwise sets to -1
        offspring = np.where(_parent1 == _parent2, parent1, -1)

        # For each conflicting assignment, choose from parent randomly.
        conflicts = np.argwhere(offspring == -1)
        for conflict in conflicts:
            chosen_parent = random.randrange(2)
            offspring[conflict] = parent1[conflict] if chosen_parent == 0 \
                                                    else parent2[conflict]
        return offspring
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)

    evaluations = np.empty(population_size)
    evaluations[:] = float('inf')

    n_routes = 1
    for i in range(2, route_length):
        n_routes *= i

    # Assign random clusters to each location
    population = np.random.randint(num_days,
                                   size=(population_size, num_locations-1))

    # Assign these before loop just in case num_generations is 0 and these are
    # used before initialisation
    index1 = 0

    for generation_number in range(num_generations):
        for i in range(population_size):
            route = find_route_from_cluster_assignments(population[i],
                                                        num_days,
                                                        routing_method,
                                                        graph)
            evaluations[i] = evaluate_route(route, num_days, graph)

        index1, index2 = np.argpartition(evaluations, 2)[:2]
        parent1 = population[index1]
        parent2 = population[index2]

        # Crossover
        population[0] = parent1
        population[1] = parent2

        for i in range(2, population_size):
            use_crossover = random.random() < crossover_probability

            # Generate random individual (increases genetic diversity)
            if not use_crossover:
                population[i] = np.random.randint(num_days,
                                                  size=num_locations-1)
                continue

            population[i] = crossover(parent1, parent2)

            for j in range(num_locations-1):
                mutate = random.random() < mutation_probability
                if mutate:
                    population[i][j] = random.randrange(num_days)

        print(f"Generation {generation_number} completed, best evaluation: "
              f"{evaluations[index1]}")

    print(f"Evolution completed, best evaluation: {evaluations[index1]}")
    route = find_route_from_cluster_assignments(population[index1],
                                                num_days,
                                                routing_method,
                                                graph)
    return route

