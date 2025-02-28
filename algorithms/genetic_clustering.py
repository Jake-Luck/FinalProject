import random
from typing import Callable

import numpy as np
from numpy import ndarray

from .utilities import RoutingMethods, evaluate_route, \
    find_route_from_cluster_assignments


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
    def crossover(_parent1: ndarray,
                  _parent2: ndarray) -> ndarray:
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

    evaluations = np.empty(population_size)
    evaluations[:] = float('inf')

    n_routes = 1
    for i in range(2, route_length):
        n_routes *= i

    # Assign random clusters to each location
    population = np.random.randint(num_days,
                                   size=(population_size, num_locations-1))

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
    route = find_route_from_cluster_assignments(population[0],
                                                num_days,
                                                routing_method,
                                                graph)
    return route

