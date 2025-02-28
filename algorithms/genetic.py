import random

import numpy as np
from numpy import ndarray

from .utilities import RoutingMethods, generate_route, evaluate_route


def genetic(num_generations: int,
            population_size: int,
            graph: ndarray,
            num_locations: int,
            num_days: int,
            route_length: int,
            routing_method: RoutingMethods | None,
            random_seed: int | None = None) -> ndarray:
    def initialise_clusters(_population: ndarray,
                            _num_locations: int,
                            _num_days: int) -> list[ndarray]:
        pass

    def initialise_routes(_population: ndarray,
                          _num_locations: int,
                          _num_days: int,
                          _n_routes: int,
                          _route_length: int) -> None:
        for _i in range(_population):
            route_number = random.randrange(_n_routes)
            _population[i] = generate_route(route_number, _n_routes,
                                            _num_locations, _num_days,
                                            _route_length)
    # Setup
    if random_seed is not None:
        random.seed(random_seed)

    evaluations = np.empty(population_size)
    evaluations[:] = float('inf')

    n_routes = 1
    for i in range(2, route_length):
        n_routes *= i

    population = np.empty((population_size, route_length))
    if routing_method is not None:
        initialise_routes(population, num_locations, num_days, n_routes,
                          route_length)  # Sets population in place, no return
    else:
        clusters = initialise_clusters()

    for generation_number in range(num_generations):
        for individual in range(population_size):
            evaluations[individual] = evaluate_route(population[individual],
                                                     graph)

        index1, index2 = np.argpartition(evaluations, 2)[:2]
        parent1 = population[index1]
        parent2 = population[index2]

        # Keep parents, mutate rest of population




