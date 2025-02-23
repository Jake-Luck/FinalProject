import random

import numpy as np
from numpy import ndarray
from .utilities import RoutingMethods, generate_route


def genetic(num_generations: int,
            population_size: int,
            graph: ndarray,
            num_locations,
            num_days: int,
            route_length: int,
            routing_method: RoutingMethods | None,
            random_seed: int | None) -> ndarray:
    def genetic_clusters():
        pass

    def genetic_routing():
        pass
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
        genetic_clusters()
    else:
        genetic_routing()

    for i in range(population_size):
        route_number = random.randrange(0, n_routes)
        population[i] = generate_route(route_number, n_routes, num_locations,
                                       num_days, route_length)





