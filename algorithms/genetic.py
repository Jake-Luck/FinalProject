import random

import numpy as np
from numpy import ndarray
from .utilities import RoutingMethods, generate_route


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
        cluster_assignments = np.empty(_num_locations)
        for _i in range(_num_locations):
            cluster_assignments[_i] = random.randrange(_num_days)

        _clusters = list[ndarray]()
        for _i in range(num_days):
            indexes_in_cluster = np.where(cluster_assignments == i)[0] + 1
            _clusters.append(np.concatenate(([0], indexes_in_cluster)))
        return _clusters

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




