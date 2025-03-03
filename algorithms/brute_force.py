import numpy as np
from numpy import ndarray  # For type hints

from .utilities import evaluate_route, generate_route


def brute_force(num_locations: int,
                num_days: int,
                graph: ndarray) -> ndarray:
    """
    A brute force solver for the problem. Can be used on its own or with
    clustering. When num_days equals 1, this becomes a TSP solver.
    :param num_locations: The number of locations in the route.
    :param num_days: The number of days in the route.
    :param graph: The adjacency matrix for the graph.
    :return: Returns a 1D np array representing the best route.
    """
    route_length = num_locations + num_days - 1

    # get (route_length - 1)! Do -1 because all routes end at '0'
    n_routes = 1
    for i in range(2, route_length):
        n_routes *= i

    best_route = generate_route(0, n_routes, num_locations, num_days,
                                route_length)
    best_evaluation = evaluate_route(best_route, num_days, graph)

    for i in range(1, n_routes):  # yikes
        route = generate_route(i, n_routes, num_locations, num_days,
                               route_length)
        evaluation = evaluate_route(route, num_days, graph)

        if evaluation < best_evaluation:
            best_route = np.array(route, copy=True)
            best_evaluation = evaluation

    return best_route
