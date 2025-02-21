import numpy as np
from numpy import ndarray  # For type hints

from .utilities import evaluate_route


def generate_route(base_set: list[int],
                   route_number: int,
                   n_routes: int,
                   route_length: int) -> ndarray:
    """
    Generates a route using factorial based division.
    :param base_set: Set to permutate.
    :param route_number: The route number/index of permutation.
    :param n_routes: The total number of routes (route_length - 1)!
    :param route_length: The length of the route.
    :return: Returns a 1D np array representing the generated route.
    """
    n_factorial = int(n_routes / (route_length - 1))
    route = np.empty(route_length, dtype=int)
    for i in range(route_length - 2, -1, -1):
        selected_index = int(route_number / n_factorial)
        route[i] = base_set.pop(selected_index)

        if i == 0: continue

        route_number %= n_factorial
        n_factorial /= i
    route[route_length - 1] = 0  # All routes end going back to centre
    return route


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

    best_evaluation = float('inf')
    best_route = np.empty(route_length, dtype=int)

    # get (route_length - 1)! Do -1 because all routes end at '0'
    n_routes = 1
    for i in range(2, route_length):
        n_routes *= i

    base_set = list()
    for _ in range(num_days - 1):  # -1 because 0 added to end of all routes
        base_set.append(0)
    for i in range(1, num_locations):
        base_set.append(i)

    for i in range(n_routes):  # yikes
        temp_set = base_set[:]
        route = generate_route(temp_set, i, n_routes, route_length)
        evaluation = evaluate_route(route, num_days, graph)

        if evaluation < best_evaluation:
            best_route = np.array(route, copy=True)
            best_evaluation = evaluation

    return best_route
