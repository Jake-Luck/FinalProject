import numpy as np
from numpy import ndarray  # For type hints
from enum import Enum

from algorithms import brute_force
from algorithms import greedy


class ClusteringMethods(Enum):
    """
    These are clustering methods for use alongside routing algorithms.
    """
    K_MEANS = 0,
    'Assigns random \'centroids\' and assign individuals to their closest, \
    then update centroids based on cluster\'s mean and repeat the process.'


class RoutingMethods(Enum):
    """
    These are travelling salesmen solvers for use with clustering.
    """
    # Brute force needs lambda for num_days (it has num_days as a parameter)
    # because it can be used without clustering too.
    BRUTE_FORCE = lambda _n, _g: brute_force.brute_force(_n, 1, _g)
    'Compares every possible route to find the best.'
    GREEDY = greedy.greedy
    'Always chooses the shortest path from any given position.'


# TODO: Move base_set construction outside of method. In case generate route is
# to be run in parallel, in which want that computation to be done outside and
# used repeatedly. Make sure to copy based set within generate route.
def generate_route(route_number: int,
                   n_routes: int,
                   num_locations: int,
                   num_days: int,
                   route_length: int) -> ndarray:
    """
    Generates a route using factorial based division.
    :param route_number: The route number/index of permutation.
    :param n_routes: The total number of routes (route_length - 1)!
    :param num_locations: The number of locations in the route.
    :param num_days: The number of days in the route.
    :param route_length: The length of the route.
    :return: Returns a 1D np array representing the generated route.
    """
    base_set = list()
    for _ in range(num_days - 1):  # -1 because 0 added to end of all routes
        base_set.append(0)
    for i in range(1, num_locations):
        base_set.append(i)

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


def find_route_from_cluster_assignments(cluster_assignments: ndarray,
                                        num_days: int,
                                        routing_method: RoutingMethods,
                                        graph: ndarray) -> ndarray:
    clusters = list[ndarray]()
    for i in range(num_days):
        indexes_in_cluster = np.where(cluster_assignments == i)[0] + 1
        clusters.append(np.concatenate(([0], indexes_in_cluster)))

    # numpy magic
    # np.ix_([1,2,3], [1,2,3]) returns [[[1],[2],[3]],[1,2,3]]
    # Which can they be used to access graphs
    graphs = [graph[np.ix_(indexes, indexes)] for indexes in clusters]

    route = np.empty(0, dtype=int)

    for sub_graph, cluster_index in zip(graphs, clusters):
        n = sub_graph.shape[0]
        sub_route = routing_method(n, sub_graph)

        route = np.concatenate((route, cluster_index[sub_route]))
    return route


def evaluate_route(route: ndarray,
                   num_days: int,
                   graph: ndarray) -> float:
    """
    Evaluates a given route using the time taken and deviation between day
    lengths.
    :param route: Route to be evaluated.
    :param num_days: Number of days in the route.
    :param graph: Graph as an adjacency matrix.
    :return: Route evaluation
    """
    evaluation = 0.0
    previous_index = current_day = 0
    evaluation_per_day = np.zeros(num_days)

    # Sum durations in route
    for index in route:
        if previous_index == index:
            return float('inf')

        evaluation += graph[previous_index][index]
        evaluation_per_day[current_day] += graph[previous_index][index]

        if index == 0:
            current_day += 1
        previous_index = index

    # Multiply durations by 1 + the standard deviation of time spent each day
    evaluation *= 1 + np.std(evaluation_per_day)

    return evaluation
