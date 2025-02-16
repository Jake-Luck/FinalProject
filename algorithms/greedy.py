import sys

import numpy as np
from numpy import ndarray  # For type hints


def greedy(num_locations: int,
           graph: ndarray) -> ndarray:
    """
    A very simple solver for use with clustering, will not find a valid route
    on its own. This is a conventional TSP solver.
    :param num_locations: The number of locations in the route.
    :param graph: The adjacency matrix for the graph.
    :return: 1D np array representing the route found.
    """
    route_length = num_locations
    route = np.empty(route_length, dtype=int)
    index = 0
    for i in range(route_length):
        graph[:, index] = sys.maxsize
        index = graph[index].argmin()
        route[i] = index
    return route
