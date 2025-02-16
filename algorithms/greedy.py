import sys

import numpy as np
from numpy import ndarray  # For type hints


def greedy(num_locations: int,
           graph: ndarray) -> ndarray:
    route_length = num_locations
    route = np.empty(route_length, dtype=int)
    index = 0
    for i in range(route_length):
        graph[:, index] = sys.maxsize
        index = graph[index].argmin()
        route[i] = index
    return route
