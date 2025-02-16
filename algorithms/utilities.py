import numpy as np
from numpy import ndarray  # For type hints
from enum import Enum


class ClusteringMethods(Enum):
    K_MEANS = 0


class RoutingMethods(Enum):
    BRUTE_FORCE = 0
    GREEDY = 1


def evaluate_route(route: ndarray,
                   num_days: int,
                   graph: ndarray) -> float:
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
