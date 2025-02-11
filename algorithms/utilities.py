import numpy as np


def evaluate_route(route: np.ndarray, num_days: int,
                   graph: np.ndarray) -> float:
    evaluation = 0.0
    previous_index = current_day = 0
    evaluation_per_day = np.zeros(num_days)

    # Sum durations in route
    for index in route:
        if previous_index == index:
            return float('inf')

        evaluation += graph[previous_index][index]
        evaluation_per_day[current_day] += graph[previous_index][index]

        if index == 0: current_day += 1
        previous_index = index

    # Multiply durations by 1 + the standard deviation of time spent each day
    evaluation *= 1 + np.std(evaluation_per_day)

    return evaluation
