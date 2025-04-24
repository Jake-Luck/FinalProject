"""
Provides algorithm base class, which each algorithm will inherit from.
"""
import math
import numpy as np
from numpy import ndarray  # For type hints


class Algorithm:
    """
    Algorithm base class. Provides methods for route evaluation and permutation
    based route generation.
    """
    @staticmethod
    def evaluate_route(route: ndarray,
                       num_days: int,
                       graph: ndarray,
                       durations: ndarray) -> tuple[float, float, ndarray]:
        """
        Evaluates a given route using the time taken and deviation between day
        lengths.
        :param route: Route to be evaluated.
        :param num_days: Number of days in the route.
        :param graph: Graph as an adjacency matrix.
        :param durations: Duration spent at each location.
        :return: Route evaluation
        """
        evaluation = 0.0
        previous_index = current_day = 0
        evaluation_per_day = np.zeros(num_days)

        # Sum durations in route
        for index in route:
            # Invalid route returns infinite cost
            if previous_index == index or current_day >= num_days:
                evaluation_per_day[:] = float('inf')
                return float('inf'), float('inf'), evaluation_per_day

            evaluation += graph[previous_index, index]
            evaluation += durations[index]
            evaluation_per_day[current_day] += graph[previous_index, index]
            evaluation_per_day[current_day] += durations[index]

            if index == 0:
                current_day += 1
            previous_index = index

        evaluation /= math.sqrt(num_days)
        standard_deviation = float(np.std(evaluation_per_day))
        evaluation += standard_deviation

        return evaluation, standard_deviation, evaluation_per_day

    @staticmethod
    def generate_route(route_number: int,
                       n_routes: int,
                       location_set: list,
                       route_length: int) -> ndarray:
        """
        Generates a route using factorial based division.
        :param route_number: The route number/index of permutation.
        :param n_routes: The total number of routes (route_length - 1)!
        :param location_set: The set of locations to choose from.
        :param route_length: The length of the route.
        :return: Returns a 1D np array representing the generated route.
        """
        route = np.empty(route_length, dtype=int)

        # This computes (n_routes - 1)!
        n_factorial = int(n_routes / (route_length - 1))

        for i in range(route_length - 2, -1, -1):
            selected_index = int(route_number / n_factorial)
            route[i] = location_set.pop(selected_index)

            if i == 0:
                continue

            route_number %= n_factorial
            n_factorial /= i
        route[route_length - 1] = 0  # All routes end going back to centre     ⠀
        return route
