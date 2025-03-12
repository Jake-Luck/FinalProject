"""
Provides algorithm base class, which each algorithm will inherit from.
"""
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

    @staticmethod
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
