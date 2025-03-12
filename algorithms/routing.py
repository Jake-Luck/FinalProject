import numpy as np
from numpy import ndarray

from algorithms.algorithm import Algorithm


class Routing(Algorithm):
    @staticmethod
    def greedy(num_locations: int,
               graph: int):
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
            graph[:, index] = np.iinfo(np.int32).max
            index = graph[index].argmin()
            route[i] = index
        return route

    @staticmethod
    def brute_force(num_locations: int,
                    num_days: int,
                    graph: ndarray) -> ndarray:
        """
        A brute force solver for the problem. Can be used on its own or with
        clustering. When num_days equals 1, this becomes a TSP solver.
        :param num_locations: The number of locations in the route.
        :param num_days: The number of days in the route.
        :param graph: The adjacency matrix for the graph.
        :return: 1D ndarray representing the best route.
        """
        if num_locations < 2:
            raise ValueError(f"num_locations={num_locations}, "
                             f"must be at least 2.")
        if num_days < 1:
            raise ValueError(f"num_days={num_days}, must be at least 1.")

        route_length = num_locations + num_days - 1

        # get (route_length - 1)! Do -1 because all routes end at '0'
        n_routes = 1
        for i in range(2, route_length):
            n_routes *= i

        best_route = Algorithm.generate_route(0, n_routes,
                                              num_locations, num_days,
                                              route_length)
        best_evaluation = Algorithm.evaluate_route(best_route, num_days, graph)

        for i in range(1, n_routes):  # yikes
            route = Algorithm.generate_route(i, n_routes, num_locations,
                                             num_days, route_length)
            evaluation = Algorithm.evaluate_route(route, num_days, graph)

            if evaluation < best_evaluation:
                best_route = np.array(route, copy=True)
                best_evaluation = evaluation

        return best_route


