"""
Provides routing class that contains various routing algorithms.
"""
from algorithms.algorithm import Algorithm

import numpy as np
from numpy import ndarray


class Routing(Algorithm):
    """
    Provides various routing algorithms as static methods.
    """
    @staticmethod
    def gift_wrapping(num_locations: int,
                       coordinates: ndarray) -> ndarray:
        """
        Finds a convex hull around the coordinates, then attempts to add
        the interior points in an order that minimises route length.
        :param num_locations: The number of locations in the route.
        :param coordinates: Coordinates of each location.
        :param graph: The adjacency matrix for the graph.
        :return: 1D ndarray representing the best route.
        """
        # Gets the westernmost point (guaranteed to be on outside).
        starting_index = np.argmin(coordinates[:, 0])
        hull = []

        current_index = starting_index

        while True:
            hull.append(current_index)
            next_index = (current_index + 1) % num_locations

            for i in range(num_locations):
                if i == current_index:
                    continue

                orientation = np.cross(
                    coordinates[next_index] - coordinates[current_index],
                    coordinates[i] - coordinates[current_index])

                # If point i is more counter-clockwise
                if orientation > 0:
                    next_index = i

            # If returned to start, finish
            if next_index == starting_index:
                break

            current_index = next_index

        return np.array(hull)

    @staticmethod
    def brute_force(num_locations: int,
                    num_days: int,
                    graph: ndarray,
                    durations: ndarray) -> ndarray:
        """
        A brute force solver for the problem. Can be used on its own or with
        clustering. When num_days equals 1, this becomes a TSP solver.
        :param num_locations: The number of locations in the route.
        :param num_days: The number of days in the route.
        :param graph: The adjacency matrix for the graph.
        :param durations: Duration spent at each location.
        :return: 1D ndarray representing the best route.
        """
        if num_days < 1:
            raise ValueError(f"num_days={num_days}, must be at least 1.")
        if num_locations < 4:
            # 3 locations only has 1 result, so just use greedy
            return Routing.greedy_routing(num_locations, num_days,
                                          graph, durations)

        route_length = num_locations + num_days - 1

        # get (route_length - 1)! Do -1 because all routes end at '0'
        n_routes = 1
        for i in range(2, route_length):
            n_routes *= i

        best_route = Algorithm.generate_route(0, n_routes,
                                              num_locations, num_days,
                                              route_length)
        best_evaluation, _, _ = Algorithm.evaluate_route(best_route, num_days,
                                                         graph, durations)

        # iterations_per_update = n_routes / 10
        # progress = 0

        for i in range(1, n_routes):  # yikes
            route = Algorithm.generate_route(i, n_routes, num_locations,
                                             num_days, route_length)
            evaluation, _, _ = Algorithm.evaluate_route(route, num_days, graph,
                                                        durations)

            if evaluation < best_evaluation:
                best_route = np.array(route, copy=True)
                best_evaluation = evaluation

            # if (i+1) % iterations_per_update == 0:
            #     progress += 10
            #     print(f"Brute force {progress}% complete: {i+1}/{n_routes}.
            #     Best evaluation: {best_evaluation}")

        return best_route

    @staticmethod
    def greedy_routing(num_locations: int,
                       num_days: int,
                       graph: ndarray,
                       durations: ndarray):
        """
        A very simple solver for use with clustering, will not find a valid
        route on its own. This is a conventional TSP solver.
        :param num_locations: The number of locations in the route.
        :param num_days: The number of days in the route.
        :param graph: The adjacency matrix for the graph.
        :param durations: Duration spent at each location.
        :return: 1D np array representing the route found.
        """
        route_length = num_locations
        route = np.empty(route_length, dtype=int)
        index = 0
        working_graph = np.array(graph, copy=True)
        for i in range(route_length):
            working_graph[:, index] = np.iinfo(np.int32).max
            index = working_graph[index].argmin()
            route[i] = index

        if num_days > 1:
            new_locations = np.zeros(num_days-1)
            route = Routing.greedy_insertion(route, new_locations, graph,
                                             durations)

        return route

    @staticmethod
    def greedy_insertion(route: ndarray,
                         new_locations: ndarray,
                         graph: ndarray,
                         durations: ndarray) -> ndarray:
        """
        Iteratively inserts new locations into the route by finding the best
        insertion point at each iteration.
        :param route: 1D ndarray representing the current route.
        :param new_locations: 1D ndarray containing locations to insert.
        :param graph: The adjacency matrix for the graph.
        :param durations: Duration spent at each location in minutes.
        :return: 1D ndarray representing the updated route with new locations.
        """
        num_days = np.count_nonzero(route == 0)
        num_days += np.count_nonzero(new_locations == 0)
        working_route = np.array(route, copy=True)

        # For each new location, find the best insertion point in the route

        for location in new_locations:
            best_route = np.insert(working_route, 0, location)
            best_evaluation, _, _ = Routing.evaluate_route(best_route, num_days,
                                                           graph, durations)

            for i in range(1, len(working_route) + 1):
                new_route = np.insert(working_route, i, location)
                evaluation, _, _ = Routing.evaluate_route(new_route, num_days,
                                                         graph, durations)

                if evaluation < best_evaluation:
                    best_route = new_route
                    best_evaluation = evaluation

            working_route = best_route

        return working_route
