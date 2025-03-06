import numpy as np
from numpy import ndarray  # For type hints
from enum import Enum

class Algorithm():
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
        BRUTE_FORCE = 0
        'Compares every possible route to find the best.'
        GREEDY = 1
        'Always chooses the shortest path from any given position.'

        def __call__(self, *args, **kwargs):
            return self.value(*args, **kwargs)

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
