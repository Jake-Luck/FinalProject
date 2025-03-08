"""Program launching point"""
from core.data_handling import DataHandling

import threading
# todo: Finish this
"""
def test_bruteforce_timings() -> None:
    with h5py.File('data/training_data.h5', 'a') as f:
        results_group = f[data_handling.DataGroups.algorithm_performance.value]
        _graphs = f[data_handling.DataGroups.graphs.value]
        for i in range(2, 25):
            for j in range(i-1):

                _graph = _graphs[str(i)]
                _num_locations = _graph.shape[0]
                _num_days = j
                np_graph = _graph[:]

                _routing = Routing()
                perf_start = perf_counter()
                _route = _routing.brute_force(_num_locations, _num_days, 
                                              np_graph)
                perf_end = perf_counter()

                print(
                    f"Bruteforce, {_num_locations} locations, {_num_days} "
                    f"days: {perf_end - perf_start} seconds. Route: {_route}")

        # Store results
"""
def main():
    data_handler = DataHandling()
    data_collection_thread = threading.Thread(
        target=data_handler.collect_test_data())
    data_collection_thread.start()
    data_collection_thread.join()
"""_
    with h5py.File('data/training_data.h5', 'r') as f:
        graphs = f[utilities.DataGroups.ordered_graphs.value]
        graph = np.array(graphs['12'], copy=True, dtype=int)
        coordinates = graphs['12'].attrs['coordinates'][:]

    num_days = 7
    num_locations = 25
    route_length = 31

    routing = Routing()
    route = routing.brute_force(num_locations, num_days, graph)

        clustering_algorithm = genetic.GeneticCentroidClustering(
            1000, 30, 0.9, 0.1, 100, False)
        cluster_assignments = clustering_algorithm.find_clusters(
            coordinates, graph, num_days,
            clustering_algorithm.RoutingMethods.GREEDY)

        clustering_algorithm = genetic.GeneticClustering(
            1000, 30, 0.9, 0.1, 100, False)
        cluster_assignments = clustering_algorithm.find_clusters(
            graph, num_locations, num_days, route_length,
            clustering_algorithm.RoutingMethods.GREEDY)

    clustering_algorithm = clustering.KMeans()
    cluster_assignments = clustering_algorithm.find_clusters(
        coordinates, num_days, num_locations)

    route = clustering_algorithm.find_route_from_cluster_assignments(
        cluster_assignments, num_days,
        clustering_algorithm.RoutingMethods.BRUTE_FORCE, graph)

    #plotting.display_route(route, coordinates)
"""


if __name__ == '__main__':
    main()
