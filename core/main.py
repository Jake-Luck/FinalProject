"""
Program launching point
"""
import csv
from time import perf_counter

from algorithms.routing import Routing, GeneticRouting
from algorithms.clustering import KMeans, GeneticClustering, GeneticCentroidClustering, Clustering
from core.algorithm_shorthands import Shorthands
from core.data_handling import DataHandling

import numpy as np
import threading
from concurrent.futures import ProcessPoolExecutor

import h5py
from h5py import Group

def test_brute_force_trip_gen(name, graph, coordinates, durations, num_locations, num_days, seed):
    if num_locations + num_days - 1 > 10:
        # 10 takes around 150s
        n = 11
        for i in range(12, num_locations + num_days + 1):
            n *= i
        time_taken = 150 * n
        evaluation = float('inf')
    else:
        time_start = perf_counter()
        route = Routing.brute_force(num_locations, num_days, graph, durations)
        time_end = perf_counter()
        time_taken = time_end - time_start
        evaluation, _, _ = Routing.evaluate_route(route, num_days, graph, durations)

    return [['BFTG',name,num_locations,num_days,evaluation,time_taken, seed]]

def test_genetic_trip_gen(name, graph, coordinates, durations, num_locations, num_days, genetic_routing, seed):
    time_start = perf_counter()
    route = genetic_routing.find_route(num_locations, num_days, graph, durations)
    time_end = perf_counter()
    time_taken = time_end - time_start
    evaluation, _, _ = Routing.evaluate_route(route, num_days, graph, durations)

    return [['GATG', name, num_locations, num_days, evaluation, time_taken, seed]]

def test_greedy_insertion_trip_gen(name, graph, coordinates, durations, num_locations, num_days, seed):
    new_locations = np.concatenate((np.arange(1, num_locations), np.zeros(num_days)))
    time_start = perf_counter()
    route = Routing.greedy_insertion(np.array([]), new_locations, graph, durations)
    time_end = perf_counter()
    time_taken = time_end - time_start
    evaluation, _, _ = Routing.evaluate_route(route, num_days, graph, durations)

    return [['GITG', name, num_locations, num_days, evaluation, time_taken, seed]]

def test_kmeans(name, graph, coordinates, durations, num_locations, num_days, k_means, seed):
    k_means_start = perf_counter()
    clusters = k_means.find_clusters(coordinates, num_days, num_locations)
    k_means_end = perf_counter()
    k_means_time = k_means_end - k_means_start

    results = []
    greedy_start = perf_counter()
    greedy_route = k_means.find_route_from_clusters(clusters, num_days, Clustering.RoutingMethods.GREEDY, graph, durations, coordinates)
    greedy_end = perf_counter()
    greedy_time = k_means_time + (greedy_end - greedy_start)
    greedy_evaluation, _, _ = k_means.evaluate_route(greedy_route, num_days, graph, durations)
    results.append(['KM+GR', name, num_locations, num_days, greedy_evaluation, greedy_time, seed])

    insertion_start = perf_counter()
    insertion_route = k_means.find_route_from_clusters(clusters, num_days, Clustering.RoutingMethods.GREEDY_INSERTION, graph, durations, coordinates)
    insertion_end = perf_counter()
    insertion_time = k_means_time + (insertion_end - insertion_start)
    insertion_evaluation, _, _ = k_means.evaluate_route(insertion_route, num_days, graph, durations)
    results.append(['KM+GI',name, num_locations,num_days,insertion_evaluation,insertion_time, seed])

    counts = np.bincount(clusters)
    biggest_cluster = np.max(counts)
    if biggest_cluster > 10:
        n = 11
        for i in range(12, biggest_cluster+1):
            n *= i
        bf_time = 150 * n
        bf_evaluation = float('inf')
    else:
        bf_start = perf_counter()
        bf_route = k_means.find_route_from_clusters(clusters, num_days, Clustering.RoutingMethods.BRUTE_FORCE, graph, durations, coordinates)
        bf_end = perf_counter()
        bf_time = k_means_time + (bf_end - bf_start)
        bf_evaluation, _, _ = k_means.evaluate_route(bf_route, num_days, graph, durations)
    results.append(['KM+BF',name,num_locations, num_days, bf_evaluation, bf_time, seed])

    hull_start = perf_counter()
    hull_route = k_means.find_route_from_clusters(clusters, num_days, Clustering.RoutingMethods.CONVEX_HULL, graph, durations, coordinates)
    hull_end = perf_counter()
    hull_time = k_means_time + (hull_end - hull_start)
    hull_evaluation, _, _ = k_means.evaluate_route(hull_route, num_days, graph, durations)
    results.append(['KM+CH',name,num_locations,num_days,hull_evaluation,hull_time,seed])

    genetic_start = perf_counter()
    genetic_route = k_means.find_route_from_clusters(clusters, num_days, Clustering.RoutingMethods.GENETIC, graph, durations, coordinates)
    genetic_end = perf_counter()
    genetic_time = k_means_time + (genetic_end - genetic_start)
    genetic_evaluation, _, _ = k_means.evaluate_route(genetic_route, num_days, graph, durations)
    results.append(['KM+GR',name,num_locations,num_days,genetic_evaluation,genetic_time,seed])

    return results

def test_genetic_clustering_greedy_routing(name, graph, coordinates, durations, num_locations, num_days, genetic_clustering, seed):
    clustering_start = perf_counter()
    clusters = genetic_clustering.find_clusters(graph, durations, num_locations, num_days, Clustering.RoutingMethods.GREEDY, coordinates)
    clustering_end = perf_counter()
    clustering_time = clustering_end - clustering_start

    results = []
    greedy_start = perf_counter()
    route = genetic_clustering.find_route_from_clusters(clusters,num_days, Clustering.RoutingMethods.GREEDY, graph, durations, coordinates)
    greedy_end = perf_counter()
    greedy_time = clustering_time + (greedy_end - greedy_start)
    greedy_evaluation, _, _ = genetic_clustering.evaluate_route(route, num_days, graph, durations)
    results.append(['GAC+GR',name,num_locations,num_days,greedy_evaluation,greedy_time, seed])

    counts = np.bincount(clusters)
    biggest_cluster = np.max(counts)
    if biggest_cluster > 10:
        n = 11
        for i in range(12, biggest_cluster+1):
            n *= i
        bf_time = 150 * n
        bf_evaluation = float('inf')
    else:
        bf_start = perf_counter()
        bf_route = genetic_clustering.find_route_from_clusters(clusters, num_days, Clustering.RoutingMethods.BRUTE_FORCE, graph, durations, coordinates)
        bf_end = perf_counter()
        bf_time = clustering_time + (bf_end - bf_start)
        bf_evaluation, _, _ = genetic_clustering.evaluate_route(bf_route, num_days, graph, durations)
    results.append(['GAC+GR+BF',name,num_locations, num_days, bf_evaluation, bf_time, seed])

    return results

def test_genetic_clustering_greedy_insertion(name, graph, coordinates, durations, num_locations, num_days, genetic_clustering, seed):
    clustering_start = perf_counter()
    clusters = genetic_clustering.find_clusters(graph, durations, num_locations, num_days, Clustering.RoutingMethods.GREEDY_INSERTION, coordinates)
    clustering_end = perf_counter()
    clustering_time = clustering_end - clustering_start

    route = genetic_clustering.find_route_from_clusters(clusters,num_days, Clustering.RoutingMethods.GREEDY_INSERTION, graph, durations, coordinates)
    evaluation, _, _ = genetic_clustering.evaluate_route(route, num_days, graph, durations)

    return [['GAC+GI',name,num_locations,num_days,evaluation,clustering_time, seed]]

def test_genetic_clustering_convex_hull(name, graph, coordinates, durations, num_locations, num_days, genetic_clustering, seed):
    clustering_start = perf_counter()
    clusters = genetic_clustering.find_clusters(graph, durations, num_locations, num_days, Clustering.RoutingMethods.CONVEX_HULL, coordinates)
    clustering_end = perf_counter()
    clustering_time = clustering_end - clustering_start

    route = genetic_clustering.find_route_from_clusters(clusters,num_days, Clustering.RoutingMethods.CONVEX_HULL, graph, durations, coordinates)
    evaluation, _, _ = genetic_clustering.evaluate_route(route, num_days, graph, durations)

    return [['GAC+CH',name,num_locations,num_days,evaluation,clustering_time, seed]]

def test_genetic_centroids_greedy_routing(name, graph, coordinates, durations, num_locations, num_days, genetic_centroids, seed):
    clustering_start = perf_counter()
    clusters = genetic_centroids.find_clusters(coordinates, graph, durations, num_days, Clustering.RoutingMethods.GREEDY)
    clustering_end = perf_counter()
    clustering_time = clustering_end - clustering_start

    results = []
    greedy_start = perf_counter()
    route = genetic_centroids.find_route_from_clusters(clusters,num_days, Clustering.RoutingMethods.GREEDY, graph, durations, coordinates)
    greedy_end = perf_counter()
    greedy_time = clustering_time + (greedy_end - greedy_start)
    greedy_evaluation, _, _ = genetic_centroids.evaluate_route(route, num_days, graph, durations)
    results.append(['GACC+GR',name,num_locations,num_days,greedy_evaluation,greedy_time, seed])

    counts = np.bincount(clusters)
    biggest_cluster = np.max(counts)
    if biggest_cluster > 10:
        n = 11
        for i in range(12, biggest_cluster + 1):
            n *= i
        bf_time = 150 * n
        bf_evaluation = float('inf')
    else:
        bf_start = perf_counter()
        bf_route = genetic_centroids.find_route_from_clusters(clusters, num_days, Clustering.RoutingMethods.BRUTE_FORCE, graph, durations, coordinates)
        bf_end = perf_counter()
        bf_time = clustering_time + (bf_end - bf_start)
        bf_evaluation, _, _ = genetic_centroids.evaluate_route(bf_route, num_days, graph, durations)
    results.append(['GACC+GR+BF', name, num_locations, num_days, bf_evaluation, bf_time, seed])

    return results

def test_genetic_centroids_greedy_insertion(name, graph, coordinates, durations, num_locations, num_days, genetic_centroids, seed):
    clustering_start = perf_counter()
    clusters = genetic_centroids.find_clusters(coordinates, graph, durations, num_days, Clustering.RoutingMethods.GREEDY_INSERTION)
    clustering_end = perf_counter()
    clustering_time = clustering_end - clustering_start

    route = genetic_centroids.find_route_from_clusters(clusters,num_days, Clustering.RoutingMethods.GREEDY_INSERTION, graph, durations, coordinates)
    evaluation, _, _ = genetic_centroids.evaluate_route(route, num_days, graph, durations)

    return [['GACC+GI', name,num_locations,num_days,evaluation,clustering_time, seed]]

def test_genetic_centroids_convex_hull(name, graph, coordinates, durations, num_locations, num_days, genetic_centroids, seed):
    clustering_start = perf_counter()
    clusters = genetic_centroids.find_clusters(coordinates, graph, durations, num_days, Clustering.RoutingMethods.CONVEX_HULL)
    clustering_end = perf_counter()
    clustering_time = clustering_end - clustering_start

    route = genetic_centroids.find_route_from_clusters(clusters,num_days, Clustering.RoutingMethods.CONVEX_HULL, graph, durations, coordinates)
    evaluation, _, _ = genetic_centroids.evaluate_route(route, num_days, graph, durations)

    return [['GACC+CH', name,num_locations,num_days,evaluation,clustering_time, seed]]

def test_brute_force_routing(name, graph, coordinates, durations, num_locations, num_days, seed):
    if num_locations - 1 > 10:
        # 10 takes around 150s
        n = 11
        for i in range(12, num_locations + num_days + 1):
            n *= i
        time_taken = 150 * n
        evaluation = float('inf')
    else:
        time_start = perf_counter()
        route = Routing.brute_force(num_locations, 1, graph, durations)
        time_end = perf_counter()
        time_taken = time_end - time_start
        new_locations = np.zeros(num_days-1)
        updated_route = Routing.greedy_insertion(route, new_locations, graph, durations)
        evaluation, _, _ = Routing.evaluate_route(updated_route, num_days, graph, durations)

    return [['BF+GI',name,num_locations,num_days,evaluation,time_taken, seed]]

def test_greedy_routing(name, graph, coordinates, durations, num_locations, num_days, seed):
    time_start = perf_counter()
    route = Routing.greedy_routing(num_locations, 1, graph, durations)
    time_end = perf_counter()
    time_taken = time_end - time_start
    new_locations = np.zeros(num_days-1)
    updated_route = Routing.greedy_insertion(route, new_locations, graph, durations)
    evaluation, _, _ = Routing.evaluate_route(updated_route, num_days, graph, durations)

    return [['GR+GI',name,num_locations,num_days,evaluation,time_taken,seed]]

def test_hull_routing(name, graph, coordinates, durations, num_locations, num_days, seed):
    time_start = perf_counter()
    route = Routing.gift_wrapping(num_locations, 1, coordinates, graph, durations)
    time_end = perf_counter()
    time_taken = time_end - time_start
    new_locations = np.zeros(num_days-1)
    updated_route = Routing.greedy_insertion(route, new_locations, graph, durations)
    evaluation, _, _ = Routing.evaluate_route(updated_route, num_days, graph, durations)

    return [['CH+GI',name,num_locations,num_days,evaluation,time_taken,seed]]

def test_genetic_routing(name, graph, coordinates, durations, num_locations, num_days, genetic_routing, seed):
    time_start = perf_counter()
    route = genetic_routing.find_route(num_locations, 1, graph, durations, coordinates)
    time_end = perf_counter()
    time_taken = time_end - time_start
    new_locations = np.zeros(num_days - 1)
    updated_route = Routing.greedy_insertion(route, new_locations, graph,
                                             durations)
    evaluation, _, _ = Routing.evaluate_route(updated_route, num_days,
                                              graph, durations)

    return [['GAR+GI', name, num_locations, num_days, evaluation, time_taken, seed]]

def test_algorithms():
    with h5py.File('data/training_data.h5', 'a') as f:
        group_items = f['test_graphs'].items()
        graph_data = [(graph[0], graph[1][:], graph[1].attrs['coordinates'][:],
                      graph[1].attrs['durations'][:]) for graph in group_items]

    k_means = KMeans()
    genetic_clustering = GeneticClustering(150, 50, 0.9, 0.1, None, False, 0, False)
    genetic_centroid_clustering = GeneticCentroidClustering(150, 50, 0.9, 0.1, None, False, 0, False)
    genetic_routing = GeneticRouting(150, 50, 0.9, 0.4, None, False, 0, False)

    test_sets = [
        # (25, 7), (20, 6), (15, 5), (10, 4), (8, 3), (5, 2),
        # (25, 6), (20, 5), (15, 4), (10, 3), (8, 2),
        # (25, 5), (20, 4), (15, 3), (10, 2),
        # (25, 4), (20, 3), (15, 2),
    ]
    test_start = 0
    already_done = [
        'addis_ababa','amsterdam','auckland','barcelona','berlin', 'birmingham',
        'buenos_aires','dublin', 'johannesburg','lagos','london','medina',
        #'mumbai','nairobi','new_york','paris', 'rio_de_janeiro','riyadh','rome',
        #'san_francisco','shanghai','sydney','taipei',
        #'tokyo','vancouver'
    ]
    for seed in range(100):
        np.random.seed(seed)
        for test_set in test_sets:
            test_set_test_start = perf_counter()
            for graph_datum in graph_data:
                if test_set == (15, 2) and graph_datum[0] in already_done:
                    continue
                location_test_start = perf_counter()
                num_locations = test_set[0]
                num_days = test_set[1]
                name = graph_datum[0]

                chosen_indices = np.arange(1, 25)
                if num_locations != 25:
                    chosen_indices = np.random.choice(chosen_indices, num_locations-1, False)
                chosen_indices = np.insert(chosen_indices, 0, 0)
                graph = graph_datum[1][np.ix_(chosen_indices, chosen_indices)]
                coordinates = graph_datum[2][chosen_indices]
                durations = graph_datum[3][chosen_indices]

                futures = []
                with ProcessPoolExecutor(max_workers=6) as executor:

                    futures.append(
                        executor.submit(
                            test_brute_force_trip_gen, name, graph, coordinates, durations, num_locations, num_days, seed
                        )
                    )
                    futures.append(
                        executor.submit(
                            test_genetic_trip_gen, name, graph, coordinates, durations, num_locations, num_days, genetic_routing, seed
                        )
                    )
                    futures.append(
                        executor.submit(
                            test_greedy_insertion_trip_gen, name, graph, coordinates, durations, num_locations, num_days, seed
                        )
                    )
                    futures.append(
                        executor.submit(
                            test_kmeans, name, graph, coordinates, durations, num_locations, num_days, k_means, seed
                        )
                    )
                    futures.append(
                        executor.submit(
                            test_genetic_clustering_greedy_routing, name, graph, coordinates, durations, num_locations, num_days, genetic_clustering, seed
                        )
                    )
                    futures.append(
                        executor.submit(
                            test_genetic_clustering_greedy_insertion, name, graph, coordinates, durations, num_locations, num_days, genetic_clustering, seed
                        )
                    )
                    futures.append(
                        executor.submit(
                            test_genetic_clustering_convex_hull, name, graph, coordinates, durations, num_locations, num_days, genetic_clustering, seed
                        )
                    )
                    futures.append(
                        executor.submit(
                            test_genetic_centroids_greedy_routing, name, graph, coordinates, durations, num_locations, num_days, genetic_centroid_clustering, seed
                        )
                    )
                    futures.append(
                        executor.submit(
                            test_genetic_centroids_greedy_insertion, name, graph, coordinates, durations, num_locations, num_days, genetic_centroid_clustering, seed
                        )
                    )
                    futures.append(
                        executor.submit(
                            test_brute_force_routing, name, graph, coordinates, durations, num_locations, num_days, seed
                        )
                    )
                    futures.append(
                        executor.submit(
                            test_greedy_routing, name, graph, coordinates, durations, num_locations, num_days, seed
                        )
                    )
                    futures.append(
                        executor.submit(
                            test_hull_routing, name, graph, coordinates, durations, num_locations, num_days, seed
                        )
                    )
                    futures.append(
                        executor.submit(
                            test_genetic_routing, name, graph, coordinates, durations, num_locations, num_days, genetic_routing, seed
                        )
                    )

                with open("data/results.csv", "a", newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    for future in futures:
                        result = future.result()
                        if result:
                            for row in result:
                                writer.writerow(row)

                location_test_end = perf_counter()
                print(f"{name}, {num_locations} locations, {num_days} days, finished in {location_test_end - location_test_start:.2f}s")
            test_set_test_end = perf_counter()
            print(f"All cities for {num_locations} locations, {num_days} days, finished in {test_set_test_end - test_set_test_start:.2f}s")




def main():
    """
    Launching point of the program, always starts a thread for collecting data.
    """
    with h5py.File('data/training_data.h5', 'a') as f:
        coordinates = f['test_graphs']['berlin'].attrs['coordinates']

    from core.plotting import Plotting
    Plotting.display_coordinates(coordinates, save_plot=True)

    return

    test_algorithms()

    return

    data_handler = DataHandling()
    data_collection_thread = threading.Thread(
        target=data_handler.collect_test_data)
    data_collection_thread.start()

    data_collection_thread.join()

    return

    num_locations = 6
    num_days = 2
    seed = 0
    np.random.seed(seed)
    graph, coordinates = DataHandling.get_random_datum(seed)
    durations = np.random.randint(1, 96, num_locations) * 15
    durations[0] = 0
    graph = graph[:num_locations, :num_locations]
    coordinates = coordinates[:num_locations]

    Shorthands.gift_wrapping(num_locations, num_days, graph, durations,
                             coordinates)
    Shorthands.brute_force(
        num_locations, num_days, graph[:num_locations, :num_locations],
        durations[:num_locations], coordinates[:num_locations])
    Shorthands.greedy(num_locations, num_days, graph, durations, coordinates)
    Shorthands.k_means(num_locations, num_days, graph, durations, coordinates)
    Shorthands.k_means(num_locations, num_days, graph, durations, coordinates,
                       KMeans.RoutingMethods.BRUTE_FORCE)
    Shorthands.genetic_clustering(
        num_locations, num_days, graph, durations, coordinates)
    Shorthands.genetic_centroid_clustering(
        num_locations, num_days, graph, durations, coordinates)
    Shorthands.genetic_routing(
        num_locations, num_days, graph, durations, coordinates)
    return
    data_collection_thread.join()


if __name__ == '__main__':
    main()
