"""
Handles the creation and use of data within the project.
"""
from algorithms.clustering import KMeans
from algorithms.genetic import GeneticClustering, GeneticCentroidClustering
from algorithms.routing import Routing
from core import hidden

import csv
from enum import Enum
import json
import h5py
import math
import numpy as np
from numpy import ndarray  # For type hints
import openrouteservice
import random
import requests
from time import perf_counter, sleep

NUMBER_OF_NODES = 25  # Max allowed by ORS api


class DataGroups(Enum):
    """
    Provides values to access groups within h5 data file.
    """
    graphs = 'graphs'
    poi_graphs = 'point_of_interest_graphs'


class DataAttributes(Enum):
    """
    Provides values to access attributes within h5 data file.
    """
    coordinates = 'coordinates'
    durations = 'durations'


class DataHandling:
    """
    Provides various methods for creating/handling data. Uses hidden api key to
    make calls to OpenRouteService.
    """
    def __init__(self):
        """
        Sets up access to OpenRouteService API.
        """
        self.api_key = hidden.apikey
        self.client = openrouteservice.Client(key=self.api_key)

    def collect_test_data(self) -> None:
        """
        Permanently loops, accessing the OpenRouteService API as often as
        possible to collect data and save it to the project database.
        """
        i = 0
        while i < 2500:
            try:
                datum = self._generate_test_datum()
            except ValueError:
                print("Unexpected response from api")
                continue
            if not isinstance(datum, int):
                self._save_test_datum(datum, DataGroups.poi_graphs)
            elif datum == 2:
                print("Reached daily API limit, exiting.")
                return
            i += 1
            if i % 60 == 0:
                print("Waiting 1 minute for api.")
                sleep(60)  # Wait 1 minute before trying again
                print("Continuing...")

    def _generate_coordinates(self,
                              centre: ndarray) -> ndarray:
        """
        Randomly generates N latitude & longitude coordinate pairs around a
        central point.

        :param centre: Central point around which to generate coordinates.
        :return: Coordinates as a 2d array
        """
        # Loop with reducing boundary size, in case boundary size is too big
        # for api call, which will time out and return 0
        boundary = 2000
        while boundary >= 250:
            coordinates = self._ors_poi_call(centre)
            if isinstance(coordinates, ndarray):
                return coordinates.round(4)
            if coordinates == 1 or coordinates == 2:
                return coordinates
            boundary /= 2
        return coordinates # Will return 0


    def _generate_graph(self,
                        coordinates: ndarray) -> ndarray | int:
        """
        Takes a set of coordinates and converts them into a complete digraph,
        with each edge being the time taken to travel by car.

        Time taken is gathered using openrouteservice API.

        :param coordinates: ndarray of coordinates, shape=(num_locations, 2).
        :return: Graph as an ndarray, shape=(num_locations, num_locations).
        If API call fails an int will be returned.
        """
        graph = self._ors_matrix_call(coordinates)
        if isinstance(graph, int):
            return graph

        # Convert graph to minutes
        graph = np.ceil(graph / 60)

        # Sets cost of travel to self to maximum possible value
        np.fill_diagonal(graph, np.finfo(np.float32).max)

        return graph

    def _generate_test_datum(self,
                             coordinates: ndarray = None,
                             centre: ndarray = None,
                             durations: ndarray = None) -> ndarray | int:
        """
        Generates a new training datum, using given values or randomly
        generated ones.

        :param coordinates: Used to generate graph (if none given).
        :param centre: Starting point, i.e., hotel. used to generate coords
        Only used if no coordinates are given.
        :param durations: Amount of time spent at each location, in minutes.
        :return: Returns a new training datum, in the form [graph, coordinates]
        """
        if coordinates is None:
            if centre is None:
                city_coordinates = np.load('data/city_coordinates.npz')[
                    'coordinates']
                index = random.randrange(len(city_coordinates))
                centre = city_coordinates[index]
            coordinates = self._generate_coordinates(centre)
            if isinstance(coordinates, int):
                return coordinates
        if durations is None:
            durations = self._generate_durations()
        graph = self._generate_graph(coordinates)

        if isinstance(graph, int):
            return graph

        return [graph, coordinates, durations]

    def _ors_matrix_call(self,
                         coordinate_array: ndarray) -> ndarray | int:
        """
        Makes a POST request to the OpenRouteService API to retrieve a distance
        matrix for a unilateral digraph between given coordinates.

        :param coordinate_array: List of coordinates to include in api call.
        :return: A 2d list representing the distance matrix.
        """
        body = {"locations": coordinate_array.tolist()}
        headers = {
            'Accept': 'application/json, application/geo+json, '
                      'application/gpx+xml, img/png; charset=utf-8',
            'Authorization': self.api_key,
            'Content-Type': 'application/json; charset=utf-8'
        }
        call = requests.post(
            url='https://api.openrouteservice.org/v2/matrix/foot-walking',
            json=body, headers=headers
        )

        response_json = json.loads(call.text)

        # Check that API didn't return an error.
        if 'error' in response_json:
            if response_json['error'] == 'Rate limit exceeded':
                print("Reached max api calls for a minute.")
                return 1  # Maxed out api calls
            elif response_json['error'] == 'Quota exceeded':
                print("Reached max api calls for the day.")
                return 2
            else:
                raise ValueError("Invalid response from API call\n"
                                 + response_json['error'])

        # Take the graph from the json response.
        graph = response_json['durations']

        # If any edges are NaN or None, graph is unusable, return 0
        if any(path is None or math.isnan(path) for path in
               [item for row in graph for item in row]):
            return 0

        return np.array(graph)

    def _ors_poi_call(self,
                      centre: ndarray) -> ndarray | int:
        """
        Makes a POST request to the OpenRouteService API to retrieve points of
        interest around a given coordinate.
        :param centre: The central coordinate, typically a city centre.
        :return: A 2d array of coordinates for points of interest.
        """
        body = {
            "request": "pois", "geometry": {
                "buffer": 2000,
                "geojson": {"type": "Point", "coordinates": centre.tolist()}
            },
            "filters": {"category_group_ids": [130, 220, 260, 420, 560]},
            "limit": 25
        }
        headers = {
            'Accept': 'application/json, application/geo+json, '
                      'application/gpx+xml, img/png; charset=utf-8',
            'Authorization': self.api_key,
            'Content-Type': 'application/json; charset=utf-8'
        }
        call = requests.post(
            url='https://api.openrouteservice.org/pois', json=body,
            headers=headers
        )

        response_json = json.loads(call.text)

        # Check that API didn't return an error.
        if 'error' in response_json:
            match response_json['error']:
                case 'Rate limit exceeded':
                    print("Reached max api calls for a minute.")
                    return 1
                case 'Quota exceeded':
                    print("Reached max api calls for the day.")
                    return 2
                case 'There was a problem proxying the request':
                    print("API call timed out.")
                    return 0
                case _:
                    raise ValueError(f"Invalid response from API call\n"
                                     f"{response_json['error']})")

        coordinates = [point['geometry']['coordinates']
                       for point in response_json['features']]
        if len(coordinates) == 0:
            "Api returned 0 locations"
            return 0
        coordinates.insert(0, centre)

        return np.array(coordinates)

    @staticmethod
    def collect_results() -> None:
        """
        Loops through saved graphs to test different algorithms on the same
        input. Saves results to csv file.
        """
        with h5py.File('data/training_data.h5', 'a') as f:
            graphs = f['graphs']
            kmeans = KMeans()
            genetic_clustering = GeneticClustering(100, 10, 0.9, 0.1, 0, False)
            genetic_centroid_clustering = GeneticCentroidClustering(100, 10,
                                                                    0.9, 0.1, 0,
                                                                    False)
            for i in range(1, len(graphs)):
                datum = graphs[str(i)]
                graph = np.array(datum)
                coordinates = np.array(datum.attrs['coordinates'])
                durations = np.random.randint(1, 96, 25) * 15
                durations[0] = 0
                new_rows = DataHandling._loop_through_locations_and_days(
                    graph, durations, coordinates, kmeans, genetic_clustering,
                    genetic_centroid_clustering)

                with open('data/results.csv', 'a', encoding='utf-32') as csvf:
                    csvwriter = csv.writer(csvf)
                    csvwriter.writerows(new_rows)
                print(f"Graph {i} saved to csv.")

    @staticmethod
    def get_random_datum(seed: int | None = None) -> tuple:
        """
        Gets a random graph and its coordinates.
        :seed: Random number generator seed.
        :return: A tuple (graph, coordinates).
        """
        if seed is not None:
            random.seed(seed)
        with h5py.File('data/training_data.h5', 'r') as f:
            graphs = f[DataGroups.graphs.value]
            chosen_index = random.randrange(len(graphs))
            graph_data = graphs[str(chosen_index)]
            chosen_graph = np.array(graph_data, copy=True, dtype=float)
            chosen_coordinates = np.array(
                graph_data.attrs[DataAttributes.coordinates.value],
                copy=True, dtype=float)

        return chosen_graph, chosen_coordinates

    @staticmethod
    def reset_database() -> None:
        """
        In case of emergency (I mess something up and corrupt the file),
        delete h5 file and run this.
        """
        city_coordinates = np.load('data/city_coordinates.npz')['coordinates']

        with h5py.File("data/training_data.h5", "w") as f:
            f.create_dataset("city_coordinates", data=city_coordinates)
            f.create_group(DataGroups.graphs.value)
            f.create_group(DataGroups.algorithm_performance.value)

    @staticmethod
    def _generate_durations() -> ndarray:
        """
        Generates random durations for each location.
        :return: ndarray storing durations for each location,
        shape=NUMBER_OF_NODES.
        """
        # Center doesn't have a duration, so is set to 0.
        # Other points are a random duration in 15 minute intervals
        durations = np.random.randint(1, 96, NUMBER_OF_NODES) * 15
        durations[0] = 0

        return durations

    @staticmethod
    def _loop_through_locations_and_days(graph,
                                         durations,
                                         coordinates,
                                         kmeans,
                                         genetic_clustering,
                                         genetic_centroid_clustering) -> list:
        """
        Loops through every possible input for num_locations and num_days
        with the given input.
        :param graph: The adjacency matrix for the graph.
        :param durations: Duration spent at each location.
        :param coordinates: Coordinates of each cluster, a 2D array with shape
        (num_coordinates, 3). Second dimension is (x, y, assigned_cluster)
        :param kmeans: Instance of kmeans class.
        :param genetic_clustering: Instance of genetic clustering class.
        :param genetic_centroid_clustering: Instance of genetic centroid
        clustering class.
        :return: Returns a list of results, to be saved as csv.
        """
        new_rows = []
        for n in range(4, graph.shape[0] + 1):
            for d in range(1, n):
                _graph = graph[:n, :n]
                _durations = durations[:n]
                _coordinates = coordinates[:n]
                # brute force
                if n + d < 11:
                    start = perf_counter()
                    route = Routing.brute_force(n, d, _graph,
                                                _durations)
                    end = perf_counter()
                    time = end - start
                    evaluation, _, _ = Routing.evaluate_route(
                        route, d, _graph, _durations)
                    new_rows.append(
                        ["Brute Force", n, d, evaluation, time,
                         "W * (1+σ)"])
                # greedy routing & greedy insertion
                start = perf_counter()
                route = Routing.greedy_routing(n, d, _graph, _durations)
                end = perf_counter()
                time = end - start
                evaluation, _, _ = Routing.evaluate_route(route, d,
                                                          _graph,
                                                          _durations)
                new_rows.append(
                    ["Greedy Routing & Greedy Insertion", n, d,
                     evaluation, time, "W * (1+σ)"])

                if d <= 1:  # No need to cluster if days is 1
                    continue

                start = perf_counter()
                clusters = kmeans.find_clusters(_coordinates, d, n)
                route = kmeans.find_route_from_clusters(
                    clusters, d, kmeans.RoutingMethods.GREEDY,
                    _graph, _durations)
                end = perf_counter()
                time = end - start
                evaluation, _, _ = Routing.evaluate_route(
                    route, d, _graph, _durations)
                new_rows.append(
                    ["K-Means & Greedy", n, d, evaluation, time,
                     "W * (1+\\sigma)"])
                # kmeans & brute force
                start = perf_counter()
                clusters = kmeans.find_clusters(_coordinates, d, n)
                counts = np.bincount(clusters)
                biggest_cluster = np.max(counts)
                if biggest_cluster < 8:
                    route = kmeans.find_route_from_clusters(
                        clusters, d,
                        kmeans.RoutingMethods.BRUTE_FORCE, _graph,
                        _durations)
                    end = perf_counter()
                    time = end - start
                    evaluation, _, _ = Routing.evaluate_route(
                        route, d, _graph, _durations)
                else:
                    end = perf_counter()
                    time = end - start
                    evaluation = float('inf')
                new_rows.append(
                    ["K-Means & Brute Force", n, d, evaluation,
                     time, "W * (1+\\sigma)"])
                # Genetic Clustering + Greedy
                start = perf_counter()
                clusters = genetic_clustering.find_clusters(
                    _graph, _durations, n, d, n + d - 1,
                    genetic_clustering.RoutingMethods.GREEDY)
                route = genetic_clustering.find_route_from_clusters(
                    clusters, d, genetic_clustering.RoutingMethods.GREEDY,
                    _graph, _durations)
                end = perf_counter()
                time = end - start
                evaluation, _, _ = Routing.evaluate_route(route, d,
                                                          _graph,
                                                          _durations)
                new_rows.append(
                    ["Genetic Clustering & Greedy Routing", n, d,
                     evaluation, time, "W * (1+\\sigma)"])
                # Genetic Centroid Clustering + Greedy
                start = perf_counter()
                clusters = genetic_centroid_clustering.find_clusters(
                    _coordinates, _graph, _durations, d,
                    genetic_centroid_clustering.RoutingMethods.GREEDY)
                route = genetic_centroid_clustering.find_route_from_clusters(
                    clusters, d,
                    genetic_clustering.RoutingMethods.GREEDY,
                    _graph, _durations)
                end = perf_counter()
                time = end - start
                evaluation, _, _ = Routing.evaluate_route(route, d,
                                                          _graph,
                                                          _durations)
                new_rows.append(
                    ["Genetic Centroid Clustering & Greedy Routing",
                     n, d, evaluation, time, "W * (1+\\sigma)"])
        return new_rows

    @staticmethod
    def _save_test_datum(datum: list[ndarray],
                         group: DataGroups):
        """
        Saves a training datum to the hdf5 file.

        :param datum: A tuple containing a graph and coordinates
        :param group: Which data group to save to.
        """
        with h5py.File("data/training_data.h5", 'a') as f:
            group = f[group.value]
            i = len(group)
            graph = group.create_dataset(str(i), compression="gzip",
                                         data=np.array(datum[0],
                                                       dtype=np.float64))

            graph.attrs[DataAttributes.coordinates.value] = datum[1]
            graph.attrs[DataAttributes.durations.value] = datum[2]
