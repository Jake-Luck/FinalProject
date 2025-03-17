"""
Handles the creation and use of data within the project.
"""
from core import hidden

from enum import Enum
import json
import h5py
import math
import numpy as np
from numpy import ndarray  # For type hints
import openrouteservice
import random
import requests
import time

NUMBER_OF_NODES = 25  # Max allowed by ORS api


class DataGroups(Enum):
    """
    Provides values to access groups within h5 data file.
    """
    graphs = 'graphs'
    algorithm_performance = 'algorithm_performance'


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
        while True:
            datum = self.generate_test_datum()
            if not isinstance(datum, int):
                self.save_test_datum(datum, DataGroups.graphs)
            if datum == 1:
                print("Waiting 1 minute for api.")
                time.sleep(60)  # Wait 1 minute before trying again
                print("Continuing...")
            if datum == 2:
                print("Waiting 24hrs for api.")
                time.sleep(86400)
                print("Restarting...")

    def create_graph(self,
                     coordinates: ndarray) -> ndarray | int:
        """
        Takes a set of coordinates and converts them into a complete digraph,
        with each edge being the time taken to travel by car.

        Time taken is gathered using openrouteservice API.

        :param coordinates: ndarray of coordinates, shape=(num_locations, 2).
        :return: Graph as an ndarray, shape=(num_locations, num_locations).
        If API call fails an int will be returned.
        """
        graph = self.openrouteservice_api_call(coordinates)
        if isinstance(graph, int):
            return graph

        # Convert graph to minutes
        graph = np.ceil(graph / 60)

        # Sets cost of travel to self to maximum possible value
        np.fill_diagonal(graph, np.finfo(np.float32).max)

        return graph

    def generate_test_datum(self,
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
            coordinates = self.generate_coordinates(centre)
        if durations is None:
            durations = self.generate_durations()
        graph = self.create_graph(coordinates)

        if isinstance(graph, int):
            return graph

        return [graph, coordinates, durations]

    def openrouteservice_api_call(self,
                                  coordinate_array: ndarray) -> ndarray | int:
        """
        Makes a POST request to the OpenRouteService API to retrieve a distance
        matrix.

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
            json=body,
            headers=headers)

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

    @staticmethod
    def generate_coordinates(centre: ndarray) -> ndarray:
        """
        Randomly generates N latitude & longitude coordinate pairs around a
        central point.

        :param centre: Central point around which to generate coordinates.
        :return: Coordinates as a 2d array
        """

        coordinate_offsets_from_centre = np.random.uniform(
            -0.1, 0.1, (NUMBER_OF_NODES, 2))

        # Add centre to offsets to produce coordinates
        coordinates = coordinate_offsets_from_centre + centre

        return np.round(coordinates, 4)

    @staticmethod
    def generate_durations() -> ndarray:
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
    def save_test_datum(datum: list[ndarray],
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

