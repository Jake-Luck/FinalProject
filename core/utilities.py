import random
import openrouteservice
import requests
import json
import h5py
import math
from enum import Enum
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from numpy import ndarray  # For type hints

from core import hidden

api_key = hidden.apikey
client = openrouteservice.Client(key=api_key)

NUMBER_OF_NODES = 25  # Max allowed by ORS api

def generate_coordinates(centre: ndarray) -> ndarray:
    """
    Randomly generates N latitude & longitude coordinate pairs around a
    central point.

    :param centre: Central point around which to generate coordinates.
    :return: Coordinates as a 2d array
    """

    coordinate_offsets_from_centre = np.random.uniform(-0.1, 0.1,
                                                       (NUMBER_OF_NODES, 2))

    # Add centre to offsets to produce coordinates
    coordinates = coordinate_offsets_from_centre + centre

    return np.round(coordinates, 4)


def generate_durations() -> ndarray:
    """
    Generates durations for each location and scales them down to ensure they
    don't exceed three quarters of all time available.

    :return: ndarray storing durations for each location, shape=NUMBER_OF_NODES
    """
    # 16 hours * 0.75, '* 0.75' to give leeway for travel time.
    free_time = 960 * 0.75

    # Center doesn't have a duration, so is set to 0.
    # Other points are a random duration in 15 minute intervals
    durations = np.random.randint(1, 96, NUMBER_OF_NODES) * 15
    durations[0] = 0

    return durations


def create_graph(coordinates: ndarray,
                 durations: ndarray) -> ndarray | int:
    """
    Takes a set of coordinates and converts them into a complete digraph, with
    each edge being the time taken to travel by car.

    Time taken is gathered using openrouteservice API.

    :param coordinates: ndarray of coordinates, shape=(num_locations, 2).
    :param durations: The amount of time to be taken at each location,
    shape=num_locations.
    :return: Graph as an ndarray, shape=(num_locations, num_locations). If API
    call fails an int will be returned.
    """
    graph = openrouteservice_api_call(coordinates)
    if isinstance(graph, int):
        return graph

    number_of_locations = len(graph)
    for i in range(number_of_locations):
        for j in range(number_of_locations):
            if i == j:
                graph[j, i] = np.finfo(np.float32).max
            # Converts graph to minutes before adding durations
            graph[j, i] = math.ceil((graph[j][i] / 60) + durations[i])

    return graph


def openrouteservice_api_call(coordinate_array: ndarray) -> ndarray | int:
    """
    Makes a POST request to the OpenRouteService API to retrieve a distance
    matrix.

    :param coordinate_array: List of coordinates to include in the api call.
    :return: A 2d list representing the distance matrix.
    """
    # For some reason ORS takes coordinates as longitude, latitude. So flip.
    body = {"locations": np.flip(coordinate_array).tolist()}
    headers = {
        'Accept': 'application/json, application/geo+json, '
                  'application/gpx+xml, img/png; charset=utf-8',
        'Authorization': api_key,
        'Content-Type': 'application/json; charset=utf-8'
    }
    call = requests.post(
        'https://api.openrouteservice.org/v2/matrix/foot-walking', json=body,
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


def generate_test_datum(coordinates: ndarray = None,
                        centre: ndarray = None,
                        durations: ndarray = None) -> ndarray | int:
    """
    Generates a new training datum, using given values or randomly generated.

    :param coordinates: Used to generate graph (if none given).
    :param centre: Starting point, i.e., hotel. (if no graph or coords given).
    :param durations: Amount of time spent at each location, in minutes.
    :return: Returns a new training datum, in the form [graph, coordinates]
    """
    if coordinates is None:
        if centre is None:
            city_coordinates = np.load('data/city_coordinates.npz')[
                'coordinates']
            index = random.randrange(len(city_coordinates))
            centre = city_coordinates[index]
        coordinates = generate_coordinates(centre)
    if durations is None:
        durations = generate_durations()
    graph = create_graph(coordinates, durations)

    if isinstance(graph, int):
        return graph

    return [graph, coordinates]


class DataGroups(Enum):
    regular_graphs = 'graphs'
    algorithm_performance = 'algorithm_performance'


class DataAttributes(Enum):
    coordinates = 'coordinates'


def save_test_datum(datum: tuple,
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
                                     data=np.array(datum[0], dtype=np.float64))

        graph.attrs[DataAttributes.coordinates.value] = datum[1]


def reset_database() -> None:
    """
    In case of emergency (I mess something up and corrupt the file),
    delete h5 file and run this.
    """
    city_coordinates = np.load('data/city_coordinates.npz')['coordinates']

    with h5py.File("data/training_data.h5", "w") as f:
        f.create_dataset("city_coordinates", data=city_coordinates)
        f.create_group(DataGroups.regular_graphs.value)
        f.create_group(DataGroups.algorithm_performance.value)
