import random
import openrouteservice
import requests
import json
import h5py
import math
import numpy as np
from numpy import ndarray  # For type hints
from concurrent.futures import ThreadPoolExecutor
from enum import Enum

import plotly.graph_objects as go

from core import hidden

api_key = hidden.apikey
client = openrouteservice.Client(key=api_key)

def generate_coordinates(number_of_nodes: int = None,
                         centre: list[float] = None) -> ndarray:
    """
    Randomly generates N latitude & longitude coordinate pairs around a
    central point.

    :param number_of_nodes: The number of coordinate pairs to generate.
    :param centre: Central point around which to generate coordinates.
    :return: Coordinates as a 2d array
    """
    coordinate_array = np.empty((number_of_nodes, 2))

    for i in range(number_of_nodes):
        latitude = round(centre[0] + random.uniform(-0.1, 0.1), 4)
        longitude = round(centre[1] + random.uniform(-0.1, 0.1), 4)
        coordinate_array[i] = [latitude, longitude]

    return coordinate_array


def generate_durations(time: float,
                       days: float,
                       number_of_locations) -> list[float]:
    """
    Generates durations for each location and scales them down to ensure they
    don't exceed three quarters of all time available.

    :param time: The time available each day.
    :param days: The number of days in the trip.
    :param number_of_locations: The number of locations being visited.
    :return:
    """
    free_time = time * 0.75  # Gives some leeway for travel time.
    total_time = free_time * days

    # Center doesn't have a duration, so is set to 0.
    # Other points are a random duration in 15 minute intervals
    durations = [0.0] + [float(random.randrange(97)) * 15
                         for _ in range(number_of_locations - 1)]
    time_taken = sum(durations)

    scaling_factor = min(1.0, total_time / time_taken)
    return [duration * scaling_factor for duration in durations]


def create_graph(coordinate_array: list[list[float]],
                 durations: list[float]) -> ndarray | int:
    """
    Takes a set of coordinates and converts them into a complete digraph, with
    each edge being the time taken to travel by car.

    Time taken is gathered using openrouteservice API.

    :param coordinate_array: The list of latitude and longitude coordinates.
    :param durations: The amount of time to be taken at each location
    :return: Graph as a 2d array, or integers representing API failings.
    """
    graph = openrouteservice_api_call(coordinate_array)
    if isinstance(graph, int):
        return graph

    number_of_locations = len(graph)
    for i in range(number_of_locations):
        for j in range(number_of_locations):
            if i == j:
                continue
            # Converts graph to minutes before adding durations
            graph[j][i] = math.ceil((graph[j][i] / 60) + durations[i])

    return graph


def openrouteservice_api_call(coordinate_array: list[list[float]]) -> ndarray | int:
    """
    Makes a POST request to the OpenRouteService API to retrieve a distance
    matrix.

    :param coordinate_array: List of coordinates to include in the api call.
    :return: A 2d list representing the distance matrix.
    """
    body = {"locations": coordinate_array, "metrics": ["duration"]}
    headers = {
        'Accept': 'application/json, application/geo+json, '
                  'application/gpx+xml, img/png; charset=utf-8',
        'Authorization': api_key,
        'Content-Type': 'application/json; charset=utf-8'
    }
    call = requests.post(
        'https://api.openrouteservice.org/v2/matrix/driving-car', json=body,
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


def generate_test_datum(days: float = None,
                        free_time: float = None,
                        coordinates: list[list[float]] = None,
                        number_of_nodes: float = None,
                        centre: list[float] = None,
                        durations: list[float] = None) -> list | int:
    """
    Generates a new training datum, using given values or randomly generated.

    :param days: Number of days visiting.
    :param free_time: Amount of free time per day, in minutes.
    :param coordinates: Used to generate graph (if none given).
    :param number_of_nodes: Number of locations (if no graph or coords given).
    :param centre: Starting point, i.e., hotel. (if no graph or coords given).
    :param durations: Amount of time spent at each location, in minutes.
    :return: Returns a new training datum, including days, time and the graph.
    """
    if days is None:
        days = float(random.randrange(1, 8))
    if free_time is None:
        free_time = float(random.randrange(73)) * 15
    if number_of_nodes is None:
        number_of_nodes = random.randrange(4, 26)
    if coordinates is None:
        if centre is None:
            city_coordinates = np.load('data/city_coordinates.npz')[
                'coordinates']
            index = random.randrange(len(city_coordinates))
            centre = city_coordinates[index]
        coordinates = generate_coordinates(number_of_nodes, centre)
    if durations is None:
        durations = generate_durations(free_time, days, number_of_nodes)
    graph = create_graph(coordinates, durations)

    if isinstance(graph, int):
        return graph

    return [graph, days, free_time, coordinates, centre]



def generate_training_datum() -> list:
    """
    Generates a training datum using random inputs for the graph, days and free
    time.

    :return: a training datum.
    """
    def generate_training_distances(num_nodes,
                                    i):
        """
        Generates a list of distances for a given node.

        :param num_nodes: Total number of nodes in the graph.
        :param i: The index of the given node.
        :return: A list containing distances from 'i' to other nodes in graph
        """
        return [0 if i == j else random.randrange(15, 960)
                for j in range(num_nodes)]

    days = float(random.randrange(1, 15))
    free_time = float(random.randrange(97)) * 15
    number_of_nodes = random.randrange(4, 26)
    with ThreadPoolExecutor() as executor:
        graph = list(executor.map(generate_training_distances,
                                  [number_of_nodes] * number_of_nodes,
                                  range(number_of_nodes)))
    return [graph, days, free_time]


class DataGroups(Enum):
    regular_graphs = 'graphs'
    ordered_graphs = 'ordered graphs'
    algorithm_performance = 'algorithm performance'


class DataAttributes(Enum):
    days = 'days'
    free_time = 'free_time'
    coordinates = 'coordinates'
    city_centre = 'centre'


def save_test_datum(datum, group: DataGroups):
    """
    Saves a training datum to the hdf5 file.

    :param datum: A list containing graph, days, free time, location coordinates
     and centre coordinates
    :param group: Which data group to save to.
    """
    with h5py.File("data/training_data.h5", 'a') as f:
        group = f[group.value]
        i = len(group)
        graph = group.create_dataset(str(i), compression="gzip",
                                     data=np.array(datum[0], dtype=np.float64))

        graph.attrs[DataAttributes.days.value] = datum[1]
        graph.attrs[DataAttributes.free_time.value] = datum[2]
        graph.attrs[DataAttributes.coordinates.value] = datum[3]
        graph.attrs[DataAttributes.city_centre.value] = datum[4]


def reset_database() -> None:
    """
    In case of emergency (I mess something up and corrupt the file),
    delete h5 file and run this.
    """
    city_coordinates = np.load('../data/city_coordinates.npz')['coordinates']

    with h5py.File("data/training_data.h5", "w") as f:
        f.create_dataset("city_coordinates", data=city_coordinates)
        f.create_group(DataGroups.regular_graphs.value)
        f.create_group(DataGroups.ordered_graphs.value)
