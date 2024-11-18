import random
import openrouteservice
import requests
import json
import numpy as np
import h5py
import math

import secrets


api_key = secrets.apikey
client = openrouteservice.Client(key=api_key)


def generate_coordinates(number_of_nodes: int = None,
                         centre: list[float] = None):
    """
    Randomly generates N latitude & longitude coordinate pairs around a
    central point.

    If no centre is provided, one shall be randomly chosen from a list of
    large cities.
    :param number_of_nodes: The number of coordinate pairs to generate.
    :param centre: Central point around which to generate coordinates.
    :return: Coordinates as a 2d array
    """
    if centre is None:
        city_coordinates = np.load('data/city_coordinates.npz')['coordinates']
        index = random.randrange(len(city_coordinates))
        centre = city_coordinates[index]
    coordinate_array = [centre.tolist()]

    if number_of_nodes is None:
        number_of_nodes = random.randrange(4, 11)

    for i in range(number_of_nodes - 1):
        # random num between -0.1 and 0.1
        latitude = round(centre[0] + random.uniform(-0.1, 0.1), 4)
        longitude = round(centre[1] + random.uniform(-0.1, 0.1), 4)
        coordinate_array.append([latitude, longitude])

    return coordinate_array


def create_graph(coordinate_array: list[list[float]]):
    """
    Takes a set of coordinates and converts them into a complete digraph, with
    each edge being the time taken to travel by car.

    Time taken is gathered using openrouteservice API.

    :param coordinate_array: The list of latitude and longitude coordinates.
    :return: Graph as a 2d array, or integers representing API failings.
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

    if 'error' in response_json:
        if response_json['error'] == 'Rate limit exceeded' \
                or response_json['error'] == 'Quota exceeded':
            return 1  # Maxed out api calls
        else:
            raise ValueError("Invalid response from API call\n"
                             + response_json['error'])

    graph = response_json['durations']

    # If any edges are NaN or None, graph is unusable, return 0
    if any(path is None or math.isnan(path) for path in
           [item for row in graph for item in row]):
        return 0

    return graph


def generate_training_datum(days: float = None,
                            free_time: float = None,
                            graph: list[list[float]] = None,
                            coordinates: list[list[float]] = None,
                            number_of_nodes: float = None,
                            centre: list[float] = None):
    """
    Generates a new training datum, using given values or randomly generated.

    :param days: Number of days visiting.
    :param free_time: Amount of free time per day.
    :param graph: Complete digraph between each visited location.
    :param coordinates: Used to generate graph (if none given).
    :param number_of_nodes: Number of locations (if no graph or coords given).
    :param centre: Starting point, i.e., hotel. (if no graph or coords given).
    :return: Returns a new training datum, including days, time and the graph.
    """
    if graph is None:
        if coordinates is None:
            coordinates = generate_coordinates(number_of_nodes, centre)
        graph = create_graph(coordinates)
        if graph == 1:
            print("Reached max api calls")
            return 1
        if graph == 0:
            return 0
    if days is None:
        days = float(random.randrange(1, 8))
    if free_time is None:
        free_time = float(random.randrange(49)) / 2

    save_training_datum(graph, days, free_time)
    return [graph, days, free_time]


def save_training_datum(graph: list[list[float]],
                        days: float,
                        free_time: float):
    """
    Saves a training datum to the hdf5 file.
    :param graph: Complete digraph between each visited location
    :param days: Number of days visiting.
    :param free_time: Amount of free time per day.
    """
    with h5py.File("data/training_data.h5", 'a') as f:
        group = f['graphs']
        i = len(group)
        graph = group.create_dataset(str(i), data=np.array(graph, dtype=np.float64), compression="gzip")
        graph.attrs['days'] = days
        graph.attrs['free_time'] = free_time


def reset_database():
    """
    In case of emergency (I mess something up and corrupt the file.
    """
    city_coordinates = np.load('data/city_coordinates.npz')['coordinates']

    with h5py.File("data/training_data.h5", "w") as f:
        f.create_dataset("city_coordinates", data=city_coordinates)
        f.create_group("graphs")
