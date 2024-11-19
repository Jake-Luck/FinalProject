import random
import openrouteservice
import requests
import json
import numpy as np
import h5py
import math
import networkx as nx
import matplotlib
import matplotlib.pyplot as plt
from concurrent.futures import ThreadPoolExecutor

import secrets

matplotlib.use('TkAgg')


api_key = secrets.apikey
client = openrouteservice.Client(key=api_key)


def generate_coordinates(number_of_nodes: int = None,
                         centre: list[float] = None):
    """
    Randomly generates N latitude & longitude coordinate pairs around a
    central point.

    :param number_of_nodes: The number of coordinate pairs to generate.
    :param centre: Central point around which to generate coordinates.
    :return: Coordinates as a 2d array
    """
    coordinate_array = [centre.tolist()]

    for i in range(number_of_nodes - 1):
        latitude = round(centre[0] + random.uniform(-0.1, 0.1), 4)
        longitude = round(centre[1] + random.uniform(-0.1, 0.1), 4)
        coordinate_array.append([latitude, longitude])

    return coordinate_array


def generate_durations(time: float, days: float, number_of_locations):
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

    # Random duration in 15 minute intervals
    durations = [0.0] + [float(random.randrange(97)) * 15
                         for _ in range(number_of_locations - 1)]
    time_taken = sum(durations)

    scaling_factor = min(1.0, total_time / time_taken)
    return [duration * scaling_factor for duration in durations]


def create_graph(coordinate_array: list[list[float]],
                 durations: list[float]):
    """
    Takes a set of coordinates and converts them into a complete digraph, with
    each edge being the time taken to travel by car.

    Time taken is gathered using openrouteservice API.

    :param coordinate_array: The list of latitude and longitude coordinates.
    :param durations: The amount of time to be taken at each location
    :return: Graph as a 2d array, or integers representing API failings.
    """


    # Add time spent at location to edges ending at location. i.e. edge is
    # travel time + time spent at destination
    graph = openrouteservice_api_call(coordinate_array)
    if isinstance(graph, int):
        return graph

    number_of_locations = len(graph)
    for i in range(number_of_locations):
        for j in range(number_of_locations):
            if i == j:
                continue
            # Converts graph to minutes before adding durations
            graph[j][i] = (graph[j][i] / 60) + durations[i]

    return graph


def openrouteservice_api_call(coordinate_array: list[list[float]]):
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

    return graph


def generate_test_datum(days: float = None,
                        free_time: float = None,
                        coordinates: list[list[float]] = None,
                        number_of_nodes: float = None,
                        centre: list[float] = None,
                        durations: list[float] = None):
    """
    Generates a new training datum, using given values or randomly generated.

    :param days: Number of days visiting.
    :param free_time: Amount of free time per day.
    :param coordinates: Used to generate graph (if none given).
    :param number_of_nodes: Number of locations (if no graph or coords given).
    :param centre: Starting point, i.e., hotel. (if no graph or coords given).
    :param durations: Amount of time spent at each location
    :return: Returns a new training datum, including days, time and the graph.
    """
    if days is None:
        days = float(random.randrange(1, 8))
    if free_time is None:
        free_time = float(random.randrange(97)) * 15
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
    if graph == 1:
        return 1
    if graph == 0:
        return 0

    return [graph, days, free_time, coordinates]


def generate_training_datum():
    """
    Generates a training datum using random inputs for the graph, days and free
    time.

    :return: a training datum.
    """
    def generate_training_distances(i, num_nodes):
        """
        Generates a list of distances for a given node.

        :param i: The index of the given node.
        :param num_nodes: Total number of nodes in the graph.
        :return: A list containing distances from 'i' to other nodes in graph
        """
        return [0 if i == j else random.randrange(15, 960)
                for j in range(num_nodes)]

    days = float(random.randrange(1, 15))
    free_time = float(random.randrange(97)) * 15
    number_of_nodes = random.randrange(4, 26)
    with ThreadPoolExecutor() as executor:
        graph = list(executor.map(generate_training_distances,
                                  range(number_of_nodes),
                                  [number_of_nodes]*number_of_nodes))
    return [graph, days, free_time]


def save_test_datum(datum):
    """
    Saves a training datum to the hdf5 file.

    :param datum: A list containing graph, days, and free time per day.
    """
    with h5py.File("data/training_data.h5", 'a') as f:
        group = f['graphs']
        i = len(group)
        graph = group.create_dataset(str(i), compression="gzip",
                                     data=np.array(datum[0], dtype=np.float64))

        graph.attrs['days'] = datum[1]
        graph.attrs['free_time'] = datum[2]
        graph.attrs['coordinates'] = datum[3]


def reset_database():
    """
    In case of emergency (I mess something up and corrupt the file),
    delete h5 file and run this.
    """
    city_coordinates = np.load('data/city_coordinates.npz')['coordinates']

    with h5py.File("data/training_data.h5", "w") as f:
        f.create_dataset("city_coordinates", data=city_coordinates)
        f.create_group("graphs")


def display_coordinates(coordinates: list[list[float]]):
    """
    Displays a scatter plot of the given coordinates.

    :param coordinates: The coordinates (e.g. [[0,0], [0,1], [1,0], [1,1]])
    """
    x_coordinates = coordinates[:, 0]
    y_coordinates = coordinates[:, 1]
    plt.scatter(x_coordinates, y_coordinates)
    [plt.text(coordinate[0], coordinate[1],
              f"({coordinate[0]}, {coordinate[1]})")
     for coordinate in coordinates]
    plt.show()


def display_route(coordinates: list[list[float]],
                  route: list[list[float]],
                  show_weights: bool = False):
    """


    Credit to:
    https://stackoverflow.com/questions/64986306/how-to-plot-a-networkx-graph-using-the-x-y-coordinates-of-the-points-list

    :param coordinates: The coordinates (e.g. [[0,0], [0,1], [1,0], [1,1]])
    :param route: The route visiting coordinates (e.g. [[0, 1, 5], [1, 2, 5],
                                                        [2, 3, 5], [2, 1, 3]])
    :param show_weights: Whether to display the edge weights or not.
    """
    G = nx.DiGraph()
    points = list(map(tuple, coordinates))
    edges = list(map(tuple, route))

    for i in range(len(route)):
        if show_weights:
            G.add_edge(points[edges[i][0]], points[edges[i][1]], weight=edges[i][2])
        else:
            G.add_edge(points[edges[i][0]], points[edges[i][1]])

    pos = {point: point for point in points}

    fig, ax = plt.subplots()
    nx.draw(G, pos=pos, node_color='k', ax=ax)
    nx.draw(G, pos=pos, node_size=1500, ax=ax)  # draw nodes and edges
    nx.draw_networkx_labels(G, pos=pos)  # draw node labels/names

    # Draw edge weights (if present)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax,
                                 rotate=False, label_pos=0.825)

    plt.axis("on")
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    plt.show()

