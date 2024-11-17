from random import random
import openrouteservice
import requests
import json

api_key = ""
client = openrouteservice.Client(key=api_key)
CENTRAL_LONDON = [-0.1279, 51.5077]

def generate_coordinates(N: int, centre = CENTRAL_LONDON):
    """
    Randomly generates N latitude & longitude coordinate pairs around a
    central point.
    :param N: The number of coordinate pairs to generate.
    :param centre: Central point around which to generate coordinates.
    :return: An array contain coordinate tuples.
    """
    coordinate_array = []
    coordinate_array.append(centre)
    for i in range(N-1):
        # random num between -0.1 and 0.1
        latitude = round(centre[0] + ((random() * 0.2) - 0.1), 4)
        longitude = round(centre[1] + ((random() * 0.2) - 0.1), 4)
        coordinate_array.append([latitude, longitude])

    return coordinate_array


def get_travel_time(coordinate_array: list[float]):
    body = {"locations": coordinate_array, "metrics": ["duration"]}

    headers = {
        'Accept': 'application/json, application/geo+json, application/gpx+xml, img/png; charset=utf-8',
        'Authorization': api_key,
        'Content-Type': 'application/json; charset=utf-8'
    }
    call = requests.post(
        'https://api.openrouteservice.org/v2/matrix/driving-car', json=body,
        headers=headers)

    response_json = json.loads(call.text)
    return response_json['durations']
