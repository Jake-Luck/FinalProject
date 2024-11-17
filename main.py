from utilities import generate_coordinates, get_travel_time
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    coordinates = generate_coordinates(10)
    for coordinate in coordinates:
        print(coordinate)

    graph = get_travel_time(coordinates)
    print(graph)
