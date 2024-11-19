import time
from concurrent.futures import ProcessPoolExecutor
import h5py
import numpy as np

import utilities


def collect_test_data():
    while True:
        datum = utilities.generate_test_datum()
        if not isinstance(datum[0], int):
            utilities.save_test_datum(datum)
        if datum[0] == 1:
            print("Waiting 1 minute for api.")
            time.sleep(60)  # Wait 1 minute before trying again
            print("Continuing...")
        if datum[0] == 2:
            print("Waiting 24hrs for api.")
            time.sleep(86400)
            print("Restarting...")


if __name__ == '__main__':
    #with ProcessPoolExecutor() as executor:
    #    future = executor.submit(collect_test_data)
    #    print("test")
    coordinates = [[0,0], [0,1], [1,0], [1,1]]
    route = [[0, 1, 5], [1, 2, 5], [2, 3, 5], [2, 1, 3]]
    # utilities.display_coordinates(coordinates)
    utilities.display_route(coordinates, route, True)
