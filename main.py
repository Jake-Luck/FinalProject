import time
from concurrent.futures import ProcessPoolExecutor
import h5py
import numpy as np

import utilities
from utilities import reset_database


def collect_test_data():
    while True:
        datum = utilities.generate_test_datum()
        if not isinstance(datum, int):
            utilities.save_test_datum(datum)
        if datum == 1:
            print("Waiting 1 minute for api.")
            time.sleep(60)  # Wait 1 minute before trying again
            print("Continuing...")
        if datum == 2:
            print("Waiting 24hrs for api.")
            time.sleep(86400)
            print("Restarting...")


if __name__ == '__main__':
    collect_test_data()
    #with h5py.File('data/training_data.h5', 'r') as f:
    #    graphs = f['graphs']
    #    coordinates = graphs['0'].attrs['coordinates'][:]
    #    graph = graphs['0'][:]

    #utilities.display_coordinates(coordinates)
    #utilities.display_graph(coordinates, graph)
