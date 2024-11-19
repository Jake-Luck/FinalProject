import time
from concurrent.futures import ProcessPoolExecutor
from utilities import generate_test_datum, save_test_datum, reset_database


def collect_test_data():
    while True:
        datum = generate_test_datum()
        if not isinstance(datum[0], int):
            save_test_datum(datum)
        if datum[0] == 1:
            print("Waiting 1 minute for api.")
            time.sleep(60)  # Wait 1 minute before trying again
            print("Continuing...")
        if datum[0] == 2:
            print("Waiting 24hrs for api.")
            time.sleep(86400)
            print("Restarting...")


if __name__ == '__main__':
    with ProcessPoolExecutor() as executor:
        future = executor.submit(collect_test_data)
        print("test")
