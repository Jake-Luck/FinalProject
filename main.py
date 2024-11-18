from utilities import generate_training_datum

if __name__ == '__main__':
    while True:
        response = generate_training_datum()
        if response == 1:
            break
