import numpy as np

def generate_route(base_set: list[int], route_number: int, n_routes: int,
                   route_length: int) -> list[int]:
    n_factorial = n_routes / route_length-1
    route = list()
    for i in range(route_length-2, -1, -1):
        selected_index = route_number / n_factorial
        route.append(base_set.pop(selected_index))

        if i == 0: continue

        route_number %= n_factorial
        n_factorial /= i
    route.append(0) # All routes end going back to centre
    return route


def brute_force(n: int, d: int, graph: np.ndarray) -> list[int]:
    pass