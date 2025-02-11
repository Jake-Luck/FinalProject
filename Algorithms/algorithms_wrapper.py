import ctypes
import numpy as np
import numpy.typing as npt

algorithms = ctypes.WinDLL("./algorithms/algorithms.dll")

algorithms.bruteForce.argtypes = [
    ctypes.c_char,
    ctypes.c_char,
    ctypes.POINTER(ctypes.c_int32)
]
algorithms.bruteForce.restype = ctypes.POINTER(ctypes.c_char)

def bruteForce(n: int, d: int, graph: np.ndarray) -> list[int]:
    c_n = ctypes.c_char(n)
    c_d = ctypes.c_char(d)
    graph = graph.flatten()
    c_graph = graph.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))

    result = algorithms.bruteForce(c_n, c_d, c_graph)
    return result
