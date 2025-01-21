cmake_minimum_required(VERSION 3.29)
project(Algorithms CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(BruteForce BruteForce.cu)

set_target_properties(BruteForce PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)
