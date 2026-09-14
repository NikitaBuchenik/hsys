cmake_minimum_required(VERSION 3.24)
project(practical1_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

option(BUILD_TESTS "Build GoogleTest tests" ON)
option(BUILD_BENCHMARKS "Build Google Benchmark benchmarks" ON)

add_subdirectory(core)

if(BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
