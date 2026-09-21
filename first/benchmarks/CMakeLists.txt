include(FetchContent)

FetchContent_Declare(
    googlebenchmark
    URL https://github.com/google/benchmark/archive/refs/tags/v1.9.4.zip
)

set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googlebenchmark)

add_executable(vector_benchmark
    src/vector_benchmark.cu
)

target_link_libraries(vector_benchmark
    PRIVATE
        vector_cuda
        benchmark::benchmark
        Eigen3::Eigen
)

set_target_properties(vector_benchmark PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
