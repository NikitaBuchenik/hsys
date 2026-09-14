include(FetchContent)

FetchContent_Declare(
    googletest
    URL https://github.com/google/googletest/archive/refs/tags/v1.17.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
    eigen
    URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
)

FetchContent_MakeAvailable(eigen)

add_executable(vector_tests
    src/vector_tests.cu
)

target_link_libraries(vector_tests
    PRIVATE
        vector_cuda
        GTest::gtest_main
        Eigen3::Eigen
)

set_target_properties(vector_tests PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

include(GoogleTest)
gtest_discover_tests(vector_tests)
