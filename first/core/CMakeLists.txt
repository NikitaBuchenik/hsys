add_library(vector_cuda STATIC
    src/vector.cu
)

target_include_directories(vector_cuda
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

set_target_properties(vector_cuda PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)
