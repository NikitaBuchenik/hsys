add_library(matrix_core STATIC src/matrix.cu)
target_include_directories(matrix_core PUBLIC include)
target_link_libraries(matrix_core PUBLIC CUDA::cudart)
set_target_properties(matrix_core PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
