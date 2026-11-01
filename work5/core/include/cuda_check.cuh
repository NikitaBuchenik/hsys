#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
inline void check_cuda(cudaError_t error, const char* expression, const char* file, int line) {
    if (error != cudaSuccess) {
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line) +
                                 " " + expression + ": " + cudaGetErrorString(error));
    }
}
#define CUDA_CHECK(expr) check_cuda((expr), #expr, __FILE__, __LINE__)
