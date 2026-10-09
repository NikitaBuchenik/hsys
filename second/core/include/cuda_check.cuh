#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
inline void cuda_check(cudaError_t status, const char* expression,
                       const char* file, int line) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(file) + ":" + std::to_string(line)
            + " " + expression + ": " + cudaGetErrorString(status));
}
#define CUDA_CHECK(expr) cuda_check((expr), #expr, __FILE__, __LINE__)
