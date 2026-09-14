#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

inline void cuda_check(cudaError_t error, const char* file, int line) {
    if (error != cudaSuccess) {
        throw std::runtime_error(
            std::string("CUDA error at ") + file + ":" + std::to_string(line) +
            ": " + cudaGetErrorString(error)
        );
    }
}

#define CUDA_CHECK(expr) cuda_check((expr), __FILE__, __LINE__)
