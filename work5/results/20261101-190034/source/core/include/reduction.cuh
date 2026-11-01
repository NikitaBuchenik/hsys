#pragma once
#include "vector.cuh"

constexpr unsigned BLOCK_THREADS = 256;
constexpr unsigned VALUES_PER_BLOCK = 2 * BLOCK_THREADS;
enum class Reduction { Shared, Shuffle };

// Scratch allocation is explicit and can be reused outside a timed interval.
class ReductionWorkspace {
public:
    explicit ReductionWorkspace(std::size_t maximum_elements);
    std::size_t capacity() const noexcept { return capacity_; }
    float* first() noexcept { return first_.data(); }
    float* second() noexcept { return second_.data(); }
private:
    std::size_t capacity_;
    DeviceBuffer<float> first_, second_;
};

// Reduces the complete vector to one float on the GPU, including all passes.
// output must point to a separate device float; no allocations or copies here.
void reduce_async(VectorView<const float> input, float* output,
                  Reduction method, ReductionWorkspace& workspace,
                  cudaStream_t stream = nullptr);

#ifdef __CUDACC__
__global__ void kernel_vecred_nobr(VectorView<const float> input, float* block_sums);
__global__ void kernel_vecred_br(VectorView<const float> input, float* block_sums);
#endif
