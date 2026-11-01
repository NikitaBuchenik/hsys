#include "reduction.cuh"
#include <algorithm>

namespace {
std::size_t blocks_for(std::size_t n) {
    return n / VALUES_PER_BLOCK + (n % VALUES_PER_BLOCK != 0);
}
__device__ float load_pair(VectorView<const float> input) {
    const std::size_t i = static_cast<std::size_t>(blockIdx.x) * VALUES_PER_BLOCK + threadIdx.x;
    float sum = i < input.size ? input.data[i] : 0.0f;
    if (i + BLOCK_THREADS < input.size) sum += input.data[i + BLOCK_THREADS];
    return sum;
}
__device__ float warp_sum(float value) {
    // All 32 lanes participate, including lanes whose input is zero padding.
    for (unsigned offset = 16; offset > 0; offset /= 2)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value; // Only lane 0 holds the complete sum.
}
}

__global__ void kernel_vecred_nobr(VectorView<const float> input, float* block_sums) {
    __shared__ float sums[BLOCK_THREADS];
    const unsigned tid = threadIdx.x;
    sums[tid] = load_pair(input);
    __syncthreads();
    // A shared-memory tree, including its final warp; no shuffle intrinsics.
    for (unsigned stride = BLOCK_THREADS / 2; stride > 0; stride /= 2) {
        if (tid < stride) sums[tid] += sums[tid + stride];
        __syncthreads();
    }
    if (tid == 0) block_sums[blockIdx.x] = sums[0];
}

__global__ void kernel_vecred_br(VectorView<const float> input, float* block_sums) {
    __shared__ float warp_sums[BLOCK_THREADS / 32];
    const unsigned lane = threadIdx.x % 32;
    const unsigned warp = threadIdx.x / 32;
    float sum = warp_sum(load_pair(input));
    if (lane == 0) warp_sums[warp] = sum;
    __syncthreads();
    // The full first warp reduces eight warp sums and 24 zeros.
    if (warp == 0) {
        sum = lane < BLOCK_THREADS / 32 ? warp_sums[lane] : 0.0f;
        sum = warp_sum(sum);
        if (lane == 0) block_sums[blockIdx.x] = sum;
    }
}

ReductionWorkspace::ReductionWorkspace(std::size_t maximum_elements)
    : capacity_(maximum_elements),
      first_(std::max<std::size_t>(1, blocks_for(maximum_elements))),
      second_(std::max<std::size_t>(1, blocks_for(maximum_elements))) {}

void reduce_async(VectorView<const float> input, float* output, Reduction method,
                  ReductionWorkspace& workspace, cudaStream_t stream) {
    if (!output || (input.size && !input.data)) throw std::invalid_argument("Null reduction pointer");
    if (input.size > workspace.capacity()) throw std::length_error("Workspace is too small");
    if (method != Reduction::Shared && method != Reduction::Shuffle)
        throw std::invalid_argument("Unknown reduction strategy");
    if (input.data == output || output == workspace.first() || output == workspace.second())
        throw std::invalid_argument("Output must not alias input or scratch buffers");
    bool use_first = true;
    do {
        const auto count = std::max<std::size_t>(1, blocks_for(input.size));
        if (count > 2147483647ULL) throw std::length_error("CUDA grid is too large");
        float* destination = count == 1 ? output : (use_first ? workspace.first() : workspace.second());
        if (method == Reduction::Shared)
            kernel_vecred_nobr<<<static_cast<unsigned>(count), BLOCK_THREADS, 0, stream>>>(input, destination);
        else
            kernel_vecred_br<<<static_cast<unsigned>(count), BLOCK_THREADS, 0, stream>>>(input, destination);
        CUDA_CHECK(cudaGetLastError());
        if (count == 1) break;
        input = {destination, count};
        use_first = !use_first;
    } while (true);
}
