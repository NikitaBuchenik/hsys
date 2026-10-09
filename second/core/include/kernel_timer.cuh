#pragma once
#include "cuda_check.cuh"

// Opt-in instrumentation: both events are created before benchmark iterations.
class KernelTimer {
public:
    KernelTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        const auto status = cudaEventCreate(&stop_);
        if (status != cudaSuccess) { (void)cudaEventDestroy(start_); CUDA_CHECK(status); }
    }
    ~KernelTimer() { (void)cudaEventDestroy(stop_); (void)cudaEventDestroy(start_); }
    KernelTimer(const KernelTimer&) = delete;
    KernelTimer& operator=(const KernelTimer&) = delete;
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    void stop() { CUDA_CHECK(cudaEventRecord(stop_)); }
    double seconds() const {
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_));
        return static_cast<double>(ms) * 1e-3;
    }
private:
    cudaEvent_t start_{}, stop_{};
};
namespace matrix_detail { inline thread_local KernelTimer* active_timer = nullptr; }
class ScopedKernelTimer {
public:
    explicit ScopedKernelTimer(KernelTimer& timer) : previous_(matrix_detail::active_timer) {
        matrix_detail::active_timer = &timer;
    }
    ~ScopedKernelTimer() { matrix_detail::active_timer = previous_; }
    ScopedKernelTimer(const ScopedKernelTimer&) = delete;
    ScopedKernelTimer& operator=(const ScopedKernelTimer&) = delete;
private:
    KernelTimer* previous_;
};
