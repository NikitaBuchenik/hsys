#pragma once
#include "cuda_check.cuh"
class EventTimer {
public:
    EventTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        const auto error = cudaEventCreate(&stop_);
        if (error != cudaSuccess) { (void)cudaEventDestroy(start_); CUDA_CHECK(error); }
    }
    ~EventTimer() { (void)cudaEventDestroy(stop_); (void)cudaEventDestroy(start_); }
    EventTimer(const EventTimer&) = delete;
    EventTimer& operator=(const EventTimer&) = delete;
    void start() { CUDA_CHECK(cudaEventRecord(start_)); }
    double stop_seconds() {
        CUDA_CHECK(cudaEventRecord(stop_));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float milliseconds = 0;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start_, stop_));
        return milliseconds * 1e-3;
    }
private:
    cudaEvent_t start_{}, stop_{};
};
