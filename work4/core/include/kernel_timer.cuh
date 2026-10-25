#pragma once
#include "cuda_check.cuh"
class KernelTimer {
public:
    KernelTimer(){CUDA_CHECK(cudaEventCreate(&start_));auto e=cudaEventCreate(&stop_);if(e!=cudaSuccess){(void)cudaEventDestroy(start_);CUDA_CHECK(e);}}
    ~KernelTimer(){(void)cudaEventDestroy(stop_);(void)cudaEventDestroy(start_);}
    KernelTimer(const KernelTimer&)=delete;KernelTimer& operator=(const KernelTimer&)=delete;
    void start(){CUDA_CHECK(cudaEventRecord(start_));}
    void stop(){CUDA_CHECK(cudaEventRecord(stop_));}
    double seconds()const{CUDA_CHECK(cudaEventSynchronize(stop_));float ms=0;CUDA_CHECK(cudaEventElapsedTime(&ms,start_,stop_));return ms*1e-3;}
private:cudaEvent_t start_{},stop_{};
};
namespace matrix_detail { inline thread_local KernelTimer* active_timer=nullptr; }
class ScopedKernelTimer {
public: explicit ScopedKernelTimer(KernelTimer& t):previous_(matrix_detail::active_timer){matrix_detail::active_timer=&t;}
    ~ScopedKernelTimer(){matrix_detail::active_timer=previous_;}
    ScopedKernelTimer(const ScopedKernelTimer&)=delete;ScopedKernelTimer& operator=(const ScopedKernelTimer&)=delete;
private:KernelTimer* previous_;
};
