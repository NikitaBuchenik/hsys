#include "matrix.cuh"
#include "kernel_timer.cuh"
#include <benchmark/benchmark.h>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// Exactly the same quantized numerical input is used by all three methods.
template<class T> std::vector<T> input(int n,unsigned seed) {
    std::mt19937 rng(seed); std::uniform_int_distribution<int> d(-8,8);
    std::vector<T> result(static_cast<std::size_t>(n)*n);
    for (auto& x:result) {
        float v=d(rng)/16.0f;
        if constexpr(std::is_same_v<T,__half>) x=__float2half_rn(v); else x=v;
    }
    return result;
}
template<class T, bool Wmma> void bench(benchmark::State& state) {
    try {
        const int n=static_cast<int>(state.range(0));
        auto ah=input<T>(n,11),bh=input<T>(n,29);
        Matrix<T> a(n,n),b(n,n);
        a.data().copy_from_host(ah.data()); b.data().copy_from_host(bh.data());
        auto product=[&]() {
            if constexpr(std::is_same_v<T,float>) return a*b;
            else if constexpr(Wmma) return a*b;
            else return multiply(a,b,MatmulStrategy::SharedMemory);
        };
        for(int i=0;i<3;++i) { auto warm=product(); }
        KernelTimer timer;
        for(auto _:state) {
            ScopedKernelTimer scope(timer);
            auto c=product();
            benchmark::DoNotOptimize(c.data().data());
            state.SetIterationTime(timer.seconds());
        }
    } catch(const std::exception& e) { state.SkipWithError(e.what()); }
}
static void BM_WMMA(benchmark::State& s) { bench<__half,true>(s); }
static void BM_SharedFP32(benchmark::State& s) { bench<float,false>(s); }
static void BM_SharedFP16(benchmark::State& s) { bench<__half,false>(s); }
BENCHMARK(BM_WMMA)->RangeMultiplier(2)->Range(16,1024)->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SharedFP32)->RangeMultiplier(2)->Range(16,1024)->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_SharedFP16)->RangeMultiplier(2)->Range(16,1024)->UseManualTime()->Unit(benchmark::kMicrosecond);
int main(int argc,char** argv) {
    try {
        int device=0,driver=0,runtime=0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp p{}; CUDA_CHECK(cudaGetDeviceProperties(&p,device));
        if(p.major<7) throw std::runtime_error("WMMA FP16 requires compute capability >= 7.0");
        CUDA_CHECK(cudaDriverGetVersion(&driver)); CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        benchmark::Initialize(&argc,argv);
        if(benchmark::ReportUnrecognizedArguments(argc,argv)) return 1;
        benchmark::AddCustomContext("gpu",p.name);
        benchmark::AddCustomContext("compute_capability",std::to_string(p.major)+"."+std::to_string(p.minor));
        benchmark::AddCustomContext("gpu_memory_bytes",std::to_string(p.totalGlobalMem));
        benchmark::AddCustomContext("cuda_driver",std::to_string(driver));
        benchmark::AddCustomContext("cuda_runtime",std::to_string(runtime));
        benchmark::AddCustomContext("wmma_precision","FP16 inputs, FP32 accumulator/output");
        benchmark::AddCustomContext("baseline","work3 shared-memory FP32, tile16");
        benchmark::RunSpecifiedBenchmarks(); benchmark::Shutdown();
        return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
