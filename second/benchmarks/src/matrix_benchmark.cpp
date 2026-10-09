#include <stdexcept>
// Keep Eigen's allocation guard active even in Release builds.
#define eigen_assert(condition) do { if (!(condition)) throw std::runtime_error("Eigen assertion: " #condition); } while(false)
#include <Eigen/Dense>
#include "matrix.cuh"
#include "kernel_timer.cuh"
#include <benchmark/benchmark.h>
#include <chrono>
#include <iostream>
#include <random>
#include <string>

static Eigen::MatrixXf input(int n, unsigned seed) {
    Eigen::MatrixXf a(n,n); std::mt19937 rng(seed);
    std::uniform_int_distribution<int> value(-8,8);
    for(int i=0;i<a.size();++i) a.data()[i]=value(rng)/16.0f;
    return a;
}
static void BM_CudaMatmul(benchmark::State& state) {
    try {
        const int n=static_cast<int>(state.range(0));
        using RowMatrix=Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
        const RowMatrix ar=input(n,11),br=input(n,29);
        Matrix<float> a(n,n),b(n,n);
        a.data().copy_from_host(ar.data()); b.data().copy_from_host(br.data());
        for(int i=0;i<3;++i) { auto warm=a*b; }
        KernelTimer timer;
        for(auto _ : state) {
            ScopedKernelTimer measurement(timer);
            auto c=a*b; // Measure the actual operator; events inside exclude allocation/free.
            benchmark::DoNotOptimize(c.data().data());
            state.SetIterationTime(timer.seconds());
        }
    } catch(const std::exception& e) { state.SkipWithError(e.what()); }
}
class EigenNoMalloc {
public:
    EigenNoMalloc() : old_(Eigen::internal::is_malloc_allowed()) { Eigen::internal::set_is_malloc_allowed(false); }
    ~EigenNoMalloc() { Eigen::internal::set_is_malloc_allowed(old_); }
private: bool old_;
};
static void BM_EigenMatmul(benchmark::State& state) {
    try {
        const int n=static_cast<int>(state.range(0));
        const Eigen::MatrixXf a=input(n,11), b=input(n,29);
        Eigen::MatrixXf c(n,n);
        EigenNoMalloc guard;
        // Packing uses Eigen's stack scratch buffers; heap allocations are forbidden.
        for(int i=0;i<3;++i) { c.noalias()=a*b; benchmark::DoNotOptimize(c.data()); benchmark::ClobberMemory(); }
        for(auto _ : state) {
            const auto start=std::chrono::steady_clock::now();
            c.noalias()=a*b;
            benchmark::DoNotOptimize(c.data()); benchmark::ClobberMemory();
            const auto stop=std::chrono::steady_clock::now();
            state.SetIterationTime(std::chrono::duration<double>(stop-start).count());
        }
    } catch(const std::exception& e) { state.SkipWithError(e.what()); }
}
BENCHMARK(BM_CudaMatmul)->RangeMultiplier(2)->Range(16,1024)->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_EigenMatmul)->RangeMultiplier(2)->Range(16,1024)->UseManualTime()->Unit(benchmark::kMicrosecond);
int main(int argc,char** argv) {
    try {
        int device=0,driver=0,runtime=0; cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDevice(&device)); CUDA_CHECK(cudaGetDeviceProperties(&prop,device));
        CUDA_CHECK(cudaDriverGetVersion(&driver)); CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        benchmark::AddCustomContext("gpu",prop.name);
        benchmark::AddCustomContext("compute_capability",std::to_string(prop.major)+"."+std::to_string(prop.minor));
        benchmark::AddCustomContext("cuda_driver",std::to_string(driver));
        benchmark::AddCustomContext("cuda_runtime",std::to_string(runtime));
        benchmark::AddCustomContext("eigen_threads","1");
        benchmark::AddCustomContext("measurement","CUDA events around kernel inside operator*; Eigen steady_clock, preallocated output, no heap allocations");
        benchmark::Initialize(&argc,argv);
        if(benchmark::ReportUnrecognizedArguments(argc,argv)) return 1;
        benchmark::RunSpecifiedBenchmarks(); benchmark::Shutdown();
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
