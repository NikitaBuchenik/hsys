#include "vector.cuh"
#include <benchmark/benchmark.h>
#include <vector>
#include <Eigen/Dense>

static void BM_CudaVectorAdd(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    std::vector<float> host_a(n, 1.25f);
    std::vector<float> host_b(n, 2.75f);
    Vector<float> a(n);
    Vector<float> b(n);
    Vector<float> result(n);
    a.copy_from_host(host_a.data());
    b.copy_from_host(host_b.data());

    launch_vecadd(a.view(), b.view(), result.view());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{}, stop{};
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (auto _ : state) {
        CUDA_CHECK(cudaEventRecord(start));
        launch_vecadd(a.view(), b.view(), result.view());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float milliseconds = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
        state.SetIterationTime(static_cast<double>(milliseconds) / 1000.0);
    }
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * n * 3 * sizeof(float));
}
BENCHMARK(BM_CudaVectorAdd)->Arg(8)->Arg(64)->Arg(512)->Arg(4096)->Arg(32768)->Arg(262144)->Arg(2097152)->Arg(16777216)->UseManualTime();

static void BM_EigenVectorAdd(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));
    
    Eigen::VectorXf a = Eigen::VectorXf::Constant(n, 1.25f);
    Eigen::VectorXf b = Eigen::VectorXf::Constant(n, 2.75f);
    Eigen::VectorXf result(n);

    for (auto _ : state) {
        benchmark::DoNotOptimize(result = a + b);
    }

    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) * n * 3 * sizeof(float));
}
BENCHMARK(BM_EigenVectorAdd)->Arg(8)->Arg(64)->Arg(512)->Arg(4096)->Arg(32768)->Arg(262144)->Arg(2097152)->Arg(16777216);

BENCHMARK_MAIN();
