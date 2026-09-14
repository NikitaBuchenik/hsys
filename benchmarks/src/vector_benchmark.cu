#include "vector.cuh"

#include <benchmark/benchmark.h>

#include <vector>

static void BM_CudaVectorAdd(benchmark::State& state) {
    const std::size_t n = static_cast<std::size_t>(state.range(0));

    std::vector<float> host_a(n, 1.25f);
    std::vector<float> host_b(n, 2.75f);

    Vector<float> a(n);
    Vector<float> b(n);
    Vector<float> result(n);

    a.copy_from_host(host_a.data());
    b.copy_from_host(host_b.data());

    // Warm-up.
    launch_vecadd(a.view(), b.view(), result.view());
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start{};
    cudaEvent_t stop{};
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

    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(n) *
        static_cast<int64_t>(3 * sizeof(float))
    );
}

BENCHMARK(BM_CudaVectorAdd)
    ->Args(8)
    ->Args(8 * 8)
    ->Args(8 * 8 * 8)
    ->Args(8 * 8 * 8 * 8)
    ->Args(8 * 8 * 8 * 8 * 8)
    ->Args(8 * 8 * 8 * 8 * 8 * 8)
    ->Args(8 * 8 * 8 * 8 * 8 * 8 * 8)
    ->Args(8 * 8 * 8 * 8 * 8 * 8 * 8 * 8)
    ->UseManualTime();

BENCHMARK_MAIN();
