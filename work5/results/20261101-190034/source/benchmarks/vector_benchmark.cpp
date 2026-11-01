#include "reduction.cuh"
#include "event_timer.cuh"
#include <benchmark/benchmark.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

namespace {
std::size_t chunk_capacity = std::size_t{1} << 26; // At most 256 MiB input on GPU.

// A period of 257 exactly representable values, with a zero period sum.
// The logical vector is x[i] = ((i % 257) - 128) / 256 for every i in [0,n).
double expected_sum(std::size_t n) {
    const double rest = static_cast<double>(n % 257);
    return (rest * (rest - 1) / 2 - 128 * rest) / 256;
}

void bench(benchmark::State& state, Reduction method) {
    try {
        const auto n = static_cast<std::size_t>(state.range(0));
        const auto chunk = std::min(n, chunk_capacity);
        const auto chunks = n / chunk + (n % chunk != 0);
        const bool streamed = chunks > 1;
        std::vector<float> host(chunk + 256);
        for (std::size_t i = 0; i < host.size(); ++i)
            host[i] = static_cast<float>(static_cast<int>(i % 257) - 128) / 256;
        DeviceBuffer<float> input(chunk), partial(chunks), answer(1);
        ReductionWorkspace scratch(std::max(chunk, chunks));
        EventTimer timer;
        if (!streamed) input.copy_from_host(host.data(), n);

        auto operation = [&]() {
            double seconds = 0;
            for (std::size_t part = 0, offset = 0; part < chunks; ++part, offset += chunk) {
                const auto count = std::min(chunk, n - offset);
                // Copy and generation are outside every CUDA Events interval.
                if (streamed) input.copy_from_host(host.data() + offset % 257, count);
                timer.start();
                reduce_async({input.data(), count}, streamed ? partial.data() + part : answer.data(), method, scratch);
                seconds += timer.stop_seconds();
            }
            if (streamed) {
                timer.start();
                reduce_async({partial.data(), chunks}, answer.data(), method, scratch);
                seconds += timer.stop_seconds();
            }
            return seconds;
        };
        auto verify = [&]() {
            float actual = 0;
            answer.copy_to_host(&actual, 1);
            if (!std::isfinite(actual) || std::abs(static_cast<double>(actual) - expected_sum(n)) > 1e-4)
                throw std::runtime_error("Reduction result check failed at n=" + std::to_string(n));
        };
        for (int warm = 0; warm < 3; ++warm) operation();
        verify();
        for (auto _ : state) {
            const double seconds = operation();
            if (!(seconds > 0)) throw std::runtime_error("CUDA Events reported zero time");
            state.SetIterationTime(seconds);
        }
        verify();
        state.SetLabel(streamed ? "streamed_compute_only" : "resident_compute_only");
        state.counters["chunks"] = static_cast<double>(chunks);
        state.counters["chunk_elements"] = static_cast<double>(chunk);
        state.counters["input_bytes"] = static_cast<double>(n) * sizeof(float);
        state.counters["verified"] = 1;
    } catch (const std::exception& error) {
        state.SkipWithError(error.what());
    }
}
}

int main(int argc, char** argv) {
    try {
        static_assert(sizeof(std::size_t) >= 8, "The complete experiment requires a 64-bit build");
        int device = 0, driver = 0, runtime = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        cudaDeviceProp props{};
        CUDA_CHECK(cudaGetDeviceProperties(&props, device));
        std::size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        while (chunk_capacity * sizeof(float) > free_bytes / 4 && chunk_capacity > (1u << 18))
            chunk_capacity /= 2;
        if (chunk_capacity * sizeof(float) > free_bytes / 4)
            throw std::runtime_error("Too little free GPU memory; close GPU applications");
        CUDA_CHECK(cudaDriverGetVersion(&driver));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        for (int exponent = 0; exponent <= 28; ++exponent) {
            const auto n = std::int64_t{8} << exponent;
            benchmark::RegisterBenchmark("BM_Shared", bench, Reduction::Shared)
                ->Arg(n)->UseManualTime()->Unit(benchmark::kMicrosecond);
            benchmark::RegisterBenchmark("BM_Shuffle", bench, Reduction::Shuffle)
                ->Arg(n)->UseManualTime()->Unit(benchmark::kMicrosecond);
        }
        benchmark::Initialize(&argc, argv);
        if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
        benchmark::AddCustomContext("gpu", props.name);
        benchmark::AddCustomContext("compute_capability", std::to_string(props.major) + "." + std::to_string(props.minor));
        benchmark::AddCustomContext("cuda_driver", std::to_string(driver));
        benchmark::AddCustomContext("cuda_runtime", std::to_string(runtime));
        benchmark::AddCustomContext("gpu_memory_bytes", std::to_string(total_bytes));
        benchmark::AddCustomContext("chunk_capacity_elements", std::to_string(chunk_capacity));
        benchmark::AddCustomContext("input_pattern", "((i % 257) - 128) / 256, float32");
        benchmark::AddCustomContext("timing", "sum of CUDA event intervals; all passes plus final chunk merge; no copies");
        benchmark::RunSpecifiedBenchmarks();
        benchmark::Shutdown();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
