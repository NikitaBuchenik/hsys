#include "reduction.cuh"
#include <Eigen/Core>
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <random>
#include <tuple>
#include <vector>

float gpu_sum(const std::vector<float>& host, Reduction method) {
    DeviceBuffer<float> input(host.size()), output(1);
    input.copy_from_host(host.data(), host.size());
    ReductionWorkspace scratch(host.size());
    reduce_async(input.read_view(), output.data(), method, scratch);
    CUDA_CHECK(cudaDeviceSynchronize());
    float answer = 0;
    output.copy_to_host(&answer, 1);
    return answer;
}

class ReductionTest : public ::testing::TestWithParam<std::tuple<Reduction, int>> {};
TEST_P(ReductionTest, MatchesEigenSum) {
    const auto method = std::get<0>(GetParam());
    const int n = std::get<1>(GetParam());
    std::mt19937 rng(2026 + n);
    std::uniform_real_distribution<float> random(-1.0f, 1.0f);
    for (int pattern = 0; pattern < 5; ++pattern) {
        SCOPED_TRACE("pattern=" + std::to_string(pattern));
        Eigen::Matrix<float, Eigen::Dynamic, 1> reference(n);
        for (int i = 0; i < n; ++i) {
            switch (pattern) {
            case 0: reference(i) = 0.0f; break;
            case 1: reference(i) = 1.0f; break;
            case 2: reference(i) = static_cast<float>(i % 257 - 128) / 256.0f; break;
            case 3: reference(i) = i % 2 ? -0.125f : 0.125f; break;
            default: reference(i) = random(rng); break;
            }
        }
        const std::vector<float> input(reference.data(), reference.data() + n);
        const float actual = gpu_sum(input, method);
        EXPECT_TRUE(std::isfinite(actual));
        EXPECT_NEAR(actual, reference.sum(), 1e-4);
    }
}
INSTANTIATE_TEST_SUITE_P(AllSizes, ReductionTest,
    ::testing::Combine(::testing::Values(Reduction::Shared, Reduction::Shuffle),
                       ::testing::Values(1, 2, 3, 127, 129, 512, 541, 1037)));

TEST(Edges, Empty) {
    for (auto method : {Reduction::Shared, Reduction::Shuffle})
        EXPECT_FLOAT_EQ(gpu_sum({}, method), 0.0f);
}
TEST(Edges, MoreThanTwoPasses) {
    std::vector<float> input(512 * 512 + 17, 0.125f);
    for (auto method : {Reduction::Shared, Reduction::Shuffle})
        EXPECT_NEAR(gpu_sum(input, method), input.size() * 0.125, 1e-4);
}
TEST(Edges, WarpAndBlockTails) {
    for (std::size_t n : {31, 32, 33, 255, 256, 257, 511, 513, 1023, 1025}) {
        std::vector<float> input(n, -0.25f);
        for (auto method : {Reduction::Shared, Reduction::Shuffle})
            EXPECT_NEAR(gpu_sum(input, method), -0.25 * n, 1e-4);
    }
}
TEST(Edges, ChunkedWithTail) {
    std::vector<float> host(1037);
    for (std::size_t i = 0; i < host.size(); ++i) host[i] = static_cast<float>(int(i % 257) - 128) / 256.0f;
    Eigen::Map<const Eigen::VectorXf> reference(host.data(), host.size());
    for (auto method : {Reduction::Shared, Reduction::Shuffle}) {
        DeviceBuffer<float> input(129), parts(9), output(1);
        ReductionWorkspace scratch(129);
        std::size_t part = 0;
        for (std::size_t offset = 0; offset < host.size(); offset += 129, ++part) {
            const auto count = std::min<std::size_t>(129, host.size() - offset);
            input.copy_from_host(host.data() + offset, count);
            reduce_async({input.data(), count}, parts.data() + part, method, scratch);
        }
        reduce_async({parts.data(), part}, output.data(), method, scratch);
        CUDA_CHECK(cudaDeviceSynchronize());
        float actual;
        output.copy_to_host(&actual, 1);
        EXPECT_NEAR(actual, reference.sum(), 1e-4);
    }
}
TEST(Edges, ReusedWorkspace) {
    DeviceBuffer<float> input(1037), output(1);
    ReductionWorkspace scratch(1037);
    for (float value : {0.5f, -0.25f, 0.0f}) {
        std::vector<float> host(1037, value);
        input.copy_from_host(host.data(), host.size());
        for (auto method : {Reduction::Shared, Reduction::Shuffle}) {
            reduce_async(input.read_view(), output.data(), method, scratch);
            float actual;
            output.copy_to_host(&actual, 1);
            EXPECT_NEAR(actual, host.size() * static_cast<double>(value), 1e-4);
        }
    }
}
TEST(Validation, RejectsInvalidArguments) {
    DeviceBuffer<float> input(2), output(1);
    ReductionWorkspace scratch(1);
    EXPECT_THROW(reduce_async({nullptr, 1}, output.data(), Reduction::Shared, scratch), std::invalid_argument);
    EXPECT_THROW(reduce_async({input.data(), 1}, nullptr, Reduction::Shared, scratch), std::invalid_argument);
    EXPECT_THROW(reduce_async(input.read_view(), output.data(), Reduction::Shared, scratch), std::length_error);
    EXPECT_THROW(reduce_async({input.data(), 1}, input.data(), Reduction::Shared, scratch), std::invalid_argument);
    EXPECT_THROW(reduce_async({input.data(), 1}, output.data(), static_cast<Reduction>(99), scratch), std::invalid_argument);
}
TEST(Ownership, MoveAndHostCopies) {
    const float host[] = {1, 2, 3};
    DeviceBuffer<float> a(3);
    a.copy_from_host(host, 3);
    DeviceBuffer<float> b(std::move(a));
    EXPECT_EQ(a.data(), nullptr);
    DeviceBuffer<float> c(1);
    c = std::move(b);
    EXPECT_EQ(b.size(), 0u);
    float result[3]{};
    c.copy_to_host(result, 3);
    EXPECT_FLOAT_EQ(result[2], 3.0f);
    EXPECT_THROW(c.copy_from_host(nullptr, 1), std::invalid_argument);
    EXPECT_THROW(c.copy_to_host(result, 4), std::invalid_argument);
}
TEST(Ownership, Overflow) {
    EXPECT_THROW((void)DeviceBuffer<float>{std::numeric_limits<std::size_t>::max()}, std::length_error);
}
