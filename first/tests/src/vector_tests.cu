#include "vector.cuh"

#include <gtest/gtest.h>
#include <Eigen/Dense>

#include <vector>
#include <type_traits>

TEST(VectorArchitectureTest, ViewIsTriviallyCopyable) {
    static_assert(std::is_trivially_copyable_v<VectorView<float>>);
    static_assert(std::is_trivially_copyable_v<VectorView<double>>);
    SUCCEED();
}

TEST(VectorDataTest, DataDeepCopy) {
    constexpr std::size_t n = 16;

    std::vector<float> host(n);
    for (std::size_t i = 0; i < n; ++i) {
        host[i] = static_cast<float>(i) * 0.5f;
    }

    Data<float> a(n);
    a.copy_from_host(host.data());

    Data<float> b(a);

    std::vector<float> copied(n, 0.0f);
    b.copy_to_host(copied.data());

    for (std::size_t i = 0; i < n; ++i) {
        EXPECT_FLOAT_EQ(copied[i], host[i]);
    }
}

TEST(VectorArchitectureTest, VectorCopiesShareData) {
    Vector<float> a(4);
    Vector<float> b = a;

    EXPECT_EQ(a.data(), b.data());
    EXPECT_EQ(a.size(), b.size());
}

TEST(VectorTest, AdditionMatchesEigen) {
    const std::vector<std::size_t> sizes{
        1, 2, 3, 127, 128, 129, 512, 1024, 1029
    };

    for (const std::size_t n : sizes) {
        std::vector<float> ha(n);
        std::vector<float> hb(n);
        std::vector<float> actual(n);

        for (std::size_t i = 0; i < n; ++i) {
            ha[i] = static_cast<float>(i) * 0.25f + 1.0f;
            hb[i] = static_cast<float>(i % 17) * 0.75f - 2.0f;
        }

        Vector<float> a(n);
        Vector<float> b(n);

        a.copy_from_host(ha.data());
        b.copy_from_host(hb.data());

        Vector<float> result = a + b;
        result.copy_to_host(actual.data());

        Eigen::VectorXf ea =
            Eigen::Map<const Eigen::VectorXf>(ha.data(),
                                               static_cast<Eigen::Index>(n));
        Eigen::VectorXf eb =
            Eigen::Map<const Eigen::VectorXf>(hb.data(),
                                               static_cast<Eigen::Index>(n));
        Eigen::VectorXf expected = ea + eb;

        Eigen::VectorXf got =
            Eigen::Map<const Eigen::VectorXf>(
                actual.data(), static_cast<Eigen::Index>(n)
            );

        EXPECT_TRUE(got.isApprox(expected, 1e-6f))
            << "n = " << n;
    }
}

TEST(VectorTest, MismatchedSizesThrow) {
    Vector<float> a(3);
    Vector<float> b(4);

    EXPECT_THROW(
        {
            auto result = a + b;
            (void)result;
        },
        std::invalid_argument
    );
}
