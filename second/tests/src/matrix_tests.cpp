#include "matrix.cuh"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <array>
#include <random>
#include <tuple>
#include <vector>

using RowMatrix = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Shape = std::tuple<int,int,int>;
class ProductTest : public ::testing::TestWithParam<Shape> {};
TEST_P(ProductTest, MatchesEigenAndAbsoluteTolerance) {
    const auto [m,n,k] = GetParam();
    SCOPED_TRACE(::testing::Message() << "m=" << m << " n=" << n << " k=" << k);
    std::mt19937 rng(2026u + m*1009u + n*9176u + k);
    std::uniform_int_distribution<int> value(-8,8);
    // Dyadic inputs keep products/sums exactly representable for k <= 512.
    // Thus the absolute 1e-5 requirement does not depend on FMA/reduction order.
    RowMatrix ar(m,k), br(k,n);
    for (int i=0; i<ar.size(); ++i) ar.data()[i] = value(rng)/16.0f;
    for (int i=0; i<br.size(); ++i) br.data()[i] = value(rng)/16.0f;
    const Eigen::MatrixXf a_ref = ar, b_ref = br;
    const Eigen::MatrixXf expected = a_ref*b_ref;
    Matrix<float> a(m,k), b(k,n);
    a.data().copy_from_host(ar.data()); b.data().copy_from_host(br.data());
    const auto c = a*b;
    ASSERT_EQ(c.nrows(), m); ASSERT_EQ(c.ncols(), n);
    RowMatrix actual_row(m,n); c.data().copy_to_host(actual_row.data());
    const Eigen::MatrixXf actual = actual_row;
    EXPECT_TRUE(actual.isApprox(expected, 1e-5f));
    EXPECT_LE((actual-expected).cwiseAbs().maxCoeff(), 1e-5f);
}
INSTANTIATE_TEST_SUITE_P(All343Shapes, ProductTest,
    ::testing::Combine(::testing::Values(1,2,3,127,128,129,512),
                       ::testing::Values(1,2,3,127,128,129,512),
                       ::testing::Values(1,2,3,127,128,129,512)));

TEST(Matrix, NonDyadicInputs) {
    constexpr int m=127,k=129,n=128;
    std::mt19937 rng(42); std::uniform_real_distribution<float> dist(-0.05f,0.05f);
    RowMatrix ar(m,k),br(k,n);
    for(int i=0;i<ar.size();++i) ar.data()[i]=dist(rng);
    for(int i=0;i<br.size();++i) br.data()[i]=dist(rng);
    Matrix<float> a(m,k),b(k,n);
    a.data().copy_from_host(ar.data()); b.data().copy_from_host(br.data());
    auto c=a*b; RowMatrix cr(m,n); c.data().copy_to_host(cr.data());
    Eigen::MatrixXf expected=Eigen::MatrixXf(ar)*Eigen::MatrixXf(br), actual=cr;
    EXPECT_TRUE(actual.isApprox(expected,1e-5f));
    EXPECT_LE((actual-expected).cwiseAbs().maxCoeff(),1e-5f);
}
TEST(Matrix, IdentityAndZero) {
    RowMatrix ar=RowMatrix::Random(3,3), eye=RowMatrix::Identity(3,3), zero=RowMatrix::Zero(3,3), out(3,3);
    Matrix<float> a(3,3),b(3,3); a.data().copy_from_host(ar.data());
    b.data().copy_from_host(eye.data()); (a*b).data().copy_to_host(out.data());
    EXPECT_TRUE(out.isApprox(ar,1e-5f));
    b.data().copy_from_host(zero.data()); (a*b).data().copy_to_host(out.data());
    EXPECT_TRUE(out.isZero(0));
}
TEST(Matrix, IncompatibleAndEmptyDimensions) {
    Matrix<float> a(2,3), b(4,2);
    EXPECT_THROW(a*b,std::invalid_argument);
    auto empty=Matrix<float>(0,3)*Matrix<float>(3,2); EXPECT_EQ(empty.size(),0);
    auto zeros=Matrix<float>(2,0)*Matrix<float>(0,3);
    std::array<float,6> out{}; zeros.data().copy_to_host(out.data());
    for(auto x:out) EXPECT_EQ(x,0);
}
TEST(Matrix, SharedCopyAndSafeMove) {
    Matrix<float> a(2,3); auto b=a;
    EXPECT_EQ(a.data().data(),b.data().data()); EXPECT_NE(&a.view(),&b.view());
    Matrix<float> c; c=b; EXPECT_EQ(c.data().data(),a.data().data());
    const auto ptr=a.data().data(); Matrix<float> moved(std::move(a));
    EXPECT_EQ(moved.data().data(),ptr); EXPECT_EQ(a.size(),0);
    EXPECT_THROW(a.data(),std::logic_error);
    Matrix<float> copy_of_moved_from(a); EXPECT_EQ(copy_of_moved_from.size(),0);
    a=std::move(moved); EXPECT_EQ(a.data().data(),ptr); EXPECT_EQ(moved.size(),0);
    a=std::move(a); EXPECT_EQ(a.data().data(),ptr);
}
TEST(Data, DeepCopyAssignmentAndMove) {
    const std::array<float,3> input{1,2,3}; Data<float> a(3); a.copy_from_host(input.data());
    Data<float> b(a); EXPECT_NE(a.data(),b.data());
    const std::array<float,3> changed{4,5,6}; a.copy_from_host(changed.data());
    std::array<float,3> out{}; b.copy_to_host(out.data()); EXPECT_EQ(out,input);
    Data<float> c; c=b; EXPECT_NE(c.data(),b.data()); c.copy_to_host(out.data()); EXPECT_EQ(out,input);
    auto ptr=c.data(); Data<float> moved(std::move(c)); EXPECT_EQ(moved.data(),ptr); EXPECT_EQ(c.size(),0);
    c=std::move(moved); EXPECT_EQ(c.data(),ptr); EXPECT_EQ(moved.size(),0);
}
TEST(MatrixView, RowMajorAndConstAccess) {
    float values[]{1,2,3,4,5,6}; MatrixView<float> v(values,2,3);
    EXPECT_EQ(v(1,0),4); v(1,2)=9; EXPECT_EQ(v[5],9);
    static_assert(std::is_same_v<decltype(std::as_const(v)(0,0)),const float&>);
}
TEST(Matrix, DoubleProduct) {
    Matrix<double> a(1,2),b(2,1); const double x[]{1.5,2},y[]{2,3}; double out=0;
    a.data().copy_from_host(x); b.data().copy_from_host(y);
    (a*b).data().copy_to_host(&out); EXPECT_DOUBLE_EQ(out,9);
}
TEST(Matrix, SizeOverflow) {
    EXPECT_THROW((Matrix<float>(std::numeric_limits<std::size_t>::max(),2)),std::length_error);
    EXPECT_THROW((Data<float>(std::numeric_limits<std::size_t>::max())),std::length_error);
}
