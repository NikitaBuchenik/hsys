#include "matrix.cuh"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

using HalfMatrix = Eigen::Matrix<Eigen::half,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using RowMatrix = Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

void check_product(int m,int k,int n,bool fractional=false) {
    HalfMatrix a(m,k), b(k,n);
    std::mt19937 rng(1729u + m*3u + k*5u + n*7u);
    std::uniform_int_distribution<int> exact(-1,1);
    std::uniform_real_distribution<float> real(-0.01f,0.01f);
    auto fill = [&](HalfMatrix& x) {
        for (Eigen::Index i=0;i<x.size();++i)
            x.data()[i]=Eigen::half(fractional?real(rng):exact(rng)/16.0f);
    };
    fill(a); fill(b);
    std::vector<__half> ah(a.size()),bh(b.size());
    for (std::size_t i=0;i<ah.size();++i) ah[i]=__float2half_rn(static_cast<float>(a.data()[i]));
    for (std::size_t i=0;i<bh.size();++i) bh[i]=__float2half_rn(static_cast<float>(b.data()[i]));
    Matrix<__half> ga(m,k),gb(k,n);
    ga.data().copy_from_host(ah.data()); gb.data().copy_from_host(bh.data());
    auto gc=ga*gb;
    ASSERT_EQ(gc.nrows(),m); ASSERT_EQ(gc.ncols(),n);
    RowMatrix got(m,n); gc.data().copy_to_host(got.data());
    // Required reference: actual Eigen half matrix product, then cast to MatrixXf.
    const HalfMatrix half_reference=(a*b).eval();
    const Eigen::MatrixXf reference=half_reference.cast<float>();
    const Eigen::MatrixXf actual=got;
    ASSERT_TRUE(actual.allFinite());
    EXPECT_TRUE(actual.isApprox(reference,1e-2f)); // Eigen isApprox is relative.
    EXPECT_LE((actual-reference).cwiseAbs().maxCoeff(),1e-2f); // Absolute check as well.
    // Independent FP32 reference guards against shared half-precision errors.
    const Eigen::MatrixXf fp32=(a.cast<float>()*b.cast<float>()).eval();
    EXPECT_LE((actual-fp32).cwiseAbs().maxCoeff(),1e-2f);
    auto control=multiply(ga,gb,MatmulStrategy::SharedMemory);
    RowMatrix ctrl(m,n); control.data().copy_to_host(ctrl.data());
    EXPECT_LE((got-ctrl).cwiseAbs().maxCoeff(),1e-2f);
    const RowMatrix fa=a.cast<float>(),fb=b.cast<float>();
    Matrix<float> fga(m,k),fgb(k,n);
    fga.data().copy_from_host(fa.data()); fgb.data().copy_from_host(fb.data());
    auto previous=fga*fgb;
    previous.data().copy_to_host(ctrl.data());
    EXPECT_LE((got-ctrl).cwiseAbs().maxCoeff(),1e-2f);
}
class ProductTest : public ::testing::TestWithParam<std::tuple<int,int,int>> {};
TEST_P(ProductTest, MatchesEigenHalf) {
    auto [m,k,n]=GetParam(); check_product(m,k,n);
}
INSTANTIATE_TEST_SUITE_P(All216Shapes,ProductTest,::testing::Combine(
    ::testing::Values(16,32,64,128,256,512),
    ::testing::Values(16,32,64,128,256,512),
    ::testing::Values(16,32,64,128,256,512)));
TEST(WmmaEdges, PartialBlock48x80) { check_product(48,48,80); }
TEST(WmmaEdges, NonDyadicInputs) { check_product(32,64,48,true); }
TEST(WmmaEdges, ZeroInnerDimension) {
    Matrix<__half> a(16,0),b(0,32); auto c=a*b;
    std::vector<float> host(c.size(),42); c.data().copy_to_host(host.data());
    for (float x:host) EXPECT_EQ(x,0);
}
TEST(WmmaEdges, EmptyOutput) {
    Matrix<__half> a(0,16),b(16,32); auto c=a*b; EXPECT_EQ(c.size(),0);
    Matrix<__half> d(16,16),e(16,0); auto f=d*e; EXPECT_EQ(f.size(),0);
}
TEST(WmmaEdges, IncompatibleDimensions) {
    Matrix<__half> a(16,32),b(16,16); EXPECT_THROW(a*b,std::invalid_argument);
}
TEST(WmmaEdges, RejectsNonMultipleDimensions) {
    Matrix<__half> a(17,16),b(16,16); EXPECT_THROW(a*b,std::invalid_argument);
    Matrix<__half> c(16,17),d(17,16); EXPECT_THROW(c*d,std::invalid_argument);
    Matrix<__half> e(16,16),f(16,17); EXPECT_THROW(e*f,std::invalid_argument);
}
TEST(Ownership, DeepDataCopyAndMove) {
    Data<__half> a(1); __half one=__float2half_rn(1),two=__float2half_rn(2),got;
    a.copy_from_host(&one); Data<__half> b(a); a.copy_from_host(&two);
    b.copy_to_host(&got); EXPECT_EQ(__half2float(got),1);
    Data<__half> c; c=b; Data<__half> d(std::move(c));
    EXPECT_EQ(c.size(),0); a=std::move(d); EXPECT_EQ(d.size(),0);
    a.copy_to_host(&got); EXPECT_EQ(__half2float(got),1);
}
TEST(Ownership, MatrixSharesAndMoves) {
    Matrix<__half> a(16,16),b=a;
    EXPECT_EQ(a.data().data(),b.data().data());
    Matrix<__half> c(std::move(b)); EXPECT_EQ(b.size(),0);
    EXPECT_THROW(b.data(),std::logic_error);
    b=std::move(c); EXPECT_EQ(b.nrows(),16); EXPECT_EQ(c.size(),0);
}
TEST(Ownership, Overflow) {
    EXPECT_THROW((Matrix<__half>(std::numeric_limits<std::size_t>::max(),2)),std::length_error);
    EXPECT_THROW((Data<float>(std::numeric_limits<std::size_t>::max())),std::length_error);
}
