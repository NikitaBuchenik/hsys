#include "matrix.cuh"
#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <array>
#include <random>
#include <tuple>
using RowMatrix=Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;
using Shape=std::tuple<int,int,int>;
class ProductTest:public ::testing::TestWithParam<Shape>{};
TEST_P(ProductTest, MatchesEigenAbsoluteTolerance){
 const auto [m,k,n]=GetParam();SCOPED_TRACE(::testing::Message()<<"m="<<m<<" k="<<k<<" n="<<n);
 std::mt19937 rng(2026u+m*1009u+k*9176u+n);std::uniform_int_distribution<int>d(-8,8);
 RowMatrix ar(m,k),br(k,n);for(int i=0;i<ar.size();++i)ar.data()[i]=d(rng)/16.0f;for(int i=0;i<br.size();++i)br.data()[i]=d(rng)/16.0f;
 Eigen::MatrixXf expected=Eigen::MatrixXf(ar)*Eigen::MatrixXf(br);
 Matrix<float>a(m,k),b(k,n);a.data().copy_from_host(ar.data());b.data().copy_from_host(br.data());auto c=a*b;
 RowMatrix cr(m,n);c.data().copy_to_host(cr.data());Eigen::MatrixXf actual=cr;
 EXPECT_TRUE(actual.isApprox(expected,1e-5f));EXPECT_LE((actual-expected).cwiseAbs().maxCoeff(),1e-5f);
}
INSTANTIATE_TEST_SUITE_P(All343Shapes,ProductTest,::testing::Combine(
 ::testing::Values(1,2,3,127,128,129,512),::testing::Values(1,2,3,127,128,129,512),::testing::Values(1,2,3,127,128,129,512)));
TEST(Matrix, NonMultipleTileDimensions){
 constexpr int m=127,k=129,n=128;std::mt19937 rng(7);std::uniform_real_distribution<float>d(-.05f,.05f);
 RowMatrix ar(m,k),br(k,n);for(int i=0;i<ar.size();++i)ar.data()[i]=d(rng);for(int i=0;i<br.size();++i)br.data()[i]=d(rng);
 Matrix<float>a(m,k),b(k,n);a.data().copy_from_host(ar.data());b.data().copy_from_host(br.data());auto c=a*b;
 RowMatrix cr(m,n);c.data().copy_to_host(cr.data());Eigen::MatrixXf actual=cr,expected=Eigen::MatrixXf(ar)*Eigen::MatrixXf(br);
 EXPECT_TRUE(actual.isApprox(expected,1e-5f));EXPECT_LE((actual-expected).cwiseAbs().maxCoeff(),1e-5f);
}
TEST(Matrix, InvalidDimensionsThrow){Matrix<float>a(2,3),b(4,2);EXPECT_THROW(a*b,std::invalid_argument);}
TEST(Matrix, ZeroInnerDimension){Matrix<float>a(2,0),b(0,3);auto c=a*b;std::array<float,6> out{};c.data().copy_to_host(out.data());for(float x:out)EXPECT_FLOAT_EQ(x,0);}
TEST(Matrix, NaiveStrategyMatchesShared){RowMatrix ar=RowMatrix::Random(17,19),br=RowMatrix::Random(19,13);Matrix<float>a(17,19),b(19,13);a.data().copy_from_host(ar.data());b.data().copy_from_host(br.data());auto x=a*b,y=multiply_naive(a,b);RowMatrix xr(17,13),yr(17,13);x.data().copy_to_host(xr.data());y.data().copy_to_host(yr.data());EXPECT_TRUE(Eigen::MatrixXf(xr).isApprox(Eigen::MatrixXf(yr),1e-5f));}
