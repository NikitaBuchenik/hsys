#include "matrix.cuh"
#include "kernel_timer.cuh"
#include <Eigen/Dense>
#include <benchmark/benchmark.h>
#include <random>
#include <stdexcept>
#include <string>
static Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> input(int n,unsigned seed){
 Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>x(n,n);std::mt19937 g(seed);std::uniform_int_distribution<int>d(-8,8);for(int i=0;i<x.size();++i)x.data()[i]=d(g)/16.f;return x;
}
static void BM_Cuda(benchmark::State&s){
 try{int n=static_cast<int>(s.range(0));auto ar=input(n,11),br=input(n,29);Matrix<float>a(n,n),b(n,n);a.data().copy_from_host(ar.data());b.data().copy_from_host(br.data());
 for(int i=0;i<3;++i){auto warm=a*b;}KernelTimer timer;
 for(auto _:s){ScopedKernelTimer scope(timer);auto c=a*b;benchmark::DoNotOptimize(c.data().data());s.SetIterationTime(timer.seconds());}
 }catch(const std::exception&e){s.SkipWithError(e.what());}
}
static void BM_CudaNaive(benchmark::State&s){
 try{int n=static_cast<int>(s.range(0));auto ar=input(n,11),br=input(n,29);Matrix<float>a(n,n),b(n,n);a.data().copy_from_host(ar.data());b.data().copy_from_host(br.data());
 for(int i=0;i<3;++i){auto warm=multiply_naive(a,b);}KernelTimer timer;
 for(auto _:s){ScopedKernelTimer scope(timer);auto c=multiply_naive(a,b);benchmark::DoNotOptimize(c.data().data());s.SetIterationTime(timer.seconds());}
 }catch(const std::exception&e){s.SkipWithError(e.what());}
}
BENCHMARK(BM_Cuda)->RangeMultiplier(2)->Range(16,1024)->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_CudaNaive)->RangeMultiplier(2)->Range(16,1024)->UseManualTime()->Unit(benchmark::kMicrosecond);
BENCHMARK_MAIN();
