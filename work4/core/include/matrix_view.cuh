#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <type_traits>
template<class T> class MatrixView {
public:
    __host__ __device__ T* data() const noexcept { return ptr_; }
    __host__ __device__ MatrixView() noexcept=default;
    __host__ __device__ MatrixView(T* p,std::size_t rows,std::size_t cols)noexcept:ptr_(p),rows_(rows),cols_(cols){}
    __host__ __device__ std::size_t size()const noexcept{return rows_*cols_;}
    __host__ __device__ std::size_t nrows()const noexcept{return rows_;}
    __host__ __device__ std::size_t ncols()const noexcept{return cols_;}
    __host__ __device__ T& operator()(std::size_t r,std::size_t c)const noexcept{return ptr_[r*cols_+c];}
private:T* ptr_=nullptr;std::size_t rows_=0,cols_=0;
};
static_assert(std::is_trivially_copyable_v<MatrixView<float>>);
static_assert(std::is_trivially_copyable_v<MatrixView<const float>>);

static_assert(std::is_trivially_copyable_v<MatrixView<const __half>>);
