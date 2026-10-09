#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <type_traits>
template<class AtomT> class MatrixView {
public:
    __host__ __device__ MatrixView() noexcept = default;
    __host__ __device__ MatrixView(AtomT* data, std::size_t rows, std::size_t cols) noexcept
        : data_(data), nrows_(rows), ncols_(cols) {}
    __host__ __device__ std::size_t size() const noexcept { return nrows_*ncols_; }
    __host__ __device__ std::size_t nrows() const noexcept { return nrows_; }
    __host__ __device__ std::size_t ncols() const noexcept { return ncols_; }
    __host__ __device__ AtomT& operator[](std::size_t n) noexcept { return data_[n]; }
    __host__ __device__ const AtomT& operator[](std::size_t n) const noexcept { return data_[n]; }
    __host__ __device__ AtomT& operator()(std::size_t i, std::size_t j) noexcept { return data_[i*ncols_+j]; }
    __host__ __device__ const AtomT& operator()(std::size_t i, std::size_t j) const noexcept { return data_[i*ncols_+j]; }
private:
    AtomT* data_ = nullptr;
    std::size_t nrows_ = 0, ncols_ = 0;
};
static_assert(std::is_trivially_copyable_v<MatrixView<float>>);
static_assert(std::is_trivially_copyable_v<MatrixView<const float>>);
