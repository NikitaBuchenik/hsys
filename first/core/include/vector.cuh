#pragma once

#include "cuda_check.cuh"
#include "data.cuh"
#include "vector_view.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <memory>
#include <stdexcept>

template <typename AtomT>
__global__ void kernel_vecadd(
    VectorView<AtomT> a,
    VectorView<AtomT> b,
    VectorView<AtomT> result
) {
    const std::size_t index =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if (index < result.size()) {
        result[index] = a[index] + b[index];
    }
}

template <typename AtomT>
inline void launch_vecadd(
    VectorView<AtomT> a,
    VectorView<AtomT> b,
    VectorView<AtomT> result,
    cudaStream_t stream = nullptr
) {
    if (a.size() != b.size() || a.size() != result.size()) {
        throw std::invalid_argument("Vector sizes must match");
    }

    if (result.size() == 0) {
        return;
    }

    constexpr unsigned int block_size = 256;
    const unsigned int grid_size =
        static_cast<unsigned int>(
            (result.size() + block_size - 1) / block_size
        );

    kernel_vecadd<<<grid_size, block_size, 0, stream>>>(a, b, result);
    CUDA_CHECK(cudaGetLastError());
}

template <typename AtomT>
class Vector {
public:
    Vector() : data_(std::make_shared<Data<AtomT>>()), view_() {}

    explicit Vector(std::size_t size)
        : data_(std::make_shared<Data<AtomT>>(size)),
          view_(data_->data(), size) {}

    Vector(const Vector& other)
        : data_(other.data_), view_(data_->data(), data_->size()) {}

    Vector(Vector&& other) noexcept
        : data_(std::move(other.data_)),
          view_(data_ ? VectorView<AtomT>(data_->data(), data_->size())
                      : VectorView<AtomT>()) {
        other.view_ = VectorView<AtomT>();
    }

    Vector& operator=(const Vector& other) {
        if (this != &other) {
            data_ = other.data_;
            view_ = VectorView<AtomT>(data_->data(), data_->size());
        }
        return *this;
    }

    Vector& operator=(Vector&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            view_ = data_
                ? VectorView<AtomT>(data_->data(), data_->size())
                : VectorView<AtomT>();
            other.view_ = VectorView<AtomT>();
        }
        return *this;
    }

    std::size_t size() const noexcept {
        return data_->size();
    }

    AtomT* data() noexcept {
        return data_->data();
    }

    const AtomT* data() const noexcept {
        return data_->data();
    }

    VectorView<AtomT> view() noexcept {
        return view_;
    }

    VectorView<const AtomT> view() const noexcept {
        return VectorView<const AtomT>(
            const_cast<AtomT*>(data_->data()), data_->size()
        );
    }

    void copy_from_host(const AtomT* host_data) {
        data_->copy_from_host(host_data);
    }

    void copy_to_host(AtomT* host_data) const {
        data_->copy_to_host(host_data);
    }

private:
    std::shared_ptr<Data<AtomT>> data_;
    VectorView<AtomT> view_;
};

template <typename AtomT>
Vector<AtomT> operator+(const Vector<AtomT>& a, const Vector<AtomT>& b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("Vector sizes must match");
    }

    Vector<AtomT> result(a.size());
    launch_vecadd(
        VectorView<AtomT>(const_cast<AtomT*>(a.data()), a.size()),
        VectorView<AtomT>(const_cast<AtomT*>(b.data()), b.size()),
        result.view()
    );
    CUDA_CHECK(cudaDeviceSynchronize());
    return result;
}
