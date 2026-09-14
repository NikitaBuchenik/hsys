#pragma once

#include "cuda_check.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <utility>

template <typename AtomT>
class Data {
public:
    Data() noexcept : size_(0), data_(nullptr) {}

    explicit Data(std::size_t size) : size_(size), data_(nullptr) {
        if (size_ != 0) {
            CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_),
                                  size_ * sizeof(AtomT)));
        }
    }

    Data(const Data& other) : Data(other.size_) {
        if (size_ != 0) {
            CUDA_CHECK(cudaMemcpy(
                data_, other.data_, size_ * sizeof(AtomT),
                cudaMemcpyDeviceToDevice
            ));
        }
    }

    Data(Data&& other) noexcept
        : size_(std::exchange(other.size_, 0)),
          data_(std::exchange(other.data_, nullptr)) {}

    Data& operator=(const Data& other) {
        if (this == &other) {
            return *this;
        }

        Data tmp(other);
        swap(tmp);
        return *this;
    }

    Data& operator=(Data&& other) noexcept {
        if (this == &other) {
            return *this;
        }

        release();
        size_ = std::exchange(other.size_, 0);
        data_ = std::exchange(other.data_, nullptr);
        return *this;
    }

    ~Data() {
        release();
    }

    AtomT* data() noexcept {
        return data_;
    }

    const AtomT* data() const noexcept {
        return data_;
    }

    std::size_t size() const noexcept {
        return size_;
    }

    void copy_to_host(AtomT* host_data) const {
        if (size_ != 0) {
            CUDA_CHECK(cudaMemcpy(
                host_data, data_, size_ * sizeof(AtomT),
                cudaMemcpyDeviceToHost
            ));
        }
    }

    void copy_from_host(const AtomT* host_data) {
        if (size_ != 0) {
            CUDA_CHECK(cudaMemcpy(
                data_, host_data, size_ * sizeof(AtomT),
                cudaMemcpyHostToDevice
            ));
        }
    }

    void swap(Data& other) noexcept {
        std::swap(size_, other.size_);
        std::swap(data_, other.data_);
    }

private:
    void release() noexcept {
        if (data_ != nullptr) {
            // Destructors cannot sensibly propagate CUDA errors.
            cudaFree(data_);
            data_ = nullptr;
        }
        size_ = 0;
    }

    std::size_t size_;
    AtomT* data_;
};
