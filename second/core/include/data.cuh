#pragma once
#include "cuda_check.cuh"
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

template<class AtomT> class Data {
    static_assert(std::is_arithmetic_v<AtomT>);
public:
    Data() noexcept = default;
    explicit Data(std::size_t size) : size_(size) {
        if (size > std::numeric_limits<std::size_t>::max() / sizeof(AtomT))
            throw std::length_error("Data byte count overflow");
        if (size_) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), size_ * sizeof(AtomT)));
    }
    Data(const Data& other) : Data(other.size_) {
        // Release the allocation explicitly if the deep copy fails.
        if (size_) {
            const auto error = cudaMemcpy(data_, other.data_, size_ * sizeof(AtomT), cudaMemcpyDeviceToDevice);
            if (error != cudaSuccess) { release(); CUDA_CHECK(error); }
        }
    }
    Data(Data&& other) noexcept
        : size_(std::exchange(other.size_, 0)), data_(std::exchange(other.data_, nullptr)) {}
    Data& operator=(const Data& other) {
        if (this != &other) { Data copy(other); swap(copy); }
        return *this;
    }
    Data& operator=(Data&& other) noexcept {
        if (this != &other) { release(); swap(other); }
        return *this;
    }
    ~Data() { release(); }
    AtomT* data() noexcept { return data_; }
    const AtomT* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    void copy_to_host(AtomT* host) const {
        if (size_ && !host) throw std::invalid_argument("Null host destination");
        if (size_) CUDA_CHECK(cudaMemcpy(host, data_, size_*sizeof(AtomT), cudaMemcpyDeviceToHost));
    }
    void copy_from_host(const AtomT* host) {
        if (size_ && !host) throw std::invalid_argument("Null host source");
        if (size_) CUDA_CHECK(cudaMemcpy(data_, host, size_*sizeof(AtomT), cudaMemcpyHostToDevice));
    }
    void swap(Data& other) noexcept { std::swap(size_, other.size_); std::swap(data_, other.data_); }
private:
    void release() noexcept {
        if (data_) (void)cudaFree(data_); // Destructors must not throw.
        data_ = nullptr; size_ = 0;
    }
    std::size_t size_ = 0;
    AtomT* data_ = nullptr;
};
