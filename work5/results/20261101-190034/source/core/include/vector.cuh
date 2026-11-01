#pragma once
#include "cuda_check.cuh"
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

// A non-owning, read-only-or-writable view passed to a kernel by value.
template<class T> struct VectorView {
    T* data = nullptr;
    std::size_t size = 0;
};
static_assert(std::is_trivially_copyable_v<VectorView<const float>>);

// Owns GPU memory. Copying is forbidden; moving transfers ownership.
template<class T> class DeviceBuffer {
public:
    explicit DeviceBuffer(std::size_t count = 0) : size_(count) {
        if (count > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::length_error("DeviceBuffer byte count overflow");
        if (count) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)));
    }
    ~DeviceBuffer() { if (data_) (void)cudaFree(data_); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(std::exchange(other.data_, nullptr)), size_(std::exchange(other.size_, 0)) {}
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_) (void)cudaFree(data_);
            data_ = std::exchange(other.data_, nullptr);
            size_ = std::exchange(other.size_, 0);
        }
        return *this;
    }
    T* data() noexcept { return data_; }
    const T* data() const noexcept { return data_; }
    std::size_t size() const noexcept { return size_; }
    VectorView<T> view() noexcept { return {data_, size_}; }
    VectorView<const T> read_view() const noexcept { return {data_, size_}; }
    void copy_from_host(const T* source, std::size_t count) {
        if (count > size_ || (count && !source)) throw std::invalid_argument("Invalid host source");
        if (count) CUDA_CHECK(cudaMemcpy(data_, source, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void copy_to_host(T* destination, std::size_t count) const {
        if (count > size_ || (count && !destination)) throw std::invalid_argument("Invalid host destination");
        if (count) CUDA_CHECK(cudaMemcpy(destination, data_, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
private:
    T* data_ = nullptr;
    std::size_t size_ = 0;
};
