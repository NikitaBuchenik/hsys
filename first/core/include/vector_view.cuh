#pragma once

#include <cstddef>
#include <type_traits>

template <typename AtomT>
class VectorView {
public:
    using value_type = AtomT;

    __host__ __device__
    VectorView() noexcept : data_(nullptr), size_(0) {}

    __host__ __device__
    VectorView(AtomT* data, std::size_t size) noexcept
        : data_(data), size_(size) {}

    __host__ __device__
    std::size_t size() const noexcept {
        return size_;
    }

    __host__ __device__
    AtomT& operator[](std::size_t index) const noexcept {
        return data_[index];
    }

    __host__ __device__
    AtomT& operator()(std::size_t index) const noexcept {
        return data_[index];
    }

    __host__ __device__
    AtomT* data() const noexcept {
        return data_;
    }

private:
    AtomT* data_;
    std::size_t size_;
};

static_assert(std::is_trivially_copyable_v<VectorView<float>>);
static_assert(std::is_trivially_copyable_v<VectorView<double>>);
