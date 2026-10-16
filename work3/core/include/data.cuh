#pragma once
#include "cuda_check.cuh"
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
template<class T> class Data {
    static_assert(std::is_arithmetic_v<T>);
public:
    Data() noexcept = default;
    explicit Data(std::size_t size): size_(size) {
        if (size > std::numeric_limits<std::size_t>::max()/sizeof(T)) throw std::length_error("Data size overflow");
        if (size_) CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&ptr_), size_*sizeof(T)));
    }
    Data(const Data& other): Data(other.size_) {
        if(size_) { auto e=cudaMemcpy(ptr_,other.ptr_,size_*sizeof(T),cudaMemcpyDeviceToDevice); if(e!=cudaSuccess){ release(); CUDA_CHECK(e); } }
    }
    Data(Data&& other) noexcept: size_(std::exchange(other.size_,0)),ptr_(std::exchange(other.ptr_,nullptr)) {}
    Data& operator=(const Data& other) { if(this!=&other){ Data copy(other); swap(copy); } return *this; }
    Data& operator=(Data&& other) noexcept { if(this!=&other){ release(); swap(other); } return *this; }
    ~Data(){ release(); }
    T* data() noexcept{return ptr_;} const T* data()const noexcept{return ptr_;}
    std::size_t size()const noexcept{return size_;}
    void copy_from_host(const T* p){if(size_&&!p)throw std::invalid_argument("Null host source");if(size_)CUDA_CHECK(cudaMemcpy(ptr_,p,size_*sizeof(T),cudaMemcpyHostToDevice));}
    void copy_to_host(T* p)const{if(size_&&!p)throw std::invalid_argument("Null host destination");if(size_)CUDA_CHECK(cudaMemcpy(p,ptr_,size_*sizeof(T),cudaMemcpyDeviceToHost));}
    void swap(Data& o)noexcept{std::swap(size_,o.size_);std::swap(ptr_,o.ptr_);}
private:
    void release()noexcept{if(ptr_)(void)cudaFree(ptr_);ptr_=nullptr;size_=0;}
    std::size_t size_=0; T* ptr_=nullptr;
};
