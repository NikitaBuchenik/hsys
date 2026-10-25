#pragma once
#include "data.cuh"
#include "matrix_view.cuh"
#include <memory>
#include <limits>
#include <stdexcept>
#include <utility>
template<class T> class Matrix {
public:
    Matrix():Matrix(0,0){}
    Matrix(std::size_t r,std::size_t c):data_(std::make_shared<Data<T>>(checked(r,c))),view_(data_->data(),r,c){}
    Matrix(const Matrix&)=default;Matrix& operator=(const Matrix&)=default;
    Matrix(Matrix&& o)noexcept:data_(std::move(o.data_)),view_(std::exchange(o.view_,{})){}
    Matrix& operator=(Matrix&& o)noexcept{if(this!=&o){data_=std::move(o.data_);view_=std::exchange(o.view_,{});}return *this;}
    std::size_t size()const noexcept{return view_.size();}
    std::size_t nrows()const noexcept{return view_.nrows();}
    std::size_t ncols()const noexcept{return view_.ncols();}
    Data<T>& data(){require();return *data_;} const Data<T>& data()const{require();return *data_;}
    MatrixView<T>& view()noexcept{return view_;} const MatrixView<T>& view()const noexcept{return view_;}
    MatrixView<const T> read_view()const{return {data().data(),nrows(),ncols()};}
private:
    static std::size_t checked(std::size_t r,std::size_t c){if(c&&r>std::numeric_limits<std::size_t>::max()/c)throw std::length_error("Matrix size overflow");return r*c;}
    void require()const{if(!data_)throw std::logic_error("Matrix was moved from");}
    std::shared_ptr<Data<T>> data_; MatrixView<T> view_;
};
enum class MatmulStrategy { Wmma, SharedMemory };
// FP16 inputs, FP32 accumulation and output.
Matrix<float> multiply(const Matrix<__half>&, const Matrix<__half>&, MatmulStrategy);
Matrix<float> multiply_shared(const Matrix<float>&, const Matrix<float>&);
inline Matrix<float> operator*(const Matrix<__half>& a, const Matrix<__half>& b) {
    return multiply(a, b, MatmulStrategy::Wmma);
}
// Previous work3 operator: FP32 shared-memory algorithm.
inline Matrix<float> operator*(const Matrix<float>& a, const Matrix<float>& b) {
    return multiply_shared(a, b);
}
