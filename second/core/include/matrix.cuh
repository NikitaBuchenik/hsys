#pragma once
#include "data.cuh"
#include "matrix_view.cuh"
#include <memory>
#include <limits>
#include <utility>

template<class AtomT> class Matrix {
public:
    Matrix() : Matrix(0, 0) {}
    Matrix(std::size_t rows, std::size_t cols)
        : data_(std::make_shared<Data<AtomT>>(checked_size(rows, cols))),
          view_(data_->data(), rows, cols) {}
    Matrix(const Matrix&) = default;
    Matrix& operator=(const Matrix&) = default;
    Matrix(Matrix&& other) noexcept
        : data_(std::move(other.data_)), view_(std::exchange(other.view_, {})) {}
    Matrix& operator=(Matrix&& other) noexcept {
        if (this != &other) {
            data_ = std::move(other.data_);
            view_ = std::exchange(other.view_, {});
        }
        return *this;
    }
    std::size_t size() const noexcept { return view_.size(); }
    std::size_t nrows() const noexcept { return view_.nrows(); }
    std::size_t ncols() const noexcept { return view_.ncols(); }
    Data<AtomT>& data() { require_data(); return *data_; }
    const Data<AtomT>& data() const { require_data(); return *data_; }
    MatrixView<AtomT>& view() noexcept { return view_; }
    const MatrixView<AtomT>& view() const noexcept { return view_; }
    MatrixView<const AtomT> read_view() const {
        return {data().data(), nrows(), ncols()};
    }
private:
    static std::size_t checked_size(std::size_t rows, std::size_t cols) {
        if (cols && rows > std::numeric_limits<std::size_t>::max()/cols)
            throw std::length_error("Matrix size overflow");
        return rows*cols;
    }
    void require_data() const {
        if (!data_) throw std::logic_error("Matrix was moved from");
    }
    std::shared_ptr<Data<AtomT>> data_;
    MatrixView<AtomT> view_;
};

template<class AtomT>
void launch_matmul(MatrixView<const AtomT> a, MatrixView<const AtomT> b, MatrixView<AtomT> c);
template<class AtomT>
Matrix<AtomT> operator*(const Matrix<AtomT>& a, const Matrix<AtomT>& b);
extern template Matrix<float> operator*(const Matrix<float>&, const Matrix<float>&);
extern template Matrix<double> operator*(const Matrix<double>&, const Matrix<double>&);
