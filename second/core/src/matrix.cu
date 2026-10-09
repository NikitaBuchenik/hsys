#include "matrix.cuh"
#include "kernel_timer.cuh"

template<class AtomT>
__global__ void kernel_matmul_naive(MatrixView<const AtomT> a,
                                    MatrixView<const AtomT> b, MatrixView<AtomT> c) {
    const std::size_t row = static_cast<std::size_t>(blockIdx.y)*blockDim.y + threadIdx.y;
    const std::size_t col = static_cast<std::size_t>(blockIdx.x)*blockDim.x + threadIdx.x;
    if (row >= c.nrows() || col >= c.ncols()) return;
    AtomT sum = AtomT{0};
    for (std::size_t k = 0; k < a.ncols(); ++k)
        sum += a(row, k) * b(k, col);
    c(row, col) = sum;
}

template<class AtomT>
void launch_matmul(MatrixView<const AtomT> a, MatrixView<const AtomT> b, MatrixView<AtomT> c) {
    if (a.ncols() != b.nrows() || c.nrows() != a.nrows() || c.ncols() != b.ncols())
        throw std::invalid_argument("Incompatible matrix dimensions");
    if (!c.size()) return;
    constexpr unsigned side = 16;
    const auto gx = c.ncols()/side + (c.ncols()%side != 0);
    const auto gy = c.nrows()/side + (c.nrows()%side != 0);
    if (gx > 2147483647ULL || gy > 65535ULL)
        throw std::length_error("Matrix exceeds CUDA grid limits");
    const dim3 block(side, side), grid(static_cast<unsigned>(gx), static_cast<unsigned>(gy));
    auto* timer = matrix_detail::active_timer;
    if (timer) timer->start();
    kernel_matmul_naive<<<grid, block>>>(a, b, c);
    if (timer) timer->stop();
    CUDA_CHECK(cudaGetLastError());
}

template<class AtomT>
Matrix<AtomT> operator*(const Matrix<AtomT>& a, const Matrix<AtomT>& b) {
    if (a.ncols() != b.nrows()) throw std::invalid_argument("A.cols must equal B.rows");
    Matrix<AtomT> result(a.nrows(), b.ncols());
    if (result.size()) {
        launch_matmul(a.read_view(), b.read_view(), result.view());
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    return result;
}
template Matrix<float> operator*(const Matrix<float>&, const Matrix<float>&);
template Matrix<double> operator*(const Matrix<double>&, const Matrix<double>&);
template void launch_matmul(MatrixView<const float>, MatrixView<const float>, MatrixView<float>);
template void launch_matmul(MatrixView<const double>, MatrixView<const double>, MatrixView<double>);
