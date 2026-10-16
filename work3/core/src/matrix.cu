#include "matrix.cuh"
#include "kernel_timer.cuh"
#include <stdexcept>
namespace { constexpr unsigned TILE=16; }
template<class T>
__global__ void kernel_matmul_naive(MatrixView<const T> a,MatrixView<const T> b,MatrixView<T> c){
    const auto row=static_cast<std::size_t>(blockIdx.y)*blockDim.y+threadIdx.y;
    const auto col=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
    if(row>=c.nrows()||col>=c.ncols())return;
    T sum=T{0};for(std::size_t k=0;k<a.ncols();++k)sum+=a(row,k)*b(k,col);c(row,col)=sum;
}
template<class T>
__global__ void kernel_matmul_shmem(MatrixView<const T> a,MatrixView<const T> b,MatrixView<T> c){
    __shared__ T tile_a[TILE][TILE];
    __shared__ T tile_b[TILE][TILE];
    const unsigned tx=threadIdx.x,ty=threadIdx.y;
    const std::size_t row=static_cast<std::size_t>(blockIdx.y)*TILE+ty;
    const std::size_t col=static_cast<std::size_t>(blockIdx.x)*TILE+tx;
    T sum=T{0};
    const std::size_t tiles=(a.ncols()+TILE-1)/TILE;
    for(std::size_t t=0;t<tiles;++t){
        const std::size_t ak=t*TILE+tx,bk=t*TILE+ty;
        tile_a[ty][tx]=(row<a.nrows()&&ak<a.ncols())?a(row,ak):T{0};
        tile_b[ty][tx]=(bk<b.nrows()&&col<b.ncols())?b(bk,col):T{0};
        __syncthreads();
        #pragma unroll
        for(unsigned j=0;j<TILE;++j)sum+=tile_a[ty][j]*tile_b[j][tx];
        __syncthreads();
    }
    if(row<c.nrows()&&col<c.ncols())c(row,col)=sum;
}
template<class T> void launch_matmul_naive(MatrixView<const T>a,MatrixView<const T>b,MatrixView<T>c){
    if(a.ncols()!=b.nrows()||c.nrows()!=a.nrows()||c.ncols()!=b.ncols())throw std::invalid_argument("Incompatible matrix dimensions");
    if(!c.size())return;
    dim3 block(TILE,TILE),grid(static_cast<unsigned>((c.ncols()+TILE-1)/TILE),static_cast<unsigned>((c.nrows()+TILE-1)/TILE));
    auto* timer=matrix_detail::active_timer;if(timer)timer->start();kernel_matmul_naive<<<grid,block>>>(a,b,c);if(timer)timer->stop();CUDA_CHECK(cudaGetLastError());
}
template<class T> void launch_matmul_shmem(MatrixView<const T>a,MatrixView<const T>b,MatrixView<T>c){
    if(a.ncols()!=b.nrows()||c.nrows()!=a.nrows()||c.ncols()!=b.ncols())throw std::invalid_argument("Incompatible matrix dimensions");
    if(!c.size())return;
    dim3 block(TILE,TILE),grid(static_cast<unsigned>((c.ncols()+TILE-1)/TILE),static_cast<unsigned>((c.nrows()+TILE-1)/TILE));
    auto* timer=matrix_detail::active_timer;if(timer)timer->start();kernel_matmul_shmem<<<grid,block>>>(a,b,c);if(timer)timer->stop();CUDA_CHECK(cudaGetLastError());
}
template<class T> Matrix<T> multiply(const Matrix<T>&a,const Matrix<T>&b,MatmulStrategy strategy){
    if(a.ncols()!=b.nrows())throw std::invalid_argument("A.cols must equal B.rows");
    Matrix<T> out(a.nrows(),b.ncols());
    if(out.size()){if(strategy==MatmulStrategy::SharedMemory)launch_matmul_shmem(a.read_view(),b.read_view(),out.view());else launch_matmul_naive(a.read_view(),b.read_view(),out.view());CUDA_CHECK(cudaDeviceSynchronize());}
    return out;
}
template Matrix<float> operator*(const Matrix<float>&,const Matrix<float>&);
template Matrix<float> multiply_naive(const Matrix<float>&,const Matrix<float>&);
template Matrix<float> multiply(const Matrix<float>&,const Matrix<float>&,MatmulStrategy);
template void launch_matmul_naive(MatrixView<const float>,MatrixView<const float>,MatrixView<float>);
template void launch_matmul_shmem(MatrixView<const float>,MatrixView<const float>,MatrixView<float>);
