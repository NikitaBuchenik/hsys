#include "matrix.cuh"
#include "kernel_timer.cuh"
#include <mma.h>
#include <climits>

namespace {
constexpr unsigned TILE = 16;
constexpr unsigned BLOCK_ROWS = 32, BLOCK_COLS = 64, THREADS = 256;
template<class T> __device__ float as_float(T x) { return static_cast<float>(x); }
template<> __device__ float as_float(__half x) { return __half2float(x); }

// Float version preserves work3's algorithm; half version is a same-input control.
template<class T>
__global__ void kernel_matmul_shmem(MatrixView<const T> a,
                                    MatrixView<const T> b, MatrixView<float> c) {
    __shared__ float tile_a[TILE][TILE];
    __shared__ float tile_b[TILE][TILE];
    const unsigned tx = threadIdx.x, ty = threadIdx.y;
    const std::size_t row = static_cast<std::size_t>(blockIdx.y)*TILE+ty;
    const std::size_t col = static_cast<std::size_t>(blockIdx.x)*TILE+tx;
    float sum = 0;
    const std::size_t tiles = a.ncols()/TILE + (a.ncols()%TILE != 0);
    for (std::size_t t = 0; t < tiles; ++t) {
        const auto ak = t*TILE+tx, bk = t*TILE+ty;
        tile_a[ty][tx] = row<a.nrows() && ak<a.ncols() ? as_float(a(row,ak)) : 0;
        tile_b[ty][tx] = bk<b.nrows() && col<b.ncols() ? as_float(b(bk,col)) : 0;
        __syncthreads();
        #pragma unroll
        for (unsigned j = 0; j < TILE; ++j) sum += tile_a[ty][j]*tile_b[j][tx];
        __syncthreads();
    }
    if (row<c.nrows() && col<c.ncols()) c(row,col) = sum;
}
}

// Eight whole warps, 2x4 output tiles, matching the assignment figure.
__global__ void kernel_matmul_wmma(MatrixView<const __half> a,
                                  MatrixView<const __half> b, MatrixView<float> c) {
    using namespace nvcuda;
    __shared__ __align__(32) __half tile_a[BLOCK_ROWS][TILE];
    __shared__ __align__(32) __half tile_b[TILE][BLOCK_COLS];
    const unsigned tid = threadIdx.x, warp = tid/32;
    const unsigned wr = (warp%2)*TILE, wc = (warp/2)*TILE;
    const std::size_t row0 = static_cast<std::size_t>(blockIdx.y)*BLOCK_ROWS;
    const std::size_t col0 = static_cast<std::size_t>(blockIdx.x)*BLOCK_COLS;
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> bf;
    wmma::fragment<wmma::accumulator,16,16,16,float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (std::size_t k0 = 0; k0 < a.ncols(); k0 += TILE) {
        for (unsigned i = tid; i < BLOCK_ROWS*TILE; i += THREADS) {
            const unsigned r = i/TILE, k = i%TILE;
            tile_a[r][k] = row0+r<a.nrows() ? a(row0+r,k0+k) : __float2half_rn(0);
        }
        for (unsigned i = tid; i < TILE*BLOCK_COLS; i += THREADS) {
            const unsigned k = i/BLOCK_COLS, col = i%BLOCK_COLS;
            tile_b[k][col] = col0+col<b.ncols() ? b(k0+k,col0+col) : __float2half_rn(0);
        }
        __syncthreads();
        // Each warp uses identical 32-byte-aligned addresses and strides.
        wmma::load_matrix_sync(af, &tile_a[wr][0], TILE);
        wmma::load_matrix_sync(bf, &tile_b[0][wc], BLOCK_COLS);
        wmma::mma_sync(acc, af, bf, acc);
        __syncthreads();
    }
    // Uniform per warp. Public WMMA API requires dimensions divisible by 16.
    // Therefore a valid warp always stores a complete 16x16 tile.
    if (row0+wr<c.nrows() && col0+wc<c.ncols())
        wmma::store_matrix_sync(c.data()+(row0+wr)*c.ncols()+col0+wc,
                               acc, static_cast<unsigned>(c.ncols()), wmma::mem_row_major);
}

namespace {
template<class T> void validate(const Matrix<T>& a, const Matrix<T>& b, bool wmma) {
    (void)a.data(); (void)b.data();
    if (a.ncols()!=b.nrows()) throw std::invalid_argument("A.cols must equal B.rows");
    if (wmma && (a.nrows()%16 || a.ncols()%16 || b.ncols()%16))
        throw std::invalid_argument("WMMA dimensions must be multiples of 16 (zero allowed)");
    if (b.ncols()>UINT_MAX) throw std::length_error("Leading dimension exceeds unsigned");
}
dim3 grid_for(std::size_t rows, std::size_t cols, unsigned br, unsigned bc) {
    const auto gx = cols/bc+(cols%bc!=0), gy = rows/br+(rows%br!=0);
    if (gx>2147483647ULL || gy>65535ULL) throw std::length_error("CUDA grid too large");
    return dim3(static_cast<unsigned>(gx),static_cast<unsigned>(gy));
}
template<class T>
Matrix<float> run(const Matrix<T>& a, const Matrix<T>& b, MatmulStrategy strategy) {
    const bool wmma = strategy==MatmulStrategy::Wmma;
    validate(a,b,wmma);
    if (strategy!=MatmulStrategy::Wmma && strategy!=MatmulStrategy::SharedMemory)
        throw std::invalid_argument("Unknown strategy");
    Matrix<float> out(a.nrows(),b.ncols());
    if (!out.size()) return out;
    const auto grid = grid_for(out.nrows(),out.ncols(),wmma?BLOCK_ROWS:TILE,wmma?BLOCK_COLS:TILE);
    auto* timer = matrix_detail::active_timer;
    if (timer) timer->start();
    if constexpr (std::is_same_v<T,__half>) {
        if (wmma) kernel_matmul_wmma<<<grid,THREADS>>>(a.read_view(),b.read_view(),out.view());
        else kernel_matmul_shmem<<<grid,dim3(TILE,TILE)>>>(a.read_view(),b.read_view(),out.view());
    } else kernel_matmul_shmem<<<grid,dim3(TILE,TILE)>>>(a.read_view(),b.read_view(),out.view());
    if (timer) timer->stop();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    return out;
}
}
Matrix<float> multiply(const Matrix<__half>& a,const Matrix<__half>& b,MatmulStrategy s) {
    return run(a,b,s);
}
Matrix<float> multiply_shared(const Matrix<float>& a,const Matrix<float>& b) {
    return run(a,b,MatmulStrategy::SharedMemory);
}
