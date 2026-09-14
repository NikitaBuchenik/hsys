#include "vector.cuh"

template class Data<float>;
template class Data<double>;

template class Vector<float>;
template class Vector<double>;

template void launch_vecadd<float>(
    VectorView<float>, VectorView<float>, VectorView<float>, cudaStream_t
);

template void launch_vecadd<double>(
    VectorView<double>, VectorView<double>, VectorView<double>, cudaStream_t
);
