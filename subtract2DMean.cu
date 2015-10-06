#include <stdio.h>
#include "cuda_funcs.h"
#incluce "arrayfire.h"

__global__ void subtract2DMean(float *phz_lo, float *meanVec, int N, int NZ) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int k = threadIdx.z + blockIdx.z*blockDim.z;

	int index = k*N*N + j*N + i;

	if (i < N && j < N && k < NZ) {
		phz_lo[index] = phz_lo[index] - meanVec[k];
	}

}

af::array subtract2DMean(float *out, dataSize size) {

	int N = size.x;
	int NZ = size.z;

	af::array phz_lo(size.x*size.y, size.z, out, afDevice);
	af::array meanVec = af::mean(phz_lo, 0);

	float * d_phz_lo = phz_lo.device<float>();
	float * d_meanVec = meanVec.device<float>();

	dim3 dimGrid (int((N-0.5)/BSZ) + 1, int((N-0.5)/BSZ) + 1, NZ);
	dim3 dimBlock (BSZ, BSZ, 1);
	subtract2DMean_kernel<<<dimGrid, dimBlock>>>(d_phz_lo, d_meanVec, N, NZ);

	return phz_lo;
}
