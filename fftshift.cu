#include <stdio.h>
#include "cuda_funcs.h"

__global__ void fftshift_kernel(cufftComplex *out, cufftComplex* in, int N, int NZ) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int k = threadIdx.z + blockIdx.z*blockDim.z;

	int index = k*N*N + j*N + i;

	if (i < N && j < N && k < NZ) {
		int eq1 = (N*N + N)/2;
		int eq2 = (N*N - N)/2;

		if (i < N/2)  {
			if (j < N/2) {   //Q1
				out[index] = in[index + eq1];
			} else {		 //Q3
				out[index] = in[index - eq2];
			}
		} else {
			if (j < N/2) {   //Q2
				out[index] = in[index + eq2];
			} else {		 //Q4
				out[index] = in[index - eq1];
			}
		}
	}

}

void fftshift(cufftComplex *out, cufftComplex *in, int N, int NZ) {

	dim3 dimGrid (int((N-0.5)/BSZ) + 1, int((N-0.5)/BSZ) + 1, NZ);
	dim3 dimBlock (BSZ, BSZ, 1);
    fftshift_kernel<<<dimGrid, dimBlock>>>(out, in, N, NZ);
	
}