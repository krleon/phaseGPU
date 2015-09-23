#include <stdio.h>
#include "cuda_funcs.h"

/* Need to try two different methods: 1) calculating PSD ahead of time and copying it to GPU, or 
   calculating it each time on the GPU
*/
__global__ void getComplexAbs(float *out, cufftComplex *in, 
							int NX, int NY, int NZ) {
	
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int k = threadIdx.z + blockIdx.z*blockDim.z;
	int index = k*NX*NY+j*NX+i;

	if (i < NX && j < NY && k < NZ) {
		
		out[index] = sqrt(powf(in[index].x,2) + powf(in[index].y,2));
	}
}

void getComplexAbs(float *out, cufftComplex *in, dataSize size) {

	dim3 dimGrid (int((size.x-0.5)/BSZ) + 1, int((size.y-0.5)/BSZ) + 1, size.z);
	dim3 dimBlock (BSZ, BSZ, 1);
	// Need to make complex numbers here
	getComplexAbs<<<dimGrid, dimBlock>>>(out, in, size.x, size.y, size.z);

}