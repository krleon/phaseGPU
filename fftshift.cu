#include <stdio.h>
#include "fftshift.h"

__global__ void fftshift(float *out, float* in, int N) {

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int index = j*N + i;

	if (i < N && j < N) {
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

void fftshift(float *out, float *in, int N) {

	dim3 dimGrid (int((N-0.5)/BSZ) + 1, int((N-0.5)/BSZ) + 1);
	dim3 dimBlock (BSZ, BSZ);
    fftshift<<<dimGrid, dimBlock>>>(out, in, N);
	
}