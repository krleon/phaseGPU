#include <stdio.h>
#include "cuda_funcs.h"

__global__ void getSubHarmonic_kernel(cufftComplex *out, float *fx, float *fy, float *SH_PSD, float *seeds, float delta, int N, int NZ) {

	//Seeds is a group of random numbers 27*2 x NZ 
	//SH_PSD is a 27 x 1 matrix with the PSD values for the respective fx and fy values
	//fx is 27 x 1 vector
	//fy is 27 x 1 vector

	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int k = threadIdx.z + blockIdx.z*blockDim.z;

	int index = k*N*N + j*N + i;

	if (i < N && j < N && k < NZ) {

		float x = (i - NX/2.)*delta;
		float y = (j - NY/2.)*delta;
		
		out[index].x = 0.0;
		out[index].y = 0.0;
		float cnx, cny;
		for (int ii = 0; ii < 3; ii++) {
			float del_f = 1./powf(3.,ii+1);
			for (int jj = 0; jj < 9; jj++) {
				int idx = 2*(k*27 + ii*9 + jj);
				cnx = seeds[idx]*sqrt(SH_PSD[ii*9 + jj])*del_f;
				cny = seeds[idx+1]*sqrt(SH_PSD[ii*9 + jj])*del_f;
				//need to define x, y and exp(ix)
				float theta = 2*PI*(fx[ii*9+jj]*x + fy[ii*9+jj]*y);
				out[index].x = out[index].x + cnx * cos(theta) - cny * sin(theta);
				out[index].y = out[index].y + cnx * sin(theta) + cny * cos(theta);
			}

		}

	}

}

void getSubHarmonic(cufftComplex *out, float *d_fx, float *d_fy, float *d_SH_PSD, float *seeds, float delta, int N, int NZ) {

	dim3 dimGrid (int((N-0.5)/BSZ) + 1, int((N-0.5)/BSZ) + 1, NZ);
	dim3 dimBlock (BSZ, BSZ, 1);
	getSubHarmonic_kernel<<<dimGrid, dimBlock>>>(d_fx, d_fy, d_SH_PSD, out);
}
