#include <stdio.h>
#include "cuda_funcs.h"

/* Need to try two different methods: 1) calculating PSD ahead of time and copying it to GPU, or 
   calculating it each time on the GPU
*/
__global__ void makeComplexPSD(float *real, float* imag, cufftComplex *fc, 
							int NX, int NY, int NZ, float r0, float delta, float L0, float l0) {
	
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int k = threadIdx.z + blockIdx.z*blockDim.z;
	int index = k*NX*NY+j*NX+i;

	if (i < NX && j < NY && k < NZ) {
		float PSD_phi;
		float fx = (i - NX/2.)/(NX*delta);
		float fy = (j - NY/2.)/(NY*delta);
		float f = sqrt(powf(fx,2) + powf(fy,2));
		float fm = 5.92/l0/(2.0*PI);
		float f0 = 1.0/L0;
		PSD_phi = 0.023*powf(r0,-5./3.)*expf(-powf((f/fm),2))/powf(powf(f,2) + powf(f0,2),11./6.);

		if (i == NX/2 && j == NX/2) {
			fc[index].x = 0;
			fc[index].y = 0;
		} else {
			fc[index].x = real[index]*sqrt(PSD_phi)/(NX*delta);
			fc[index].y = imag[index]*sqrt(PSD_phi)/(NX*delta);
		}
	}
}

void makeComplexPSD(float *real, float *imag, cufftComplex *fc, 
					float r0, float delta, float L0, float l0, dataSize size) {
	dim3 dimGrid (int((size.x-0.5)/BSZ) + 1, int((size.y-0.5)/BSZ) + 1, size.z);
	dim3 dimBlock (BSZ, BSZ, 1);
	// Need to make complex numbers here
	makeComplexPSD<<<dimGrid, dimBlock>>>(real, imag, fc, size.x, size.y, size.z, r0, delta, L0, l0);

}