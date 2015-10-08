#include <cufft.h>
#include <curand.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "arrayfire.h"
#include "cuda_funcs.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

void getSH_PSD(float D, float l0, float L0, float r0, float* PSD_phi, float* fx, float *fy) {

	float y_vals[] = {-1.0,-1.0,-1.0,0.0,0.0,0.0,1.0,1.0,1.0};
	float x_vals[] = {-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0};
	float f[9];

	float del_f;
	float fm = 5.92/l0/(2*PI);
	float f0 = 1/L0;

	for (int ii = 0; ii < 3; ii++) {
		del_f = 1./(powf(3.0,ii+1)*D);
		for (int jj = 0; jj < 9; jj++) {
			fx[ii*9+jj] = x_vals[jj]*del_f;
			fy[ii*9+jj] = y_vals[jj]*del_f;
			f[jj] = sqrt(powf(fx[ii*9+jj],2) + powf(fy[ii*9+jj],2));
			if (jj == 4) 
				PSD_phi[ii*9+jj] = 0;
			else
				PSD_phi[ii*9+jj] = 0.023*powf(r0,-5./3.)*exp(-powf(f[jj]/fm,2)/powf(powf(f[jj],2)+powf(f0,2),11./6.));
		}
	}
}

//512 x 512 x 1000 in 32-bit floats => 1.05GB => 2.1GB "complex"
// my device has 1GB of memory, roughly 512 x 512 x 1000 x 32 bit
// will try 250 at a time first
// 1024 threads per block (warp size is 32)
int main() {

	float D = 2.0;
	float r0 = 0.1;
	float L0 = 100;
	float l0 = 0.01;

	dataSize size;   //Might want to set up constructor and volume elem
	size.x = 256;
	size.y = 256;
	size.z = 2;

	char out_window[] = "Result";

	float delta = D/size.x;

	cufftComplex *d_complexA, *d_complexB;
	float *d_floatA, *d_floatB, *d_seed_data;

	CUDA_CALL(cudaMalloc((void**)&d_complexA, sizeof(cufftComplex)*size.x*size.y*size.z));
	CUDA_CALL(cudaMalloc((void**)&d_floatA, sizeof(float)*size.x*size.y*size.z));
	CUDA_CALL(cudaMalloc((void**)&d_floatB, sizeof(float)*size.x*size.y*size.z));
	CUDA_CALL(cudaMalloc((void**)&d_seed_data, sizeof(float)*size.z*27*2));

	// Initialize the gpu "arrays" with randn numbers
	curandGenerator_t gen;
	srand(time(NULL));
	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, rand()));

	/* Generate real and imag normally random distributed numbers */
	CURAND_CALL(curandGenerateNormal(gen, d_floatA, size.x*size.y*size.z, 0.0, 1.0));
	CURAND_CALL(curandGenerateNormal(gen, d_floatB, size.x*size.y*size.z, 0.0, 1.0));
	CURAND_CALL(curandGenerateNormal(gen, d_seed_data, size.z*27*2, 0.0, 1.0));
	cudaThreadSynchronize();

	// Need to make complex numbers here
	makeComplexPSD(d_floatA, d_floatB, d_complexA, r0, delta, L0, l0, size);

	// Free up floatA and float B to make room for complexB
	CUDA_CALL(cudaFree(d_floatA));
	CUDA_CALL(cudaFree(d_floatB));

	// Create complexB now that there is room
	CUDA_CALL(cudaMalloc((void**)&d_complexB, sizeof(cufftComplex)*size.x*size.y*size.z));
	
	// Set up fft2d with batch in 3rd dimension
	cufftHandle plan;
	int n[2] = {size.y, size.x};
	int inembed[2] = {size.x*size.y, size.x};
	cufftPlanMany(&plan, 2, n, inembed, 1, inembed[0], inembed, 1, inembed[0], CUFFT_C2C, size.z);
	
	// Do ift2 function
	fftshift(d_complexB, d_complexA, size.x, size.z);
	cufftExecC2C(plan, d_complexA, d_complexB, CUFFT_INVERSE);
	fftshift(d_complexB, d_complexA, size.x, size.z);

	// Save high frequency screen to host and free memory on device
	//cufftComplex phz_hi[size.x*size.y*size.z];
	//CUDA_CALL(cudaMemcpy(phz_hi, d_complexB, sizeof(cufftComplex)*size.x*size.y*size.z, cudaMemcpyDeviceToHost));
	CUDA_CALL(cudaFree(d_complexA));
	CUDA_CALL(cudaMalloc((void**)&d_floatA, sizeof(float)*size.x*size.y*size.z));
	CUDA_CALL(cudaMalloc((void**)&d_floatB, sizeof(float)*size.x*size.y*size.z));
	getComplexReal(d_floatA, d_complexB, size, 1.0);
	//CUDA_CALL(cudaFree(d_complexB));	

	//CUDA_CALL(cudaMalloc((void**)&d_floatA, sizeof(float)*size.x*size.y*size.z));
	
	/********Need to get phz Hi back in to combine and remember to get real*************/
	//getComplexReal(real_data, data, size, 1);

	float SH_PSD[27], fx[27], fy[27];
	getSH_PSD(D, l0, L0, r0, SH_PSD, fx, fy);

	float *d_SH_PSD, *d_fx, *d_fy;
	CUDA_CALL(cudaMalloc((void**)&d_fx, sizeof(float)*27));
	CUDA_CALL(cudaMalloc((void**)&d_fy, sizeof(float)*27));
	CUDA_CALL(cudaMalloc((void**)&d_SH_PSD, sizeof(float)*27));
	CUDA_CALL(cudaMemcpy(d_fx, fx, 27*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_fy, fy, 27*sizeof(float), cudaMemcpyHostToDevice));
	CUDA_CALL(cudaMemcpy(d_SH_PSD, SH_PSD, 27*sizeof(float), cudaMemcpyHostToDevice));

	getSubHarmonic(d_complexB, d_fx, d_fy, d_SH_PSD, d_seed_data, delta, size.x, size.z);
	getComplexReal(d_floatB, d_complexB, size, 1);
	CUDA_CALL(cudaFree(d_complexB));

        
	af::array phz_lo = subtract2DMean(d_floatB, size);
	
        af::array phz(size.x, size.y, size.z, d_floatA, afDevice);

       
	
	printf("Dim 1: %lu, Dim 2: %lu, Dim 3: %lu \n", phz.dims(0), phz.dims(1), phz.dims(2));
	printf("Dim 1: %lu, Dim 2: %lu, Dim 3: %lu \n", phz_lo.dims(0), phz_lo.dims(1), phz_lo.dims(2));
	phz = phz + phz_lo;
	
	float *d_out = phz.device<float>();
        float out[size.x*size.y*size.z];
	CUDA_CALL(cudaMemcpy(out, d_out, sizeof(float)*size.x*size.y*size.z, cudaMemcpyDeviceToHost));
        phz.unlock();
	phz_lo.unlock();

	cv::Mat screen = cv::Mat(size.x, size.y, CV_32FC1, &out);
    	//cv::namedWindow(out_window);
	cv::imshow(out_window, screen);
    cv::waitKey(0);

    double min, max;
	cv::minMaxLoc(screen, &min, &max);
	printf("Min: %f\nMax: %f \n", min, max);

	cv::destroyAllWindows();

	delete[] out;
	/* Destroy the CUFFT plan */
	cufftDestroy(plan);
	CUDA_CALL(cudaFree(d_floatA));
	CUDA_CALL(cudaFree(d_floatB));
	CUDA_CALL(cudaFree(d_seed_data));
	CUDA_CALL(cudaFree(d_fx));
	CUDA_CALL(cudaFree(d_fy));
	CUDA_CALL(cudaFree(d_SH_PSD));
}
