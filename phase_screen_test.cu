#include <cufft.h>
#include <curand.h>
#include <math.h>
#include <stdio.h>
#include "cuda_funcs.h"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#define PI 3.14159265358979323846

//512 x 512 x 1000 in 32-bit floats => 1.05GB => 2.1GB "complex"
// my device has 1GB of memory, roughly 512 x 512 x 1000 x 32 bit
// will try 250 at a time first
// 1024 threads per block (warp size is 32)
int main() {

	float D = 2.0;
	float r0 = 0.1;
	float L0 = 100;
	//float l0 = 0.01;

	dataSize size;   //Might want to set up constructor and volume elem
	size.x = 512;
	size.y = 512;
	size.z = 1;

	char out_window[] = "Result";
	float out[size.x*size.y*size.z];

	float delta = D/size.x;

	curandGenerator_t gen;

	cufftHandle plan;
	cufftComplex *data, *shift_out;
	float *real_data;
	float *imag_data;
	CUDA_CALL(cudaMalloc((void**)&data, sizeof(cufftComplex)*size.x*size.y*size.z));
	CUDA_CALL(cudaMalloc((void**)&real_data, sizeof(float)*size.x*size.y*size.z));
	CUDA_CALL(cudaMalloc((void**)&imag_data, sizeof(float)*size.x*size.y*size.z));

	// Initialize the gpu "arrays" with randn numbers
	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	/* Set seed */
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

	/* Generate real and imag normally random distributed numbers */
	CURAND_CALL(curandGenerateNormal(gen, real_data, size.x*size.y*size.z, 0.0, 1.0));
	CURAND_CALL(curandGenerateNormal(gen, imag_data, size.x*size.y*size.z, 0.0, 1.0));

	// Need to make complex numbers here
	makeComplexPSD(real_data, imag_data, data, r0, delta, L0, size);
	CUDA_CALL(cudaFree(real_data));
	CUDA_CALL(cudaFree(imag_data));
	CUDA_CALL(cudaMalloc((void**)&shift_out, sizeof(cufftComplex)*size.x*size.y*size.z));
	fftshift(shift_out, data, size.x, size.z);

	//2^a x 3^b is most efficient size
	/* Create a 2D plan */
	//Need advanced data layout to do batch in 2D using cufftPlanMany
	cufftPlan2d(&plan, size.x, size.y, CUFFT_C2C);
	cufftExecC2C(plan, shift_out, shift_out, CUFFT_INVERSE);
	
	fftshift(data, shift_out, size.x, size.z);
	CUDA_CALL(cudaFree(shift_out));
	CUDA_CALL(cudaMalloc((void**)&real_data, sizeof(float)*size.x*size.y*size.z));
	getComplexAbs(real_data, data, size);

	CUDA_CALL(cudaMemcpy(out, real_data, size.x*size.y*size.z*sizeof(float), cudaMemcpyDeviceToHost));

	cv::Mat screen = cv::Mat(size.x*size.z, size.y, CV_32FC1, &out);
    cv::imshow(out_window, screen);
    cv::waitKey(0);

    double min, max;
	cv::minMaxLoc(screen, &min, &max);
	printf("Min: %f\nMax: %f \n", min, max);

	cv::destroyAllWindows();

	/* Destroy the CUFFT plan */
	cufftDestroy(plan);
	cudaFree(data);
	cudaFree(real_data);

}