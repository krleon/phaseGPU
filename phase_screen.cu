#include <cuda.h>
#include <cufft.h>
#include <curand.h>
#include <math.h>

#define NX 512
#define NY 512
#define BATCH 250
#define BSZ 64
#define PI 3.14159265358979323846

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

__global__ void fftshift(float *out, float* in, N) {

	int i = threadIdx.x + blockIdx.x*blockDim.x
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int k = threadIdx.z + blockIdx.z*blockDim.z;
	int index = k*N*N + j*N + i;

	if (i < N && j < N && k < BATCH) {
		int eq1 = (N*N + N)/2;
		int eq2 = (N*N - N)/2;

		if (i < N/2)  {
			if (j < N/2) {   //Q1
				out[index] = in[index + eq1];
			} else {		 //Q2
				out[index] = in[index + eq2];
			}
		} else {
			if (j < N/2) {   //Q3
				out[index] = in[index - eq2];
			} else {		 //Q4
				out[index] = in[index - eq1];
			}
		}
	}

}

/* Need to try two different methods: 1) calculating PSD ahead of time and copying it to GPU, or 
   calculating it each time on the GPU
*/
__global__ void makeComplex(float *real, float* imag, cufftComplex *fc, float r0, float delta, float L0) {
	
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	int k = threadIdx.z + blockIdx.z*blockDim.z;
	int index = k*NX*NY+j*NX+i;

	if (i < NX && j < NY && k < BATCH) {
		float PSD_phi;
		float fx = (i - NX/2.)/(NX*delta);
		float fy = (j - NY/2.)/(NY*delta);
		float f = sqrt(powf(fx,2) + powf(fy,2))
		float fm = 5.92/10/(2*PI);
		float f0 = 1/L0
		PSD_phi = 0.023*powf(r0,-5./3.)*expf(-powf((f/fm),2))/(powf(f,2) + powf(f0,2));
		PSD_phi = pow(PSD_phi,11./6.);

		fc[index].x = real[index]*sqrt(PSD_phi)/(NX*delta);
		fc[index].y = imag[index]*sqrt(PSD_phi)/(NX*delta);
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

	float delta = D/NX;

	curandGenerator_t gen;

	cufftHandle plan;
	cufftComplex *data;
	float *real_data;
	float *imag_data;
	CUDA_CALL(cudaMalloc((void**)&data, sizeof(cufftComplex)*NX*NY*BATCH));
	CUDA_CALL(cudaMalloc((void**)&real_data, sizeof(float)*NX*NY*BATCH));
	CUDA_CALL(cudaMalloc((void**)&imag_data, sizeof(float)*NX*NY*BATCH));

	// Initialize the gpu "arrays" with randn numbers
	CURAND_CALL(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT));
	/* Set seed */
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen, 1234ULL));

	/* Generate real and imag normally random distributed numbers */
	CURAND_CALL(curandGenerateNormal(gen, real_data, NX*NY*BATCH));
	CURAND_CALL(curandGenerateNormal(gen, imag_data, NX*NY*BATCH));

	dim3 dimGrid (int((NX-0.5)/BSZ) + 1, int((NY-0.5)/BSZ) + 1, int((BATCH-0.5)/BSZ) + 1);
	dim3 dimBlock (BSZ, BSZ, BSZ);
	// Need to make complex numbers here
	makeComplex<<<dimGrid, dimBlock>>>(real_data, imag_data, data, NX, NY, NZ, r0, delta, L0);

	//2^a x 3^b is most efficient size
	/* Create a 2D plan */
	cufftPlan2d(&plan, NX, NY, CUFFT_C2C, BATCH);

	//IFFTSHIFT NEEDED HERE
	cufftExecC2C(plan, data, data, CUFFT_INVERSE);
	//IFFTSHIFT AGAIN HERE

	/* Destroy the CUFFT plan */
	cufftDestroy(plan);
	cudaFree(data);

}