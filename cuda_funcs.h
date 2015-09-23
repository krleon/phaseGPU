#include <cufft.h>
#include <curand.h>

#define BSZ 32
#define PI 3.14159265358979323846

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

typedef struct {
	int x;
	int y;
	int z;
} dataSize;

void fftshift(cufftComplex*, cufftComplex*, int, int);
void makeComplexPSD(float*, float*, cufftComplex*, float, float, float, float, dataSize);
void getComplexAbs(float*, cufftComplex*, dataSize);
