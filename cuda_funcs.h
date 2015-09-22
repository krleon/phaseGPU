#include <cufft.h>

#define BSZ 32

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

typedef struct dataSize {
	int x;
	int y;
	int z;
};

void fftshift(float*, float*, int, int);
void makeComplexPSD(float*, float*, cufftComplex*, float, float, float, dataSize);
