#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"

#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

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

int main(int argc, char **argv) {

	char out_window[] = "Result";
	float *dev_mat, *dev_out;
	float out[512*512], *in;
	int N = 512;

	cv::Mat img = cv::Mat::zeros(512,512, CV_32FC1);
	cv::Point center = cv::Point(256,256);
	cv::circle( img, 
			    center, 
		    	60,
            	cv::Scalar( 255 ),
         		-1,
         		8 );

	if (img.isContinuous())
    	in = (float *) img.data;

    CUDA_CALL(cudaMalloc((void **) &dev_mat, 512*512*sizeof(float)));
    CUDA_CALL(cudaMalloc((void **) &dev_out, 512*512*sizeof(float)));
    CUDA_CALL(cudaMemcpy(dev_mat, in, 512*512*sizeof(float),cudaMemcpyHostToDevice));

    dim3 dimGrid (int((N-0.5)/64) + 1, int((N-0.5)/64) + 1);
	dim3 dimBlock (64, 64);
    fftshift<<<dimGrid, dimBlock>>>(dev_out, dev_mat, 512);

    CUDA_CALL(cudaMemcpy(out, dev_out, 512*512*sizeof(float), cudaMemcpyDeviceToHost));

    cv::Mat shifted = cv::Mat(512, 512, CV_32FC1, &out);
    cv::imshow(out_window, shifted);
    cv::waitKey(0);
    cv::destroyAllWindows();

    cudaFree( dev_out );
    cudaFree( dev_mat );

    return 0;
}
