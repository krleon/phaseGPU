#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include "cuda_funcs.h"
#include <cuda.h>
#include <cuda_runtime_api.h>

int main(int argc, char **argv) {

	char in_window[] = "Initial";
	char out_window[] = "Result";
	float *dev_mat, *dev_out;
	int N = 512;
	int NZ = 2;
	float out[N*N*NZ], *in;
	
	cv::Mat img = cv::Mat::zeros(N*NZ, N, CV_32FC1);
	cv::Point center = cv::Point(N/2,N/2);
	cv::Point center2 = cv::Point(N/2, N + N/2);
	cv::circle( img, 
			    center, 
		    	N/2,
            	cv::Scalar( 255 ),
         		-1,
         		8 );

	cv::circle( img,
				center2,
				N/2,
				cv::Scalar( 128 ),
				-1,
				8 );

	if (img.isContinuous())
    	in = (float *) img.data;

    CUDA_CALL(cudaMalloc((void **) &dev_mat, N*N*NZ*sizeof(float)));
    CUDA_CALL(cudaMalloc((void **) &dev_out, N*N*NZ*sizeof(float)));
    CUDA_CALL(cudaMemcpy(dev_mat, in, N*N*NZ*sizeof(float),cudaMemcpyHostToDevice));

    fftshift(dev_out, dev_mat, N, NZ);

    CUDA_CALL(cudaMemcpy(out, dev_out, N*N*NZ*sizeof(float), cudaMemcpyDeviceToHost));

    cv::Mat shifted = cv::Mat(N, N*NZ, CV_32FC1, &out);
    cv::imshow(in_window, img);
    cv::imshow(out_window, shifted);
    cv::waitKey(0);
    cv::destroyAllWindows();

    cudaFree( dev_out );
    cudaFree( dev_mat );

    return 0;
}
