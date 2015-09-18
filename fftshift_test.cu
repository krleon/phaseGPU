#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <stdio.h>
#include "fftshift.h"
#include "cuda.h"

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

    fftshift(dev_out, dev_mat, 512);

    CUDA_CALL(cudaMemcpy(out, dev_out, 512*512*sizeof(float), cudaMemcpyDeviceToHost));

    cv::Mat shifted = cv::Mat(512, 512, CV_32FC1, &out);
    cv::imshow(out_window, shifted);
    cv::waitKey(0);
    cv::destroyAllWindows();

    cudaFree( dev_out );
    cudaFree( dev_mat );

    return 0;
}
