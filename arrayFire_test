#include <arrayfire.h>
#include <math.h>
#include <stdio.h>

af::array getPSDArray(float D, float r0, float L0, float l0, int N) {

	af::array b = af::array(af::seq(-N/2,N/2-1))/D;
	af::array x = af::tile(b,1,N);
	af::array y = af::tile(b.T(),N,1);

	af::array f = af::sqrt(af::pow(x,2) + af::pow(y,2));

	float fm = 5.92/l0/(2.0*af::Pi);
	float f0 = 1.0/L0;

	af::array PSD_phi = 0.023*powf(r0,-5./3.)*af::exp(-af::pow((f/fm),2))/af::pow(af::pow(f,2)+powf(f0,2),11./6.);

	return PSD_phi;

}

int main(void) {

	int N = 256;
	
	float D = 2.0;
	float r0 = 0.1;
	float L0 = 100;
	float l0 = 0.01;

	af::array PSD_phi = getPSDArray(D, r0, L0, l0, N)

	af::array a = af::randn(N, N)*af::sqrt(PSD_phi)/D;
	af::array b = af::randn(N, N)*af::sqrt(PSD_phi)/D;

	af::array c = af::complex(a,b);

	//shift
	af::shift(c, c.dims(0)/2, c.dims(1)/2);
	//fft
	fft2InPlace(c);
	//shift
	af::shift(c, c.dims(0)/2, c.dims(1)/2);
	//get real
	af::array out = af::real)(c);

	//show
	
	//print min/max
	printf("Minimum Val: %f\n", af::min(out));
	printf("Maximum Val: %f\n", af::max(out));

	af::Window wnd(N, N, "Phase Screen");
    wnd.image(out);

    return 0;
}