#include <arrayfire.h>
#include <math.h>
#include <stdio.h>

af::array getPSDArray(float D, float r0, float L0, float l0, int N) {

	af::array b = af::array(af::seq(-N/2,N/2-1))/D;
	af::array y = af::tile(b,1,N);
	af::array x = af::tile(b.T(),N,1);

	//af::print("x",x(af::seq(0,9)),0);
	//af::print("y",y(0,af::seq(0,9)));

	af::array f = af::sqrt(af::pow(x,2) + af::pow(y,2));

	float fm = 5.92/l0/(2.0*af::Pi);
	float f0 = 1.0/L0;

	af::array PSD_phi = 0.023*powf(r0,-5./3.)*af::exp(-af::pow((f/fm),2))/af::pow(af::pow(f,2)+powf(f0,2),11./6.);

	return PSD_phi;

}

int main(void) {

	int N = 4096;
	
	float D = 2.0;
	float r0 = 0.1;
	float L0 = 100;
	float l0 = 0.01;

	af::setDevice(0);
	af::info();

	af::array PSD_phi = getPSDArray(D, r0, L0, l0, N);
	PSD_phi(N/2,N/2) = 0.0;

	af::array out;

	gfor (af::seq i, 10) {
	af::array a = af::randn(N, N)*af::sqrt(PSD_phi)/D;
	af::array b = af::randn(N, N)*af::sqrt(PSD_phi)/D;
	//af::array a = af::randn(2,2);
	//af::array b = af::randn(2,2);
	
	
	af::array c = af::complex(a,b);

	//shift
	af::array shift_out = af::shift(c, (c.dims(0)+1)/2, (c.dims(1)+1)/2);
	//ifft
	ifft2InPlace(shift_out);
	//shift
	c = af::shift(shift_out, (shift_out.dims(0)+1)/2, (shift_out.dims(1)+1)/2);
	//get real
	
	out = af::real(c)*N*N;
}
	//show
	
	//print min/max
	float minVal = af::min<float>(out);
	float maxVal = af::max<float>(out);
	printf("Minimum Val: %f\n", minVal);
	printf("Maximum Val: %f\n", maxVal);

	af::Window wnd(N, N, "Phase Screen");
	//wnd.setColorMap(AF_COLORMAP_SPECTRUM);
   	//af::print("out",(out-minVal)/(maxVal-minVal));
	af::array out2 = (out-minVal)/(maxVal-minVal);

	af::print("out",out(1,af::seq(1,10)),10);
	while(!wnd.close()) {
		wnd.image(out2);
		wnd.show();
	}
    return 0;
}
