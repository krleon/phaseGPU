#include <arrayfire.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

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
	PSD_phi(N/2,N/2) = 0.0;

	return PSD_phi;

}

af::array ift2(af::array in) {

	af::array temp = af::shift(in, (in.dims(0)+1)/2, (in.dims(1)+1)/2);
	af::ifft2InPlace(temp);

	return af::shift(temp, (temp.dims(0)+1)/2, (temp.dims(1)+1)/2);

}

af::array getRandom(int NX, int NY, int NZ) {
	af::array a = af::randn(NX, NY, NZ);
	af::array b = af::randn(NX, NY, NZ);

	return af::complex(a,b);
}

int main(void) {

	int N = 512;
	int num_screens = 40;
	
	float D = 2.0;
	float r0 = 0.1;
	float L0 = 40;
	float l0 = 0.01;

	af::setDevice(0);
	af::info();

	af::array PSD_phi = af::tile(getPSDArray(D, r0, L0, l0, N),1,1,num_screens);

	af::setSeed(time(NULL));
	af::array c = getRandom(N, N, num_screens)*af::sqrt(PSD_phi)/D;

	af::array out(N,N,num_screens);

	gfor (af::seq i, num_screens) {
		out(af::span,af::span,i) = af::real(ift2(c(af::span,af::span,i)))*N*N;	
	}
	
	//print min/max
	float minVal = af::min<float>(out(af::span,af::span,0));
	float maxVal = af::max<float>(out(af::span,af::span,0));
	printf("Minimum Val: %f\n", minVal);
	printf("Maximum Val: %f\n", maxVal);

	af::Window wnd(N, N, "Phase Screen");
	//wnd.setColorMap(AF_COLORMAP_SPECTRUM);
   	//af::print("out",(out-minVal)/(maxVal-minVal));
	af::array out2 = (out(af::span,af::span,0)-minVal)/(maxVal-minVal);

	while(!wnd.close()) {
		wnd.image(out2);
		wnd.show();
	}
    return 0;
}
