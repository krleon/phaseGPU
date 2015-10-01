#include <arrayfire.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

af::array getPSDArray(float D, float r0, float L0, float l0, int N) {

	af::array b = af::array(af::seq(-N/2,N/2-1))/D;
	af::array y = af::tile(b,1,N);
	af::array x = af::tile(b.T(),N,1);

	af::array f = af::sqrt(af::pow(x,2) + af::pow(y,2));

	float fm = 5.92/l0/(2.0*af::Pi);
	float f0 = 1.0/L0;

	af::array PSD_phi = 0.023*powf(r0,-5./3.)*af::exp(-af::pow((f/fm),2))/af::pow(af::pow(f,2)+powf(f0,2),11./6.);
	PSD_phi(N/2,N/2) = 0.0;

	return PSD_phi;

}

af::array ift2(af::array in, float delta_f) {

	int N = in.dims(0);

	af::array temp = af::shift(in, (in.dims(0)+1)/2, (in.dims(1)+1)/2);
	af::ifft2InPlace(temp);

	return af::shift(temp, (temp.dims(0)+1)/2, (temp.dims(1)+1)/2)*powf(N*delta_f,2);

}

af::array ft2(af::array in, float delta) {

	af::array temp = af::shift(in, in.dims(0)/2, in.dims(1)/2);
	af::fft2InPlace(temp);
	return af::shift(temp, temp.dims(0)/2, temp.dims(1)/2)*powf(delta,2);

}

af::array getRandom(int NX, int NY, int NZ) {
	af::array a = af::randn(NX, NY, NZ);
	af::array b = af::randn(NX, NY, NZ);

	return af::complex(a,b);
}

af::array calculateStructureFn(af::array ph, af::array mask, float delta) {

	af::array P = ft2(af::complex(ph*mask), delta);
	af::array S = ft2(af::complex(af::pow(ph*mask,2)), delta);
	af::array W = ft2(af::complex(mask), delta);

	float delta_f = 1/(ph.dims(0)*delta);
	af::array w2 = ift2(W*af::conjg(W), delta_f);

	return (mask*2*ift2(af::complex(af::real(S*conjg(W))) - af::pow(af::abs(P),2), delta_f) / w2);

}

int main(void) {

	int N = 512;
	int num_screens = 40;

	float D = 6.0;
	float r0 = 0.2;
	float L0 = 40;
	float l0 = 0.01;

	af::setDevice(0);
	af::info();

	af::array PSD_phi = af::tile(getPSDArray(D, r0, L0, l0, N),1,1,num_screens);

	af::setSeed(time(NULL));
	af::array c = getRandom(N, N, num_screens)*af::sqrt(PSD_phi)/D;

	af::array out(N,N,num_screens);
	af::array avgstrfn = af::constant(0.0,N,N);

	gfor (af::seq i, num_screens) {
		out(af::span,af::span,i) = af::real(ift2(c(af::span,af::span,i),1));
	}

	af::array b = af::array(af::seq(-N/2,N/2-1))*D/N;
	af::array y = af::tile(b,1,N);
	af::array x = af::tile(b.T(),N,1);
        af::array f = af::sqrt(af::pow(x,2) + af::pow(y,2));

	af::array mask = af::constant(0.0,N,N);
	mask(af::where(f < .85*(D/2.))) = 1.0;
	//gfor (af::seq i, num_screens)
	//	fullstrfn(af::span,af::span,i) = calculateStructureFn(out(af::span,af::span,i),mask,D/N);

	for (int ii = 0; ii < num_screens; ii++) {
		avgstrfn = avgstrfn + calculateStructureFn(out(af::span, af::span, ii), mask, D/N);
	}

	avgstrfn = avgstrfn/num_screens;

	//print min/max
	float minVal = af::min<float>(avgstrfn);
	float maxVal = af::max<float>(avgstrfn);
	printf("Minimum Val: %f\n", minVal);
	printf("Maximum Val: %f\n", maxVal);

	af::Window wnd(N, N, "Phase Screen");
	af::array out3 = (af::abs(avgstrfn)-minVal)/(maxVal-minVal);

	af::array yaxes = af::abs(avgstrfn(255,af::seq(256,511)));
	af::array xaxes = af::seq(0,N/2-1);;
	xaxes = xaxes*D/(N*r0);

	while(!wnd.close()) {
		//af::array y = avgstrfn(255,af::seq(256,511));
		wnd.plot(xaxes,yaxes.T());
		wnd.show();
	}
    return 0;
}
