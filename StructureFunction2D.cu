#include <stdio.h>
#include "cuda_funcs.h"

/* 2D Structure Function
	
	N = size(ph, 1);
	ph = ph.*mask;
	P = ft2(ph, delta);
	S = ft2(ph.^2, delta);
	W = ft2(mask, delta);
	delta_f = 1/(N*delta)
	w2 = ift2(W.*conj(W), delta_f);

	D = 2 * ift2(real(S.*conj(W)) - abs(P).^2, delta_f) ./ w2 .* mask;
*/

void get2DStructureFn(float *ph, float *mask, float delta, dataSize size) {

	//Should I use thrust to do the multiplication of ph and mask and others?
	//Also need to think about allocation of memory.  Can probably use the same plan for
	//all the fft's since they are the same size.

	//Need space for ph, P, S, W, w2 and D (could probably get away with 5 3D matrices instead of 6)
}