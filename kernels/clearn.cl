/*

	clearn.cl - Bryan Little 6/2024

	Clears prime counter.

*/


__kernel void clearn(__global uint *primecount){

	const uint i = get_global_id(0);

	if(i==0){
		primecount[0]=0;
	}


}



