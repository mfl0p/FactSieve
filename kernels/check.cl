/*

	check.cl - Bryan Little 6/2024

	Generate a checksum for boinc quorum using residue of the final factorial mod P for each P.

	Using only .s6 for checksum because it depends on all other .s# values to be correct.
	Therefore, a missed prime or any calculation error will change the final checksum value.

*/


__kernel __attribute__ ((reqd_work_group_size(256, 1, 1))) void check(	__global ulong8 * g_prime,
									__global uint * g_primecount,
									__global ulong * g_sum,
									const uint numgroups ) {

	const uint gid = get_global_id(0);
	const uint lid = get_local_id(0);
	const uint pcnt = g_primecount[0];
	__local ulong sum[256];

	if(gid < pcnt){
		// .s0=p, .s1=q, .s2=r2, .s3=one, .s4=two, .s5=nmo, .s6=residue of final factorial, .s7= montgomery form of last n
		sum[lid] = g_prime[gid].s6;
	}
	else{
		sum[lid] = 0;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// local memory reduction.  group size is forced to 256
//	for(uint s = get_local_size(0) / 2; s > 0; s >>= 1){
	for(uint s = 128; s > 0; s >>= 1){
		if(lid < s){
			sum[lid] += sum[lid + s];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(lid == 0){
		uint index = get_group_id(0) + 1;

		if(index < numgroups){	
			g_sum[index] += sum[0];
		}
	}

	if(gid == 0){

		// add primecount to total primecount
		g_sum[0] += pcnt;

		// store largest kernel prime count
		if( pcnt > g_primecount[1] ){
			g_primecount[1] = pcnt;
		}
	}

}





