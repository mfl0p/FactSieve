/*

	clearresult kernel
	
	clear prime counters and checksum

*/


__kernel void clearresult(__global uint *g_primecount, __global ulong *g_sum, const uint numgroups){

	const uint i = get_global_id(0);

	if(i == 0){
		g_primecount[1] = 0;	// keep track of largest kernel prime count
		g_primecount[2] = 0;	// # of factors found
		g_primecount[3] = 0;	// flag set for power table error
		g_primecount[4] = 0;	// flag set for getsegprimes local memory overflow
	}

	if(i < numgroups){
		g_sum[i] = 0;	// index 0 is total primecount between checkpoints.  index 1 to 'numgroups' are for each workgroup's checksum
	}

}

