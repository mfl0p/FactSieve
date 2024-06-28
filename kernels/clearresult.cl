/*

	clearresult.cl - Bryan Little 6/2024
	
	clear prime counters and checksum

*/


__kernel void clearresult(__global uint *g_primecount, __global ulong *g_sum, const uint numgroups){

	const uint gid = get_global_id(0);

	if(gid < numgroups){
		g_sum[gid] = 0;	// index 0 is total primecount between checkpoints.  index 1 to 'numgroups' are for each workgroup's checksum
	}

	if(gid == 0){
		g_primecount[1] = 0;	// keep track of largest kernel prime count
		g_primecount[2] = 0;	// # of factors found
		g_primecount[3] = 0;	// flag set for power table error
		g_primecount[4] = 0;	// flag set for getsegprimes local memory overflow
		g_primecount[5] = 0;	// flag set for gpu validation failure
	}

}

