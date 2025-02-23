/*

	check.cl - Bryan Little 6/2024, montgomery arithmetic by Yves Gallot

	Generate a checksum for boinc quorum using the residue of the final factorial mod P (.s6) added to the
	last N in montgomery form (.s7) for each P.
	
	A validation test is performed by converting the final N value out of montgomery form and comparing
	it to nmax.  Result should be equal after iterating from nmin to nmax in montgomery form.  If the
	comparison fails, the CPU is notified by setting a flag.

*/

// r0 + 2^64 * r1 = a * b
ulong2 mul_wide(const ulong a, const ulong b){
	ulong2 r;
#ifdef __NV_CL_C_VERSION
	const uint a0 = (uint)(a), a1 = (uint)(a >> 32);
	const uint b0 = (uint)(b), b1 = (uint)(b >> 32);
	uint c0 = a0 * b0, c1 = mul_hi(a0, b0), c2, c3;
	asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r" (c1) : "r" (a0), "r" (b1), "r" (c1));
	asm volatile ("madc.hi.u32 %0, %1, %2, 0;" : "=r" (c2) : "r" (a0), "r" (b1));
	asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r" (c2) : "r" (a1), "r" (b1), "r" (c2));
	asm volatile ("madc.hi.u32 %0, %1, %2, 0;" : "=r" (c3) : "r" (a1), "r" (b1));
	asm volatile ("mad.lo.cc.u32 %0, %1, %2, %3;" : "=r" (c1) : "r" (a1), "r" (b0), "r" (c1));
	asm volatile ("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r" (c2) : "r" (a1), "r" (b0), "r" (c2));
	asm volatile ("addc.u32 %0, %1, 0;" : "=r" (c3) : "r" (c3));
	r.s0 = upsample(c1, c0); r.s1 = upsample(c3, c2);
#else
	r.s0 = a * b; r.s1 = mul_hi(a, b);
#endif
	return r;
}

ulong m_mul(ulong a, ulong b, ulong p, ulong q){
	ulong2 ab = mul_wide(a,b);
	ulong m = ab.s0 * q;
	ulong mp = mul_hi(m,p);
	ulong r = ab.s1 - mp;
	return ( ab.s1 < mp ) ? r + p : r;
}

ulong add(ulong a, ulong b, ulong p){
	ulong r;
	ulong c = (a >= p - b) ? p : 0;
	r = a + b - c;
	return r;
}

__kernel __attribute__ ((reqd_work_group_size(256, 1, 1))) void check(	__global ulong8 * g_prime,
									__global uint * g_primecount,
									__global ulong * g_sum,
									const uint nmax ) {

	const uint gid = get_global_id(0);
	const uint lid = get_local_id(0);
	const uint pcnt = g_primecount[0];
	__local ulong sum[256];

	if(gid < pcnt){
		// .s0=p, .s1=q, .s2=r2, .s3=one, .s4=two, .s5=nmo, .s6=residue of final factorial, .s7= montgomery form of last n
		const ulong8 prime = g_prime[gid];

		sum[lid] = prime.s6 + prime.s7;

		// convert last n out of montgomery form
		uint result = (uint)m_mul(prime.s7, 1, prime.s0, prime.s1);

		// adjust result for case where nmax > pmin
		if(prime.s0 <= nmax){
			if(nmax % prime.s0 == result){
				result = nmax;
			}
		}

		if(result != nmax){
			atomic_or(&g_primecount[5], 1);
		}
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
		g_sum[index] += sum[0];
	}

	if(gid == 0){
		// add primecount to total primecount
		g_sum[0] += pcnt;
		// store largest kernel prime count for array bounds check
		if( pcnt > g_primecount[1] ){
			g_primecount[1] = pcnt;
		}
	}

}





