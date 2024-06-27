/* 

	verifyresult.cl - Bryan Little 6/2024, montgomery arithmetic by Yves Gallot
	
	verify power table final reduction and compare

*/



// r0 + 2^64 * r1 = a * b
ulong2 mul_wide(const ulong a, const ulong b)
{
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


ulong m_mul(ulong a, ulong b, ulong p, ulong q)
{
	ulong2 ab = mul_wide(a,b);

	ulong m = ab.s0 * q;

	ulong mp = mul_hi(m,p);

	ulong r = ab.s1 - mp;

	return ( ab.s1 < mp ) ? r + p : r;
}


ulong add(ulong a, ulong b, ulong p)
{
	ulong r;

	ulong c = (a >= p - b) ? p : 0;

	r = a + b - c;

	return r;
}


__kernel __attribute__ ((reqd_work_group_size(256, 1, 1))) void verifyresult(
				__global ulong8 * g_prime,
				__global ulong4 * g_verify,
				__global uint *g_primecount,
				const uint num_totals				)
{
	const uint lid = get_local_id(0);
	__local ulong2 total[256];
	// .s0=p, .s1=q, .s2=r2, .s3=one, .s4=two, .s5=nmo, .s6=residue
	const ulong8 prime = g_prime[0];

	if(lid < num_totals){					// needs to be <=256
		total[lid].s0 = g_verify[lid].s2;
		total[lid].s1 = g_verify[lid].s3;
	}
	else{
		total[lid].s0 = prime.s3;
		total[lid].s1 = prime.s3;
	}

	barrier(CLK_LOCAL_MEM_FENCE);

	// local memory reduction.  group size is forced to 256
//	for(uint s = get_local_size(0) / 2; s > 0; s >>= 1){
	for(uint s = 128; s > 0; s >>= 1){
		if(lid < s){
			total[lid].s0 = m_mul( total[lid].s0, total[lid+s].s0, prime.s0, prime.s1);
			total[lid].s1 = m_mul( total[lid].s1, total[lid+s].s1, prime.s0, prime.s1);
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	if(lid == 0){
		// factorial calculated with power table should match iterative algorithm
		if( total[0].s0 != total[0].s1 ){
			g_primecount[3] = 1;
		}
	}
}





