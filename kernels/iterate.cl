/* 

	iterate.cl - Bryan Little 6/2024, montgomery arithmetic by Yves Gallot
	
	iterate from nmin! to nmax-1! mod P

	The CPU will run this kernel in many small chunks to limit kernel runtime.

	Iterate and setup kernels are the main compute intensive kernels.
	
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


__kernel void iterate(		__global ulong8 * g_prime,
				__global uint * g_primecount,
				__global ulong2 * g_factor,
				const uint startN,
				const uint endN )
{
	const uint gid = get_global_id(0);

	if(gid < g_primecount[0]){
		// .s0=p, .s1=q, .s2=r2, .s3=one, .s4=two, .s5=nmo, .s6=residue, .s7=N in montgomery form
		ulong8 prime = g_prime[gid];

		for(uint currN = startN; currN < endN; ++currN){	
			if(prime.s6 == prime.s3){
				// -1 factor
				uint i = atomic_inc(&g_primecount[2]);
				int n = -((int)currN);				// sign bit of n is factor +1 or -1
				g_factor[i] = (ulong2)(prime.s0, (ulong)n);
			}
			else if(prime.s6 == prime.s5){
				// +1 factor
				uint i = atomic_inc(&g_primecount[2]);
				int n = (int)currN;
				g_factor[i] = (ulong2)(prime.s0, (ulong)n);
			}
			prime.s7 = add(prime.s7, prime.s3, prime.s0);
			prime.s6 = m_mul(prime.s6, prime.s7, prime.s0, prime.s1);
		}

		// store final residue and n
		g_prime[gid].s6 = prime.s6;
		g_prime[gid].s7 = prime.s7;
	}
}




