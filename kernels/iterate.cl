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
				__global ulong * g_specialprime,
				__global uint * g_specialn,
				__global int * g_specialval,
				const uint startN,
				const uint endN )
{
	const uint gid = get_global_id(0);

	if(gid < g_primecount[0]){
		// .s0=p, .s1=q, .s2=r2, .s3=one, .s4=two, .s5=nmo, .s6=residue of startN! mod P, .s7=startN in montgomery form
		const ulong8 prime = g_prime[gid];
		ulong residue = prime.s6;			// startN! mod P
		ulong next = prime.s7;				// montgomery form of startN

		for(uint currN = startN; currN < endN; ++currN){	
			if(residue == prime.s3){
				// -1 factor
				uint i = atomic_inc(&g_primecount[2]);
				g_specialprime[i] = prime.s0;
				g_specialn[i] = currN;
				g_specialval[i] = -1;
			}
			else if(residue == prime.s5){
				// +1 factor
				uint i = atomic_inc(&g_primecount[2]);
				g_specialprime[i] = prime.s0;
				g_specialn[i] = currN;
				g_specialval[i] = 1;
			}
			next = add(next, prime.s3, prime.s0);
			residue = m_mul(residue, next, prime.s0, prime.s1);
		}

		// store final residues
		g_prime[gid].s6 = residue;
		g_prime[gid].s7 = next;
	}
}




