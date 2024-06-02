/*

	setup kernel
	
	generates nmin! mod P using power table

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

__kernel void setup(__global ulong8 * g_prime, __global uint * g_primecount,
		 	__global uint * g_smallprimes, __global uint * g_smallpowers, const uint start, const uint end) {

	const uint gid = get_global_id(0);

	if(gid < g_primecount[0]){

		// .s0=p, .s1=q, .s2=r2, .s3=one, .s4=two, .s5=nmo, .s6=residue of startN! mod P, .s7=startN in montgomery form
		const ulong8 prime = g_prime[gid];
		uint i = start;
		ulong total = prime.s6;

		if(start == 0){
			++i;
			// first iteration, base prime = 2
			uint exp = g_smallpowers[0];
			// left to right binary exponentiation
			uint curBit = 0x80000000;
			curBit >>= ( clz(exp) + 1 );
			ulong a = prime.s4;
			while( curBit ){
				a = m_mul(a, a, prime.s0, prime.s1);
				if(exp & curBit){
					a = add(a, a, prime.s0);		// base 2 we can add
				}
				curBit >>= 1;
			}
			total = a;
		}
		for(; i<end; ++i){
			// remaining iterations, starting at prime = 3
			uint sm_prime = g_smallprimes[i];
			uint exp = g_smallpowers[i];
			ulong x = m_mul(sm_prime, prime.s2, prime.s0, prime.s1);
			ulong primepow;
			if(exp == 1){
				primepow = x;
			}
			else{
				uint curBit = 0x80000000;
				curBit >>= ( clz(exp) + 1 );
				ulong a = x;
				while( curBit ){
					a = m_mul(a, a, prime.s0, prime.s1);
					if(exp & curBit){
						a = m_mul(a, x, prime.s0, prime.s1);
					}
					curBit >>= 1;
				}
				primepow = a;
			}
			total = m_mul(total, primepow, prime.s0, prime.s1);
		}

		// done with power table, store to global
		// residue is equal to startN! mod P
		g_prime[gid].s6 = total;
	}
}



