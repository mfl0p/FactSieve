/* 

	powers.cl - Bryan Little 2/2025

	generate 32 bit prime/power table for fast calculation of nmin! mod P

	this kernel only runs once at program start

*/

ulong mul_wide_u32 (const uint a, const uint b) {

	ulong c;
#ifdef __NV_CL_C_VERSION
	asm volatile ("mul.wide.u32 %0, %1, %2;" : "+l" (c) : "r" (a) , "r" (b));
#else
	c = upsample(mul_hi(a, b), a*b);
#endif
	return c;
}

__kernel void powers(	__global uint * g_smallprimes,
			__global uint2 * g_smallpowers,
			const uint startN, const uint smallcount )
{
	const uint gid = get_global_id(0);
	const uint gs = get_global_size(0);

	for(uint position = gid; position < smallcount; position+=gs){

		uint prime = g_smallprimes[position];
		uint totalpower = startN / prime;			// 2^1, 3^1, 5^1 ...

		if(prime < 46340){					// sqrt(maximum N)
			uint currpower = prime*prime;
			uint q = startN / currpower;			// 2^2, 3^2, 5^2 ...
			while( q > 0 ){
				totalpower += q;
				ulong nextpow = mul_wide_u32(currpower, prime);
				if(nextpow > startN){
					break;
				}
				currpower = (uint)nextpow;
				q = startN / currpower;			// 2^3, 3^3, 5^3 ...
			}
		}

		uint curBit = 0x80000000;
		if(totalpower > 1){
			curBit >>= ( clz(totalpower) + 1 );
		}
		g_smallpowers[position] = (uint2)(totalpower, curBit);

	}

}








