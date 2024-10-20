/* 
	verifyprime.c

	Bryan Little October 2024

	functions to verify the factor is prime and to verify the factor on CPU

	Montgomery arithmetic by Yves Gallot,
	Peter L. Montgomery, Modular multiplication without trial division, Math. Comp.44 (1985), 519–521.

	Optimal 7 base primality test by Jim Sinclair
	see https://miller-rabin.appspot.com/

*/

#include <cinttypes>


uint64_t invert(uint64_t p)
{
	uint64_t p_inv = 1, prev = 0;
	while (p_inv != prev) { prev = p_inv; p_inv *= 2 - p * p_inv; }
	return p_inv;
}


uint64_t m_mul(uint64_t a, uint64_t b, uint64_t p, uint64_t q)
{
	unsigned __int128 res;

	res  = (unsigned __int128)a * b;
	uint64_t ab0 = (uint64_t)res;
	uint64_t ab1 = res >> 64;

	uint64_t m = ab0 * q;

	res = (unsigned __int128)m * p;
	uint64_t mp = res >> 64;

	uint64_t r = ab1 - mp;

	return ( ab1 < mp ) ? r + p : r;
}


uint64_t add(uint64_t a, uint64_t b, uint64_t p)
{
	uint64_t r;

	uint64_t c = (a >= p - b) ? p : 0;

	r = a + b - c;

	return r;
}


/* Used in the prime validator
   Returns 0 only if p is composite.
   Otherwise p is a strong probable prime to base a.
 */
bool strong_prp(uint32_t base, uint64_t p, uint64_t q, uint64_t one, uint64_t pmo, uint64_t r2, int t, uint64_t exp, uint64_t curBit)
{
	/* If p is prime and p = d*2^t+1, where d is odd, then either
		1.  a^d = 1 (mod p), or
		2.  a^(d*2^s) = -1 (mod p) for some s in 0 <= s < t    */

	uint64_t a = m_mul(base,r2,p,q);  // convert base to montgomery form
	const uint64_t mbase = a;

  	/* r <-- a^d mod p, assuming d odd */
	while( curBit )
	{
		a = m_mul(a,a,p,q);

		if(exp & curBit){
			a = m_mul(a,mbase,p,q);
		}

		curBit >>= 1;
	}

	/* Clause 1. and s = 0 case for clause 2. */
	if (a == one || a == pmo){
		return true;
	}

	/* 0 < s < t cases for clause 2. */
	for (int s = 1; s < t; ++s){

		a = m_mul(a,a,p,q);

		if(a == pmo){
	    		return true;
		}
	}


	return false;
}


// prime if the number passes prp test to 7 bases.  good to 2^64
// this is very fast
bool isPrime(uint64_t p)
{
	const uint32_t bases[7] = {2, 325, 9375, 28178, 450775, 9780504, 1795265022};

	if (p % 2==0)
		return false;

	uint64_t q = invert(p);
	uint64_t one = (-p) % p;
	uint64_t pmo = p - one;
	uint64_t two = add(one, one, p);
	uint64_t r2 = add(two, two, p);
	for (int i = 0; i < 5; ++i)
		r2 = m_mul(r2, r2, p, q);	// 4^{2^5} = 2^64

	int t = __builtin_ctzll( (p-1) );
	uint64_t exp = p >> t;
	uint64_t curBit = 0x8000000000000000;
	curBit >>= ( __builtin_clzll(exp) + 1 );

	for (int i = 0; i < 7; ++i){

		uint32_t base = bases[i];

		// needed for composite bases
		if (base >= p){
			base %= p;
			if (base == 0)
				continue;
		}

		if (!strong_prp(base, p, q, one, pmo, r2, t, exp, curBit))
			return false;
	}

	return true;
}



// verifies the factor on CPU using slow algorithm
bool verify(uint64_t p, uint32_t n, int32_t c)
{
	// precompute 34! that fits in 128 bits
	const unsigned __int128 f34 = ((unsigned __int128)0xde1bc4d19efcac82 << 64) | 0x445da75b00000000;

	uint64_t result = f34 % p;

	if(p < 0xFFFFFFFF){
		for(uint32_t i=35; i<=n; ++i){
			result = (result * i) % p;
		}
	}
	else{
		for(uint32_t i=35; i<=n; ++i){
			result = ((unsigned __int128)result * i) % p;
		}
	}

	if(result == 1 && c == -1){
		return true;
	}
	else if(result == p-1 && c == 1){
		return true;
	}

	return false;

}








