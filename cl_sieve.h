
// cl_sieve.h

typedef struct {
	cl_ulong p;
	cl_int nc;
}factor;

typedef struct {
	uint64_t pmin, pmax, p, checksum, primecount, factorcount, last_trickle, state_sum;
	uint32_t nmin, nmax;
}workStatus;

typedef struct {
	uint64_t maxmalloc;
	uint32_t computeunits, nstep, sstep, smcount, numresults, threadcount, range, psize, numgroups;
	bool test, compute, write_state_a_next;
}searchData;

typedef struct {
	cl_mem d_factor;
	cl_mem d_sum;
	cl_mem d_primes;
	cl_mem d_primecount;
	cl_mem d_SmallPrimes;
	cl_mem d_SmallPowers;
	sclSoft check, iterate, clearn, clearresult, setup, getsegprimes, powers, verifyslow, verifypow, verifyreduce, verifyresult;
}progData;

void cl_sieve( sclHard hardware, workStatus & st, searchData & sd );

void run_test( sclHard hardware, workStatus & st, searchData & sd );
