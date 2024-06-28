
// cl_sieve.h

typedef struct {

	uint64_t pmin = 0, pmax = 0;
	uint32_t nmin = 0, nmax = 0;
	bool test = false;
	bool verify = false;
	uint64_t checksum = 0;
	bool compute = false;
	int computeunits;
	uint64_t primecount = 0;
	uint64_t factorcount = 0;
	uint32_t nstep;
	uint32_t sstep;
	uint32_t numresults = 1000000;
	uint64_t workunit;
	uint64_t p;
	bool write_state_a_next = true;
	uint64_t last_trickle;

}searchData;


typedef struct {

	uint32_t range;
	uint32_t psize;
	uint32_t numgroups;

	cl_mem d_factorP = NULL;
	cl_mem d_factorN = NULL;
	cl_mem d_factorVal = NULL;

	cl_mem d_sum = NULL;

	cl_mem d_primes = NULL;
	cl_mem d_primecount = NULL;

	cl_mem d_SmallPrimes = NULL;
	cl_mem d_SmallPowers = NULL;

	sclSoft check, iterate, clearn, clearresult, setup, getsegprimes, powers, verifyslow, verifypow, verifyreduce, verifyresult;

}progData;


typedef struct {

	uint64_t p;
	uint32_t n;
	int32_t c;

}factorData;

void cl_sieve( sclHard hardware, searchData & sd );

void run_test( sclHard hardware, searchData & sd );
