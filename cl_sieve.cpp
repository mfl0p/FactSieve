/*
	FactSieve
	Bryan Little, Jun 2024
	
	with contributions by Yves Gallot, Mark Rodenkirch, and Kim Walisch

	Required minimum OpenCL version is 1.1
	CL_TARGET_OPENCL_VERSION 110 in simpleCL.h

	Search limits:  P up to 2^64 and N up to 2^31

	Using OpenMP for multithreaded factor verification.

*/

#include <unistd.h>
#include <cinttypes>
#include <math.h>
#include <omp.h>

#include "boinc_api.h"
#include "boinc_opencl.h"
#include "simpleCL.h"

#include "check.h"
#include "clearn.h"
#include "clearresult.h"
#include "powers.h"
#include "getsegprimes.h"
#include "setup.h"
#include "iterate.h"
#include "verifyslow.h"
#include "verifypow.h"
#include "verifyreduce.h"
#include "verifyresult.h"

#include "primesieve.h"
#include "putil.h"
#include "cl_sieve.h"
#include "verifyprime.h"

#define RESULTS_FILENAME "factors.txt"
#define STATE_FILENAME_A "FSstateA.txt"
#define STATE_FILENAME_B "FSstateB.txt"



void handle_trickle_up(searchData & sd)
{
	if(boinc_is_standalone()) return;

	uint64_t now = (uint64_t)time(NULL);

	if( (now-sd.last_trickle) > 86400 ){	// Once per day

		sd.last_trickle = now;

		double progress = boinc_get_fraction_done();
		double cpu;
		boinc_wu_cpu_time(cpu);
		APP_INIT_DATA init_data;
		boinc_get_init_data(init_data);
		double run = boinc_elapsed_time() + init_data.starting_elapsed_time;

		char msg[512];
		sprintf(msg, "<trickle_up>\n"
			    "   <progress>%lf</progress>\n"
			    "   <cputime>%lf</cputime>\n"
			    "   <runtime>%lf</runtime>\n"
			    "</trickle_up>\n",
			     progress, cpu, run  );
		char variety[64];
		sprintf(variety, "factsieve_progress");
		boinc_send_trickle_up(variety, msg);
	}

}


FILE *my_fopen(const char * filename, const char * mode)
{
	char resolved_name[512];

	boinc_resolve_filename(filename,resolved_name,sizeof(resolved_name));
	return boinc_fopen(resolved_name,mode);

}


void cleanup( progData pd ){
	sclReleaseMemObject(pd.d_factorP);
	sclReleaseMemObject(pd.d_factorN);
	sclReleaseMemObject(pd.d_factorVal);

	sclReleaseMemObject(pd.d_sum);

	sclReleaseMemObject(pd.d_primes);
	sclReleaseMemObject(pd.d_primecount);

	sclReleaseMemObject(pd.d_SmallPrimes);
	sclReleaseMemObject(pd.d_SmallPowers);

	sclReleaseClSoft(pd.check);
	sclReleaseClSoft(pd.clearn);
	sclReleaseClSoft(pd.clearresult);
        sclReleaseClSoft(pd.iterate);
        sclReleaseClSoft(pd.setup);
        sclReleaseClSoft(pd.powers);
        sclReleaseClSoft(pd.getsegprimes);

        sclReleaseClSoft(pd.verifyslow);
        sclReleaseClSoft(pd.verifypow);
        sclReleaseClSoft(pd.verifyreduce);
        sclReleaseClSoft(pd.verifyresult);

}


void write_state( searchData & sd ){

	FILE *out;

        if (sd.write_state_a_next){
		if ((out = my_fopen(STATE_FILENAME_A,"w")) == NULL)
			fprintf(stderr,"Cannot open %s !!!\n",STATE_FILENAME_A);
	}
	else{
                if ((out = my_fopen(STATE_FILENAME_B,"w")) == NULL)
                        fprintf(stderr,"Cannot open %s !!!\n",STATE_FILENAME_B);
        }
	if (fprintf(out,"%" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",sd.workunit,sd.p,sd.primecount,sd.checksum,sd.factorcount,sd.last_trickle) < 0){
		if (sd.write_state_a_next)
			fprintf(stderr,"Cannot write to %s !!! Continuing...\n",STATE_FILENAME_A);
		else
			fprintf(stderr,"Cannot write to %s !!! Continuing...\n",STATE_FILENAME_B);

		// Attempt to close, even though we failed to write
		fclose(out);
	}
	else{
		// If state file is closed OK, write to the other state file
		// next time around
		if (fclose(out) == 0) 
			sd.write_state_a_next = !sd.write_state_a_next; 
	}
}

/* Return 1 only if a valid checkpoint can be read.
   Attempts to read from both state files,
   uses the most recent one available.
 */
int read_state( searchData & sd ){

	FILE *in;
	bool good_state_a = true;
	bool good_state_b = true;
	uint64_t workunit_a, workunit_b;
	uint64_t current_a, current_b;
	uint64_t primecount_a, primecount_b;
	uint64_t checksum_a, checksum_b;
	uint64_t factorcount_a, factorcount_b;
	uint64_t trickle_a, trickle_b;

        // Attempt to read state file A
	if ((in = my_fopen(STATE_FILENAME_A,"r")) == NULL){
		good_state_a = false;
        }
	else if (fscanf(in,"%" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",&workunit_a,&current_a,&primecount_a,&checksum_a,&factorcount_a,&trickle_a) != 6){
		fprintf(stderr,"Cannot parse %s !!!\n",STATE_FILENAME_A);
		good_state_a = false;
	}
	else{
		fclose(in);
		/* Check that start stop match */
		if (workunit_a != sd.workunit){
			good_state_a = false;
		}
	}

        // Attempt to read state file B
        if ((in = my_fopen(STATE_FILENAME_B,"r")) == NULL){
                good_state_b = false;
        }
	else if (fscanf(in,"%" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 " %" PRIu64 "\n",&workunit_b,&current_b,&primecount_b,&checksum_b,&factorcount_b,&trickle_b) != 6){
                fprintf(stderr,"Cannot parse %s !!!\n",STATE_FILENAME_B);
                good_state_b = false;
        }
        else{
                fclose(in);
		/* Check that start stop match */
		if (workunit_b != sd.workunit){
                        good_state_b = false;
                }
        }

        // If both state files are OK, check which is the most recent
	if (good_state_a && good_state_b)
	{
		if (current_a > current_b)
			good_state_b = false;
		else
			good_state_a = false;
	}

        // Use data from the most recent state file
	if (good_state_a && !good_state_b)
	{
		sd.p = current_a;
		sd.primecount = primecount_a;
		sd.checksum = checksum_a;
		sd.factorcount = factorcount_a;
		sd.last_trickle = trickle_a;
		sd.write_state_a_next = false;
		return 1;
	}
        if (good_state_b && !good_state_a)
        {
                sd.p = current_b;
		sd.primecount = primecount_b;
		sd.checksum = checksum_b;
		sd.factorcount = factorcount_b;
		sd.last_trickle = trickle_b;
		sd.write_state_a_next = true;
		return 1;
        }

	// If we got here, neither state file was good
	return 0;
}


void checkpoint( searchData & sd ){

	handle_trickle_up( sd );

	write_state( sd );

	if(boinc_is_standalone()){
		printf("Checkpoint, current p: %" PRIu64 "\n", sd.p);
	}

	boinc_checkpoint_completed();
}


// sleep CPU thread while waiting on the specified event to complete in the command queue
// using critical sections to prevent BOINC from shutting down the program while kernels are running on the GPU
void waitOnEvent(sclHard hardware, cl_event event){

	cl_int err;
	cl_int info;
#ifdef _WIN32
#else
	struct timespec sleep_time;
	sleep_time.tv_sec = 0;
	sleep_time.tv_nsec = 1000000;	// 1ms
#endif

	boinc_begin_critical_section();

	err = clFlush(hardware.queue);
	if ( err != CL_SUCCESS ) {
		printf( "ERROR: clFlush\n" );
		fprintf(stderr, "ERROR: clFlush\n" );
		sclPrintErrorFlags( err );
       	}

	while(true){

#ifdef _WIN32
		Sleep(1);
#else
		nanosleep(&sleep_time,NULL);
#endif

		err = clGetEventInfo(event, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &info, NULL);
		if ( err != CL_SUCCESS ) {
			printf( "ERROR: clGetEventInfo\n" );
			fprintf(stderr, "ERROR: clGetEventInfo\n" );
			sclPrintErrorFlags( err );
	       	}

		if(info == CL_COMPLETE){
			err = clReleaseEvent(event);
			if ( err != CL_SUCCESS ) {
				printf( "ERROR: clReleaseEvent\n" );
				fprintf(stderr, "ERROR: clReleaseEvent\n" );
				sclPrintErrorFlags( err );
		       	}

			boinc_end_critical_section();

			return;
		}
	}
}


// queue a marker and sleep CPU thread until marker has been reached in the command queue
void sleepCPU(sclHard hardware){

	cl_event kernelsDone;
	cl_int err;
	cl_int info;
#ifdef _WIN32
#else
	struct timespec sleep_time;
	sleep_time.tv_sec = 0;
	sleep_time.tv_nsec = 1000000;	// 1ms
#endif

	boinc_begin_critical_section();

	// OpenCL v2.0
/*
	err = clEnqueueMarkerWithWaitList( hardware.queue, 0, NULL, &kernelsDone);
	if ( err != CL_SUCCESS ) {
		printf( "ERROR: clEnqueueMarkerWithWaitList\n");
		fprintf(stderr, "ERROR: clEnqueueMarkerWithWaitList\n");
		sclPrintErrorFlags(err); 
	}
*/
	err = clEnqueueMarker( hardware.queue, &kernelsDone);
	if ( err != CL_SUCCESS ) {
		printf( "ERROR: clEnqueueMarker\n");
		fprintf(stderr, "ERROR: clEnqueueMarker\n");
		sclPrintErrorFlags(err); 
	}

	err = clFlush(hardware.queue);
	if ( err != CL_SUCCESS ) {
		printf( "ERROR: clFlush\n" );
		fprintf(stderr, "ERROR: clFlush\n" );
		sclPrintErrorFlags( err );
       	}

	while(true){

#ifdef _WIN32
		Sleep(1);
#else
		nanosleep(&sleep_time,NULL);
#endif

		err = clGetEventInfo(kernelsDone, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int), &info, NULL);
		if ( err != CL_SUCCESS ) {
			printf( "ERROR: clGetEventInfo\n" );
			fprintf(stderr, "ERROR: clGetEventInfo\n" );
			sclPrintErrorFlags( err );
	       	}

		if(info == CL_COMPLETE){
			err = clReleaseEvent(kernelsDone);
			if ( err != CL_SUCCESS ) {
				printf( "ERROR: clReleaseEvent\n" );
				fprintf(stderr, "ERROR: clReleaseEvent\n" );
				sclPrintErrorFlags( err );
		       	}

			boinc_end_critical_section();

			return;
		}
	}
}



// find mod 30 wheel index based on starting N
// this is used by gpu threads to iterate over the number line
void findWheelOffset(uint64_t & start, int32_t & index){

	int32_t wheel[8] = {4, 2, 4, 2, 4, 6, 2, 6};
	int32_t idx = -1;

	// find starting number using mod 6 wheel
	// N=(k*6)-1, N=(k*6)+1 ...
	// where k, k+1, k+2 ...
	uint64_t k = start / 6;
	int32_t i = 1;
	uint64_t N = (k * 6)-1;


	while( N < start || N % 5 == 0 ){
		if(i){
			i = 0;
			N += 2;
		}
		else{
			i = 1;
			N += 4;
		}
	}

	start = N;

	// find mod 30 wheel index by iterating with a mod 6 wheel until finding N divisible by 5
	// forward to find index
	while(idx < 0){

		if(i){
			N += 2;
			i = 0;
			if(N % 5 == 0){
				N -= 2;
				idx = 5;
			}

		}
		else{
			N += 4;
			i = 1;
			if(N % 5 == 0){
				N -= 4;
				idx = 7;
			}
		}
	}

	// reverse to find starting index
	while(N != start){
		--idx;
		if(idx < 0)idx=7;
		N -= wheel[idx];
	}


	index = idx;

}


int fcomp(const void *a, const void *b) {
  
	factorData *factA = (factorData *)a;
	factorData *factB = (factorData *)b;

	if(factB->p < factA->p){
		return 1;
	}
	else if(factB->p == factA->p){
		if(factB->n < factA->n){
			return 1;
		}
	}

	return -1;
}


void getResults( progData pd, searchData & sd, sclHard hardware ){

	uint64_t * h_checksum = (uint64_t *)malloc(pd.numgroups*sizeof(uint64_t));
	if( h_checksum == NULL ){
		fprintf(stderr,"malloc error\n");
		exit(EXIT_FAILURE);
	}

	// copy checksum and total prime count to host memory
	// blocking read
	sclRead(hardware, pd.numgroups*sizeof(uint64_t), pd.d_sum, h_checksum);

	// index 0 is the gpu's total prime count
	sd.primecount += h_checksum[0];

	// sum blocks
	for(uint32_t i=1; i<pd.numgroups; ++i){
		sd.checksum += h_checksum[i];
	}

	uint32_t * h_primecount = (uint32_t *)malloc(6*sizeof(uint32_t));
	if( h_primecount == NULL ){
		fprintf(stderr,"malloc error\n");
		exit(EXIT_FAILURE);
	}

	// copy prime count to host memory
	// blocking read
	sclRead(hardware, 6*sizeof(uint32_t), pd.d_primecount, h_primecount);

	// largest kernel prime count.  used to check array bounds
	if(h_primecount[1] > pd.psize){
		fprintf(stderr,"error: gpu prime array overflow\n");
		printf("error: gpu prime array overflow\n");
		exit(EXIT_FAILURE);
	}

	// flag set if there is a gpu power table error
	if(h_primecount[3] == 1){
		fprintf(stderr,"error: power table verification failed\n");
		printf("error: power table verification failed\n");
		exit(EXIT_FAILURE);
	}

	// flag set if there is a gpu overflow error
	if(h_primecount[4] == 1){
		fprintf(stderr,"error: getsegprimes kernel local memory overflow\n");
		printf("error: getsegprimes kernel local memory overflow\n");
		exit(EXIT_FAILURE);
	}

	// flag set if there is a gpu validation failure
	if(h_primecount[5] == 1){
		fprintf(stderr,"error: gpu validation failure\n");
		printf("error: gpu validation failure\n");
		exit(EXIT_FAILURE);
	}

	uint32_t numfactors = h_primecount[2];

	if(numfactors > 0){

		if(numfactors > sd.numresults){
			fprintf(stderr,"Error: number of results (%u) overflowed array.\n", numfactors);
			exit(EXIT_FAILURE);
		}

		if(boinc_is_standalone()){
			printf("processing %u factors on CPU\n", numfactors);
		}

		uint64_t * h_factorP = (uint64_t *)malloc(numfactors * sizeof(uint64_t));
		if( h_factorP == NULL ){
			fprintf(stderr,"malloc error\n");
			exit(EXIT_FAILURE);
		}
		uint32_t * h_factorN = (uint32_t *)malloc(numfactors * sizeof(uint32_t));
		if( h_factorN == NULL ){
			fprintf(stderr,"malloc error\n");
			exit(EXIT_FAILURE);
		}
		int32_t * h_factorVal = (int32_t *)malloc(numfactors * sizeof(int32_t));
		if( h_factorVal == NULL ){
			fprintf(stderr,"malloc error\n");
			exit(EXIT_FAILURE);
		}

		// copy factors to host memory
		// blocking read
		sclRead(hardware, numfactors * sizeof(uint64_t), pd.d_factorP, h_factorP);
		sclRead(hardware, numfactors * sizeof(uint32_t), pd.d_factorN, h_factorN);
		sclRead(hardware, numfactors * sizeof(int32_t), pd.d_factorVal, h_factorVal);

		// move factors into struct so we can sort
		factorData * factors = (factorData *)malloc(numfactors * sizeof(factorData));
		if( factors == NULL ){
			fprintf(stderr,"malloc error\n");
			exit(EXIT_FAILURE);
		}
		for(uint32_t i = 0; i < numfactors; ++i){
			factors[i].p = h_factorP[i];
			factors[i].n = h_factorN[i];
			factors[i].c = h_factorVal[i];
		}

		free(h_factorP);
		free(h_factorN);
		free(h_factorVal);

		// sort results by prime size if needed
		if(numfactors > 1){
			if(boinc_is_standalone()){
				printf("sorting factors\n");
			}
			qsort(factors, numfactors, sizeof(factorData), fcomp);
		}

		// verify all factors on CPU using slow test
		if(boinc_is_standalone()){
			printf("Verifying factors on CPU...\n");
		}
		double last = 0.0;
		uint32_t tested = 0;

		#pragma omp parallel for
		for(uint32_t m=0; m<numfactors; ++m){
			if( verify( factors[m].p, factors[m].n, factors[m].c ) == false ){
				fprintf(stderr,"CPU factor verification failed!  %" PRIu64 " is not a factor of %u!%+d\n", factors[m].p, factors[m].n, factors[m].c);
				printf("\nCPU factor verification failed!  %" PRIu64 " is not a factor of %u!%+d\n", factors[m].p, factors[m].n, factors[m].c);
				exit(EXIT_FAILURE);
			}

			if(boinc_is_standalone()){
				#pragma omp atomic
				++tested;

				double done = (double)(tested+1) / (double)numfactors * 100.0;
				if(done > last+0.1){
					last = done;
					printf("\r%.1f%%     ",done);
					fflush(stdout);
				}
			}

		}

		fprintf(stderr,"Verified %u factors.\n", numfactors);
		if(boinc_is_standalone()){
			printf("\rVerified %u factors.\n", numfactors);
		}

		// write factors to file
		FILE * resfile = my_fopen(RESULTS_FILENAME,"a");

		if( resfile == NULL ){
			fprintf(stderr,"Cannot open %s !!!\n",RESULTS_FILENAME);
			exit(EXIT_FAILURE);
		}

		uint64_t lastgoodp = 0;

		if(boinc_is_standalone()){
			printf("writing factors to %s\n", RESULTS_FILENAME);
		}

		for(uint32_t m=0; m<numfactors; ++m){

			uint64_t p = factors[m].p;
			uint32_t n = factors[m].n;
			int32_t c = factors[m].c;

			if( p == lastgoodp ){		// avoid primality testing twice
				++sd.factorcount;
				if( fprintf( resfile, "%" PRIu64 " | %u!%+d\n",p,n,c) < 0 ){
					fprintf(stderr,"Cannot write to %s !!!\n",RESULTS_FILENAME);
					exit(EXIT_FAILURE);
				}
				// add the factor to checksum
				sd.checksum += n + c;
			}
			else if( isPrime(p) ){
				lastgoodp = p;
				++sd.factorcount;
				if( fprintf( resfile, "%" PRIu64 " | %u!%+d\n",p,n,c) < 0 ){
					fprintf(stderr,"Cannot write to %s !!!\n",RESULTS_FILENAME);
					exit(EXIT_FAILURE);
				}				
				// add the factor to checksum
				sd.checksum += n + c;
			}
			else{
				fprintf(stderr,"discarded 2-PRP factor %" PRIu64 "\n", p);
				printf("discarded 2-PRP factor %" PRIu64 "\n", p);
			}	
		}

		fclose(resfile);
		free(factors);

	}

	free(h_checksum);
	free(h_primecount);

}


void setupSearch(searchData & sd){

	sd.p = sd.pmin;

	if(sd.pmin == 0 || sd.pmax == 0){
		printf("-p and -P arguments are required\nuse -h for help\n");
		fprintf(stderr, "-p and -P arguments are required\n");
		exit(EXIT_FAILURE);
	}

	if(sd.nmin == 0 || sd.nmax == 0){
		printf("-n and -N arguments are required\nuse -h for help\n");
		fprintf(stderr, "-n and -N arguments are required\n");
		exit(EXIT_FAILURE);
	}

	if (sd.nmin > sd.nmax){
		printf("nmin <= nmax is required\nuse -h for help\n");
		fprintf(stderr, "nmin <= nmax is required\n");
		exit(EXIT_FAILURE);
	}

	if (sd.pmin > sd.pmax){
		printf("pmin <= pmax is required\nuse -h for help\n");
		fprintf(stderr, "pmin <= pmax is required\n");
		exit(EXIT_FAILURE);
	}

	if (sd.pmin < sd.nmin){
		printf("pmin must be >= nmin, there are no factors when p <= nmin\nuse -h for help\n");
		fprintf(stderr, "pmin must be >= nmin, there are no factors when p <= nmin\n");
		exit(EXIT_FAILURE);
	}

	// increase result buffer at low P range
	// it's still possible to overflow this with a fast GPU and large search range
	if(sd.pmin < 0xFFFFFFFF){
		sd.numresults = 30000000;
	}

	fprintf(stderr, "Starting sieve at p: %" PRIu64 " n: %u\nStopping sieve at P: %" PRIu64 " N: %u\n", sd.pmin, sd.nmin, sd.pmax, sd.nmax);
	if(boinc_is_standalone()){
		printf("Starting sieve at p: %" PRIu64 " n: %u\nStopping sieve at P: %" PRIu64 " N: %u\n", sd.pmin, sd.nmin, sd.pmax, sd.nmax);
	}

	// for checkpoints
	sd.workunit = sd.pmin + sd.pmax + (uint64_t)sd.nmin + (uint64_t)sd.nmax;

	// setup and iterate kernel size
	if(sd.compute){
		sd.sstep = 25 * sd.computeunits;
		sd.nstep = 300 * sd.computeunits;
	}
	else{
		sd.sstep = 5 * sd.computeunits;
		sd.nstep = 60 * sd.computeunits;
	}


}



void profileGPU(progData & pd, searchData sd, sclHard hardware, int debuginfo ){

	// calculate approximate chunk size based on gpu's compute units
	cl_int err = 0;

	uint64_t calc_range = sd.computeunits * (uint64_t)265000;

	// limit kernel global size
	if(calc_range > 4294900000){
		calc_range = 4294900000;
	}

	uint64_t estimated = calc_range;

	uint64_t prof_start = sd.p;

	uint64_t prof_stop = prof_start + calc_range;

	// check overflow at 2^64
	if(prof_stop < prof_start){
		prof_stop = 0xFFFFFFFFFFFFFFFF;
		calc_range = prof_stop - prof_start;
	}

	sclSetGlobalSize( pd.getsegprimes, (calc_range/60)+1 );

	// get a count of primes in the gpu worksize
	uint64_t prof_range_primes = primesieve_count_primes( prof_start, prof_stop );

	// calculate prime array size based on result
	uint64_t prof_mem_size = (uint64_t)(1.5 * (double)prof_range_primes);

	// kernels use uint for global id
	if(prof_mem_size > UINT32_MAX){
		fprintf(stderr, "ERROR: prof_mem_size too large.\n");
                printf( "ERROR: prof_mem_size too large.\n" );
		exit(EXIT_FAILURE);
	}

	// allocate temporary gpu prime array for profiling
	cl_mem d_profileprime = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, prof_mem_size*sizeof(cl_ulong8), NULL, &err );
	if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure.\n");
	        printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}

	int32_t prof_wheelidx;
	uint64_t prof_kernel_start = prof_start;

	findWheelOffset(prof_kernel_start, prof_wheelidx);

	// set static args
	sclSetKernelArg(pd.getsegprimes, 0, sizeof(uint64_t), &prof_kernel_start);
	sclSetKernelArg(pd.getsegprimes, 1, sizeof(uint64_t), &prof_stop);
	sclSetKernelArg(pd.getsegprimes, 2, sizeof(int32_t), &prof_wheelidx);
	sclSetKernelArg(pd.getsegprimes, 3, sizeof(cl_mem), &d_profileprime);
	sclSetKernelArg(pd.getsegprimes, 4, sizeof(cl_mem), &pd.d_primecount);
	sclSetKernelArg(pd.getsegprimes, 5, sizeof(uint32_t), &sd.nmin);

	// zero prime count
	sclEnqueueKernel(hardware, pd.clearn);

	// Benchmark the GPU
	double kernel_ms = ProfilesclEnqueueKernel(hardware, pd.getsegprimes);

	// target runtime for prime generator kernel is 1.0 ms
	double prof_multi = 1.0 / kernel_ms;

	// update chunk size based on the profile
	calc_range = (uint64_t)( (double)calc_range * prof_multi );

	// limit kernel global size
	if(calc_range > 4294900000){
		calc_range = 4294900000;
	}

	if(debuginfo){
		printf("pgen %" PRIu64 " %" PRIu64 "\n",estimated,calc_range);
	}

	// get a count of primes in the new gpu worksize
	prof_stop = prof_start + calc_range;

	// check overflow at 2^64
	if(prof_stop < prof_start){
		prof_stop = 0xFFFFFFFFFFFFFFFF;
		calc_range = prof_stop - prof_start;
	}
	uint64_t range_primes = primesieve_count_primes( prof_start, prof_stop );

	// calculate prime array size based on result
	uint64_t mem_size = (uint64_t)( 1.5 * (double)range_primes );

	if(mem_size > UINT32_MAX){
		fprintf(stderr, "ERROR: mem_size too large.\n");
                printf( "ERROR: mem_size too large.\n" );
		exit(EXIT_FAILURE);
	}

	pd.range = calc_range;
	pd.psize = mem_size;

	// free temporary array
	sclReleaseMemObject(d_profileprime);

}


void cl_sieve( sclHard hardware, searchData & sd ){

	progData pd;
	bool first_iteration = true;
	bool debuginfo = false;
	time_t boinc_last, boinc_curr;
	time_t ckpt_curr, ckpt_last;
	cl_int err = 0;

	// setup kernel parameters
	setupSearch(sd);

	// device arrays
	pd.d_primecount = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, 6*sizeof(cl_uint), NULL, &err );
        if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure.\n");
                printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}
        pd.d_factorP = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, sd.numresults*sizeof(cl_ulong), NULL, &err );
        if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure: factorP array.\n");
                printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}
        pd.d_factorN = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, sd.numresults*sizeof(cl_uint), NULL, &err );
        if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure: factorN array.\n");
                printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}
        pd.d_factorVal = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, sd.numresults*sizeof(cl_int), NULL, &err );
        if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure: factorVal array.\n");
                printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}

        pd.clearn = sclGetCLSoftware(clearn_cl,"clearn",hardware, 1, debuginfo);
        pd.clearresult = sclGetCLSoftware(clearresult_cl,"clearresult",hardware, 1, debuginfo);
        pd.powers = sclGetCLSoftware(powers_cl,"powers",hardware, 1, debuginfo);
        pd.setup = sclGetCLSoftware(setup_cl,"setup",hardware, 1, debuginfo);
        pd.iterate = sclGetCLSoftware(iterate_cl,"iterate",hardware, 1, debuginfo);
        pd.check = sclGetCLSoftware(check_cl,"check",hardware, 1, debuginfo);
        pd.verifyslow = sclGetCLSoftware(verifyslow_cl,"verifyslow",hardware, 1, debuginfo);
        pd.verifypow = sclGetCLSoftware(verifypow_cl,"verifypow",hardware, 1, debuginfo);
        pd.verifyreduce = sclGetCLSoftware(verifyreduce_cl,"verifyreduce",hardware, 1, debuginfo);
        pd.verifyresult = sclGetCLSoftware(verifyresult_cl,"verifyresult",hardware, 1, debuginfo);

	if(sd.pmax < 0xFFFFFFFFFF000000){
		// faster kernel with no overflow checking
	        pd.getsegprimes = sclGetCLSoftware(getsegprimes_cl,"getsegprimes",hardware, 1, debuginfo);
	}
	else{
		// use kernel that has overflow checking near 2^64
        	pd.getsegprimes = sclGetCLSoftware(getsegprimes_cl,"getsegprimesmax",hardware, 1, debuginfo);
		fprintf(stderr, "using pmax kernel\n");
	}


	// kernels have __attribute__ ((reqd_work_group_size(256, 1, 1)))
	// it's still possible the CL complier picked a different size
	if(pd.getsegprimes.local_size[0] != 256){
		pd.getsegprimes.local_size[0] = 256;
		fprintf(stderr, "Set getsegprimes kernel local size to 256\n");
	}
	if(pd.verifyslow.local_size[0] != 256){
		pd.verifyslow.local_size[0] = 256;
		fprintf(stderr, "Set verifyslow kernel local size to 256\n");
	}
	if(pd.verifypow.local_size[0] != 256){
		pd.verifypow.local_size[0] = 256;
		fprintf(stderr, "Set verifypow kernel local size to 256\n");
	}
	if(pd.verifyreduce.local_size[0] != 256){
		pd.verifyreduce.local_size[0] = 256;
		fprintf(stderr, "Set verifyreduce kernel local size to 256\n");
	}
	if(pd.verifyresult.local_size[0] != 256){
		pd.verifyresult.local_size[0] = 256;
		fprintf(stderr, "Set verifyresult kernel local size to 256\n");
	}
	if(pd.check.local_size[0] != 256){
		pd.check.local_size[0] = 256;
		fprintf(stderr, "Set check kernel local size to 256\n");
	}


	if( sd.test ){
		// clear result file
		FILE * temp_file = my_fopen(RESULTS_FILENAME,"w");
		if (temp_file == NULL){
			fprintf(stderr,"Cannot open %s !!!\n",RESULTS_FILENAME);
			exit(EXIT_FAILURE);
		}
		fclose(temp_file);
	}
	else{
		// Resume from checkpoint if there is one
		if( read_state( sd ) ){
			if(boinc_is_standalone()){
				printf("Resuming search from checkpoint. Current p: %" PRIu64 "\n", sd.p);
			}
			fprintf(stderr,"Resuming search from checkpoint. Current p: %" PRIu64 "\n", sd.p);

			//trying to resume a finished workunit
			if( sd.p == sd.pmax ){
				if(boinc_is_standalone()){
					printf("Workunit complete.\n");
				}
				fprintf(stderr,"Workunit complete.\n");
				boinc_finish(EXIT_SUCCESS);
			}
		}
		// starting from beginning
		else{
			// clear result file
			FILE * temp_file = my_fopen(RESULTS_FILENAME,"w");
			if (temp_file == NULL){
				fprintf(stderr,"Cannot open %s !!!\n",RESULTS_FILENAME);
				exit(EXIT_FAILURE);
			}
			fclose(temp_file);

			// setup boinc trickle up
			sd.last_trickle = (uint64_t)time(NULL);
		}
	}

	// kernel used in profileGPU, setup arg
	sclSetKernelArg(pd.clearn, 0, sizeof(cl_mem), &pd.d_primecount);
	sclSetGlobalSize( pd.clearn, 64 );

	profileGPU(pd,sd,hardware,debuginfo);

	// number of gpu workgroups, used to size the sum array on gpu
	pd.numgroups = (pd.psize / pd.check.local_size[0]) + 2;

	// primes for power table
	size_t primelistsize;
	uint32_t *smlist = (uint32_t*)primesieve_generate_primes(2, sd.nmin, &primelistsize, UINT32_PRIMES);
	uint32_t smcount = (uint32_t)primelistsize;

	pd.d_SmallPrimes = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, smcount*sizeof(cl_uint), NULL, &err );
        if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure: SmallPrimes array\n");
                printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}
	pd.d_SmallPowers = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, smcount*sizeof(cl_uint), NULL, &err );
        if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure: SmallPowers array.\n");
                printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}

	// send primes to gpu, blocking
	sclWrite(hardware, smcount * sizeof(uint32_t), pd.d_SmallPrimes, smlist);

	free(smlist);

	uint32_t stride = 2560000;

	sclSetGlobalSize( pd.powers, stride );
	sclSetGlobalSize( pd.getsegprimes, (pd.range/60)+1 );
	sclSetGlobalSize( pd.setup, pd.psize );
	sclSetGlobalSize( pd.iterate, pd.psize );
	sclSetGlobalSize( pd.check, pd.psize );
	sclSetGlobalSize( pd.clearresult, pd.numgroups );

	sclSetGlobalSize( pd.verifyslow, stride );
	sclSetGlobalSize( pd.verifypow, stride );
	uint32_t ver_groups = stride / 256;				// 10000
	sclSetGlobalSize( pd.verifyreduce, ver_groups );
	uint32_t red_groups = (ver_groups / 256)+1;			// 40
	sclSetGlobalSize( pd.verifyresult, red_groups );

	pd.d_primes = clCreateBuffer(hardware.context, CL_MEM_READ_WRITE, pd.psize*sizeof(cl_ulong8), NULL, &err);
        if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure.\n");
                printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}
        pd.d_sum = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, pd.numgroups*sizeof(cl_ulong), NULL, &err );
        if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure.\n");
                printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}
	cl_mem d_verify = clCreateBuffer( hardware.context, CL_MEM_READ_WRITE, ver_groups*sizeof(cl_ulong4), NULL, &err );
        if ( err != CL_SUCCESS ) {
		fprintf(stderr, "ERROR: clCreateBuffer failure.\n");
                printf( "ERROR: clCreateBuffer failure.\n" );
		exit(EXIT_FAILURE);
	}


	// set static kernel args
	sclSetKernelArg(pd.clearresult, 0, sizeof(cl_mem), &pd.d_primecount);
	sclSetKernelArg(pd.clearresult, 1, sizeof(cl_mem), &pd.d_sum);
	sclSetKernelArg(pd.clearresult, 2, sizeof(uint32_t), &pd.numgroups);

	sclSetKernelArg(pd.powers, 0, sizeof(cl_mem), &pd.d_SmallPrimes);
	sclSetKernelArg(pd.powers, 1, sizeof(cl_mem), &pd.d_SmallPowers);
	sclSetKernelArg(pd.powers, 2, sizeof(uint32_t), &stride);
	sclSetKernelArg(pd.powers, 3, sizeof(uint32_t), &sd.nmin);
	sclSetKernelArg(pd.powers, 4, sizeof(uint32_t), &smcount);

	sclSetKernelArg(pd.getsegprimes, 3, sizeof(cl_mem), &pd.d_primes);
	sclSetKernelArg(pd.getsegprimes, 4, sizeof(cl_mem), &pd.d_primecount);
	sclSetKernelArg(pd.getsegprimes, 5, sizeof(uint32_t), &sd.nmin);

	sclSetKernelArg(pd.setup, 0, sizeof(cl_mem), &pd.d_primes);
	sclSetKernelArg(pd.setup, 1, sizeof(cl_mem), &pd.d_primecount);
	sclSetKernelArg(pd.setup, 2, sizeof(cl_mem), &pd.d_SmallPrimes);
	sclSetKernelArg(pd.setup, 3, sizeof(cl_mem), &pd.d_SmallPowers);

	sclSetKernelArg(pd.iterate, 0, sizeof(cl_mem), &pd.d_primes);
	sclSetKernelArg(pd.iterate, 1, sizeof(cl_mem), &pd.d_primecount);
	sclSetKernelArg(pd.iterate, 2, sizeof(cl_mem), &pd.d_factorP);
	sclSetKernelArg(pd.iterate, 3, sizeof(cl_mem), &pd.d_factorN);
	sclSetKernelArg(pd.iterate, 4, sizeof(cl_mem), &pd.d_factorVal);

	sclSetKernelArg(pd.check, 0, sizeof(cl_mem), &pd.d_primes);
	sclSetKernelArg(pd.check, 1, sizeof(cl_mem), &pd.d_primecount);
	sclSetKernelArg(pd.check, 2, sizeof(cl_mem), &pd.d_sum);
	sclSetKernelArg(pd.check, 3, sizeof(uint32_t), &pd.numgroups);
	sclSetKernelArg(pd.check, 4, sizeof(uint32_t), &sd.nmax);

	sclSetKernelArg(pd.verifyslow, 0, sizeof(cl_mem), &pd.d_primes);
	sclSetKernelArg(pd.verifyslow, 1, sizeof(cl_mem), &d_verify);
	sclSetKernelArg(pd.verifyslow, 2, sizeof(uint32_t), &sd.nmin);
	sclSetKernelArg(pd.verifyslow, 3, sizeof(uint32_t), &stride);

	sclSetKernelArg(pd.verifypow, 0, sizeof(cl_mem), &pd.d_primes);
	sclSetKernelArg(pd.verifypow, 1, sizeof(cl_mem), &pd.d_SmallPrimes);
	sclSetKernelArg(pd.verifypow, 2, sizeof(cl_mem), &pd.d_SmallPowers);
	sclSetKernelArg(pd.verifypow, 3, sizeof(cl_mem), &d_verify);
	sclSetKernelArg(pd.verifypow, 4, sizeof(uint32_t), &smcount);
	sclSetKernelArg(pd.verifypow, 5, sizeof(uint32_t), &stride);

	sclSetKernelArg(pd.verifyreduce, 0, sizeof(cl_mem), &pd.d_primes);
	sclSetKernelArg(pd.verifyreduce, 1, sizeof(cl_mem), &d_verify);
	sclSetKernelArg(pd.verifyreduce, 2, sizeof(uint32_t), &ver_groups);

	sclSetKernelArg(pd.verifyresult, 0, sizeof(cl_mem), &pd.d_primes);
	sclSetKernelArg(pd.verifyresult, 1, sizeof(cl_mem), &d_verify);
	sclSetKernelArg(pd.verifyresult, 2, sizeof(cl_mem), &pd.d_primecount);
	sclSetKernelArg(pd.verifyresult, 3, sizeof(uint32_t), &red_groups);


	time(&boinc_last);
	time(&ckpt_last);
	time_t totals, totalf;
	if(boinc_is_standalone()){
		time(&totals);
	}

	float kernel_ms;
	uint32_t kernelq = 0;
	cl_event launchEvent = NULL;
	const double irsize = 1.0 / (double)(sd.pmax-sd.pmin);

	sclEnqueueKernel(hardware, pd.clearresult);

	// main search loop
	while(sd.p < sd.pmax){

		uint64_t stop = sd.p + pd.range;
		if(stop > sd.pmax || stop < sd.p){
			// ck overflow
			stop = sd.pmax;
		}

		// clear prime count
		sclEnqueueKernel(hardware, pd.clearn);

		// update BOINC fraction done every 2 sec
		time(&boinc_curr);
		if( ((int)boinc_curr - (int)boinc_last) > 1 ){
    			double fd = (double)(sd.p-sd.pmin)*irsize;
			boinc_fraction_done(fd);
			if(boinc_is_standalone()) printf("Sieve Progress: %.1f%%\n",fd*100.0);
			boinc_last = boinc_curr;
		}

		// 1 minute checkpoint
		time(&ckpt_curr);
		if( ((int)ckpt_curr - (int)ckpt_last) > 60 ){
			if(kernelq > 0){
				waitOnEvent(hardware, launchEvent);
				kernelq = 0;
			}
			sleepCPU(hardware);
			boinc_begin_critical_section();
			getResults(pd, sd, hardware);
			checkpoint(sd);
			boinc_end_critical_section();
			ckpt_last = ckpt_curr;
			// clear result arrays
			sclEnqueueKernel(hardware, pd.clearresult);
		}

		// get a segment of primes.  very fast, target kernel time is 1ms
		int32_t wheelidx;
		uint64_t kernel_start = sd.p;
		findWheelOffset(kernel_start, wheelidx);

		sclSetKernelArg(pd.getsegprimes, 0, sizeof(uint64_t), &kernel_start);
		sclSetKernelArg(pd.getsegprimes, 1, sizeof(uint64_t), &stop);
		sclSetKernelArg(pd.getsegprimes, 2, sizeof(int32_t), &wheelidx);
		sclEnqueueKernel(hardware, pd.getsegprimes);


		uint32_t sstart = 0;
		uint32_t smax;
		uint32_t nstart = sd.nmin;
		uint32_t nmax;

		// setup power table, then profile setup kernel once at program start.  adjust work size to target kernel runtime.
		if(first_iteration){
			// setup and verify power table
			sclEnqueueKernel(hardware, pd.powers);
			sclEnqueueKernel(hardware, pd.verifyslow);
			sclEnqueueKernel(hardware, pd.verifypow);
			sclEnqueueKernel(hardware, pd.verifyreduce);
			sclEnqueueKernel(hardware, pd.verifyresult);
			sleepCPU(hardware);
			getResults(pd, sd, hardware);
			sclEnqueueKernel(hardware, pd.clearresult);
			sclReleaseMemObject(d_verify);
			fprintf(stderr,"Setup and verified power table with %u primes (%" PRIu64 " bytes).  Starting sieve...\n",smcount, (uint64_t)smcount*2*4);
			if(boinc_is_standalone()){
				printf("Setup and verified power table with %u primes (%" PRIu64 " bytes).  Starting sieve...\n",smcount, (uint64_t)smcount*2*4);
			}

			smax = sstart + sd.sstep;
			if(smax > smcount)smax = smcount;
			sclSetKernelArg(pd.setup, 4, sizeof(uint32_t), &sstart);
			sclSetKernelArg(pd.setup, 5, sizeof(uint32_t), &smax);
			kernel_ms = ProfilesclEnqueueKernel(hardware, pd.setup);
			sstart += sd.sstep;
			double multi = (sd.compute)?(50.0 / kernel_ms):(10.0 / kernel_ms);	// target kernel time 50ms or 10ms
			uint32_t new_sstep = (uint32_t)( ((double)sd.sstep) * multi);
			if(new_sstep == 0) new_sstep=1;
			sd.sstep = new_sstep;
		}

		// setup residue for nmin! mod P
		for(; sstart < smcount; sstart += sd.sstep){
			smax = sstart + sd.sstep;
			if(smax > smcount)smax = smcount;
			sclSetKernelArg(pd.setup, 4, sizeof(uint32_t), &sstart);
			sclSetKernelArg(pd.setup, 5, sizeof(uint32_t), &smax);
			if(kernelq == 0){
				launchEvent = sclEnqueueKernelEvent(hardware, pd.setup);
			}
			else{
				sclEnqueueKernel(hardware, pd.setup);
			}
			if(++kernelq == 25){
				// limit cl queue depth and sleep cpu
				waitOnEvent(hardware, launchEvent);
				kernelq = 0;
			}
		}

		// profile iterate kernel once at program start.  adjust work size to target kernel runtime.
		if(first_iteration){
			first_iteration = false;
			nmax = nstart + sd.nstep;
			if(nmax > sd.nmax)nmax = sd.nmax;
			sclSetKernelArg(pd.iterate, 5, sizeof(uint32_t), &nstart);
			sclSetKernelArg(pd.iterate, 6, sizeof(uint32_t), &nmax);
			kernel_ms = ProfilesclEnqueueKernel(hardware, pd.iterate);
			nstart += sd.nstep;
			double multi = (sd.compute)?(50.0 / kernel_ms):(10.0 / kernel_ms);	// target kernel time 50ms or 10ms
			uint32_t new_nstep = (uint32_t)( ((double)sd.nstep) * multi);
			if(new_nstep == 0) new_nstep=1;
			sd.nstep = new_nstep;
			fprintf(stderr,"c: %u p: %u s: %u n: %u\n", sd.threadcount, pd.range, sd.sstep, sd.nstep);
		}

		// iterate from nmin! to nmax-1! mod P
		for(; nstart < sd.nmax; nstart += sd.nstep){
			nmax = nstart + sd.nstep;
			if(nmax > sd.nmax)nmax = sd.nmax;
			sclSetKernelArg(pd.iterate, 5, sizeof(uint32_t), &nstart);
			sclSetKernelArg(pd.iterate, 6, sizeof(uint32_t), &nmax);
			if(kernelq == 0){
				launchEvent = sclEnqueueKernelEvent(hardware, pd.iterate);
			}
			else{
				sclEnqueueKernel(hardware, pd.iterate);
			}
			if(++kernelq == 25){
				// limit cl queue depth and sleep cpu
				waitOnEvent(hardware, launchEvent);
				kernelq = 0;
			}
		}



		// checksum kernel
		sclEnqueueKernel(hardware, pd.check);

		uint64_t nextp = sd.p + pd.range;
		if(nextp < sd.p){
			// ck overflow at 2^64
			break;
		}
		else{
			sd.p = nextp;
		}

	}


	// final checkpoint
	if(kernelq > 0){
		waitOnEvent(hardware, launchEvent);
	}
	sleepCPU(hardware);
	boinc_begin_critical_section();
	sd.p = sd.pmax;
	boinc_fraction_done(1.0);
	if(boinc_is_standalone()) printf("Sieve Progress: %.1f%%\n",100.0);
	getResults(pd, sd, hardware);
	checkpoint(sd);

	// print checksum
	FILE * resfile = my_fopen(RESULTS_FILENAME,"a");

	if( resfile == NULL ){
		fprintf(stderr,"Cannot open %s !!!\n",RESULTS_FILENAME);
		exit(EXIT_FAILURE);
	}

	if(sd.factorcount == 0){
		if( fprintf( resfile, "no factors\n%016" PRIX64 "\n", sd.checksum ) < 0 ){
			fprintf(stderr,"Cannot write to %s !!!\n",RESULTS_FILENAME);
			exit(EXIT_FAILURE);
		}
	}
	else{
		if( fprintf( resfile, "%016" PRIX64 "\n", sd.checksum ) < 0 ){
			fprintf(stderr,"Cannot write to %s !!!\n",RESULTS_FILENAME);
			exit(EXIT_FAILURE);
		}
	}

	fclose(resfile);

	boinc_end_critical_section();


	fprintf(stderr,"Search complete.\nfactors %" PRIu64 ", prime count %" PRIu64 "\n", sd.factorcount, sd.primecount);

	if(boinc_is_standalone()){
		time(&totalf);
		printf("Search finished in %d sec.\n", (int)totalf - (int)totals);
		printf("factors %" PRIu64 ", prime count %" PRIu64 ", checksum %016" PRIX64 "\n", sd.factorcount, sd.primecount, sd.checksum);
	}


	cleanup(pd);

}


void run_test( sclHard hardware, searchData & sd ){

	int goodtest = 0;

	printf("Beginning self test of 4 ranges.\n");

	time_t st, fin;
	time(&st);

//	-p 100e6 -P 101e6 -n 1e6 -N 2e6
	sd.pmin = 100000000;
	sd.pmax = 101000000;
	sd.nmin = 1000000;
	sd.nmax = 2000000;
	cl_sieve( hardware, sd );
	if( sd.factorcount == 1071 && sd.primecount == 54211 && sd.checksum == 0x000004F4F744CA97 ){
		printf("test case 1 passed.\n\n");
		fprintf(stderr,"test case 1 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 1 failed.\n\n");
		fprintf(stderr,"test case 1 failed.\n");
	}
	sd.checksum = 0;
	sd.primecount = 0;
	sd.factorcount = 0;

//	-p 1e12 -P 100001e7 -n 10000 -N 2e6
	sd.pmin = 1000000000000;
	sd.pmax = 1000010000000;
	sd.nmin = 10000;
	sd.nmax = 2000000;
	cl_sieve( hardware, sd );
	if( sd.factorcount == 3 && sd.primecount == 361727 && sd.checksum == 0x05052A2D65F3D735 ){
		printf("test case 2 passed.\n\n");
		fprintf(stderr,"test case 2 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 2 failed.\n\n");
		fprintf(stderr,"test case 2 failed.\n");
	}
	sd.checksum = 0;
	sd.primecount = 0;
	sd.factorcount = 0;

//	-p 127 -P 100000 -n 127 -N 1e6
	sd.pmin = 127;
	sd.pmax = 100000;
	sd.nmin = 127;
	sd.nmax = 1000000;
	cl_sieve( hardware, sd );
	if( sd.factorcount == 42770 && sd.primecount == 9566 && sd.checksum == 0x0000000065C074F0 ){
		printf("test case 3 passed.\n\n");
		fprintf(stderr,"test case 3 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 3 failed.\n\n");
		fprintf(stderr,"test case 3 failed.\n");
	}
	sd.checksum = 0;
	sd.primecount = 0;
	sd.factorcount = 0;

//	-p 2e12 -P 200001e7 -n 2e6 -N 6e6
	sd.pmin = 2000000000000;
	sd.pmax = 2000010000000;
	sd.nmin = 2000000;
	sd.nmax = 6000000;
	cl_sieve( hardware, sd );
	if( sd.factorcount == 3 && sd.primecount == 352866 && sd.checksum == 0x09C9742A908451EB ){
		printf("test case 4 passed.\n\n");
		fprintf(stderr,"test case 4 passed.\n");
		++goodtest;
	}
	else{
		printf("test case 4 failed.\n\n");
		fprintf(stderr,"test case 4 failed.\n");
	}
	sd.checksum = 0;
	sd.primecount = 0;
	sd.factorcount = 0;


//	done
	if(goodtest == 4){
		printf("All test cases completed successfully!\n");
		fprintf(stderr, "All test cases completed successfully!\n");
	}
	else{
		printf("Self test FAILED!\n");
		fprintf(stderr, "Self test FAILED!\n");
	}

	time(&fin);
	printf("Elapsed time: %d sec.\n", (int)fin - (int)st);

}


