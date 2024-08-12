# FactSieve

FactSieve by Bryan Little

A BOINC enabled OpenCL standalone sieve for factors of factorial prime candidates of the form n!+-1

Using OpenMP for multithreaded factor verification on CPU.

With contributions by
* Yves Gallot
* Mark Rodenkirch
* Kim Walisch

## Requirements

* OpenCL v1.1
* 64 bit operating system

## How it works

1. Search parameters are given on the command line.
2. A small group of sieve primes are generated on the GPU.
3. The group of primes are tested for factors in the N range specified.
4. Repeat #2-3 until checkpoint.  Gather factors and checksum data from GPU.
5. Report any factors in factors.txt, along with a checksum at the end.
6. Checksum can be used to compare results in a BOINC quorum.

## Running the program
```
Note that when p <= n there are no factors for numbers of the form n!+-1
All p <= n are factors of n!
When there are less than 100 factors at checkpoint they will be verified on the CPU.
An optional argument -v # can be used to verify all factors on CPU using the specified number of threads #.

command line options
* -n #		Start factorial n!+-1
* -N #		End factorial N!+-1, range [-n, -N) exclusive, 127 <= -n <= n < -N < 2^31
* -p #		Starting prime factor p
* -P #		End prime factor P, range [-p, -P) exclusive, 127 <= -n <= -p <= p < -P < 2^64
* -s		Perform self test to verify proper operation of the program
* -v #		Verify all factors on CPU using # threads.  This can be slow with a large number of factors.
* -h		Print help

Program gets the OpenCL GPU device index from BOINC.  To run stand-alone, the program will
default to GPU 0 unless an init_data.xml is in the same directory with the format:

<app_init_data>
<gpu_type>NVIDIA</gpu_type>
<gpu_device_num>0</gpu_device_num>
</app_init_data>

or

<app_init_data>
<gpu_type>ATI</gpu_type>
<gpu_device_num>0</gpu_device_num>
</app_init_data>
```

## Related Links
* [Yves Gallot on GitHub](https://github.com/galloty)
* [primesieve by Kim Walisch](https://github.com/kimwalisch/primesieve)
* [Mark Rodenkirch on SourceForge](https://sourceforge.net/projects/mtsieve/)
