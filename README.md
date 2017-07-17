Benchmark Suite for the paper "Strategies for Regular Segmented Reductions on GPU"
==

This tooling was used to perform measurements for the micro-benchmarks
for the paper.  The Rodinia benchmarks were measured manually.

Requirements
--

The tooling will automatically install (a patched version of) the
Futhark compiler, assuming you have [the Haskell Tool
Stack](https://docs.haskellstack.org/) installed.

You must have a working CUDA and OpenCL installation.  By default,
only an OpenCL platform containing the string `NVIDIA` will be used.
This can be changed by setting the environment variables
`OPENCL_PLATFORM`, or editing the `Makefile`.

Usage
--

Assuming the requirements are fulfilled, just run `make`.  The results
will be stored in JSON format in the `results` subdirectory.
