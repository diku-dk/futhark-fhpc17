export FUTHARK_OPENCL=bin/futhark-opencl
export OPENCL_PLATFORM?=NVIDIA

.SECONDARY:

all: bin/futhark-opencl benchmarks/data sum_results mss_results index_of_max_results blackscholes_results

# For sum we want to try three different workgroup sizes; the others
# will have to make do with 512.
sum_results: bin/futhark-opencl benchmarks/%_expected
	./mkbench.sh $* 128 512 1024

%_results: bin/futhark-opencl benchmarks/%_expected
	./mkbench.sh $* 512

benchmarks/blackscholes_expected:
	./mkresults.sh blackscholes blackscholes || (rm -rf benchmarks/blackscholes_expected && false)

benchmarks/%_expected:
	./mkresults.sh $* i32 || (rm -rf benchmarks/$*_expected && false)

bin/futhark-opencl: futhark-patched
	mkdir -p bin
	cd futhark-patched && stack setup
	cd futhark-patched && stack build
	cp `cd futhark-patched && stack exec which futhark-opencl` bin/futhark-opencl
	cp `cd futhark-patched && stack exec which futhark` bin/futhark

futhark-patched:
	git clone --depth 1 https://github.com/HIPERFIT/futhark futhark-patched
	cd futhark-patched && patch -p 1 -u < ../futhark-instrumentation.patch

benchmarks/data:
	mkdir -p benchmarks/data
	./mkdata.sh || rm -rf benchmarks/data

clean:
	rm -rf benchmarks/data results futhark-patched bin
