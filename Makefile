export FUTHARK_OPENCL=bin/futhark-opencl
export OPENCL_PLATFORM?=NVIDIA

CUB_PATH?=$(HOME)/cub-1.7.0

.SECONDARY:

all: bin/futhark-opencl benchmarks/inputs sum_results mss_results index_of_max_results blackscholes_results results/sum_segmented_cub.json results/index_of_max_segmented_cub.json results/blackscholes_segmented_cub.json

# For sum we want to try three different workgroup sizes; the others
# will have to make do with 512.
sum_results: bin/futhark-opencl benchmarks/sum_expected
	./mkbench.sh sum 128 512 1024

%_results: bin/futhark-opencl benchmarks/%_expected
	./mkbench.sh $* 512

benchmarks/blackscholes_expected: bin/futhark-opencl
	./mkresults.sh blackscholes blackscholes || (rm -rf benchmarks/blackscholes_expected && false)

benchmarks/%_expected: bin/futhark-opencl
	./mkresults.sh $* i32 || (rm -rf benchmarks/$*_expected && false)

bin/futhark-opencl: futhark-patched
	mkdir -p bin
	cd futhark-patched && stack setup
	cd futhark-patched && stack build
	cp `cd futhark-patched && stack exec which futhark-opencl` bin/futhark-opencl
	cp `cd futhark-patched && stack exec which futhark-c` bin/futhark-c
	cp `cd futhark-patched && stack exec which futhark` bin/futhark

futhark-patched:
	git clone https://github.com/HIPERFIT/futhark futhark-patched
	cd futhark-patched && git checkout f6049bf4b666847c4e9c46cccda7e4a72f39c492 && patch -p 1 -u < ../futhark-instrumentation.patch

benchmarks/inputs:
	mkdir -p benchmarks/inputs
	./mkdata.sh || rm -rf benchmarks/inputs

results/sum_segmented_cub.json: cub/sum_segmented cub.sh
	./cub.sh i32 sum_segmented

cub/sum_segmented: cub/sum_segmented.cu
	nvcc -o $@ $< -O3 -I$(CUB_PATH)

results/index_of_max_segmented_cub.json: cub/index_of_max_segmented cub.sh
	./cub.sh i32 index_of_max_segmented

cub/index_of_max_segmented: cub/index_of_max_segmented.cu
	nvcc -o $@ $< -O3 -I$(CUB_PATH)

results/blackscholes_segmented_cub.json: cub/blackscholes_segmented cub.sh
	./cub.sh blackscholes blackscholes_segmented

cub/blackscholes_segmented: cub/blackscholes_segmented.cu
	nvcc -o $@ $< -O3 -I$(CUB_PATH)

clean:
	rm -rf benchmarks/data results futhark-patched bin cub/sum_segmented
