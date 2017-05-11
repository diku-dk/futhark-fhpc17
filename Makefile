FUTHARK_OPENCL=bin/futhark-opencl

all: results/segsum_map_with_loop.json results/segsum_large.json results/segsum_small.json results/segsum_automatic.json

bin/futhark-opencl: futhark-patched
	cd futhark-patched && stack setup
	cd futhark-patched && stack build
	cp `cd futhark-patched && stack exec which futhark-opencl` $@

futhark-patched:
	git clone --depth 1 https://github.com/HIPERFIT/futhark futhark-patched
	cd futhark-patched && patch -p 1 -u < ../futhark-instrumentation.patch

benchmarks/data:
	mkdir -p benchmarks/data
	./mkdata.sh || rm -rf benchmarks/data

results/segsum_map_with_loop.json:
	mkdir -p results
	FUTHARK_MAP_WITH_LOOP=1 futhark-bench --compiler=$(FUTHARK_OPENCL) benchmarks/segsum.fut --json $@

results/reduce.json:
	mkdir -p results
	futhark-bench --compiler=$(FUTHARK_OPENCL) benchmarks/reduce.fut --json $@

results/segsum_automatic.json:
	mkdir -p results
	FUTHARK_VERSIONED_CODE=1 futhark-bench --compiler=$(FUTHARK_OPENCL) benchmarks/segsum.fut --json $@

results/segsum_as_scan.json:
	mkdir -p results
	futhark-bench --compiler=$(FUTHARK_OPENCL) benchmarks/segsum-as-scan.fut --json $@

# The large and small kernel variants are not expected to work for all
# data sets, so we add a '|| true' to futhark-bench.
results/segsum_large.json:
	mkdir -p results
	FUTHARK_LARGE_KERNEL=1 futhark-bench --compiler=$(FUTHARK_OPENCL) benchmarks/segsum.fut --json $@ || true

results/segsum_small.json:
	mkdir -p results
	FUTHARK_SMALL_KERNEL=1 futhark-bench --compiler=$(FUTHARK_OPENCL) benchmarks/segsum.fut --json $@ || true

clean:
	rm -rf benchmarks/data
