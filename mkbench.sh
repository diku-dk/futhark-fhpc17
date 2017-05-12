#!/bin/sh

set -e
set -x

benchmark=$1

OPENCL_PLATFORM=${OPENCL_PLATFORM:-NVIDIA}
group_sizes='128 512 1024'

echo "Benchmarking $1"

mkdir -p results

for group_size in $group_sizes; do
    echo "With group size $group_size:"

    futhark_bench="futhark-bench --pass-option=-p${OPENCL_PLATFORM} --pass-option=--group-size=${group_size} --compiler=${FUTHARK_OPENCL}"

    json() {
        echo "results/${benchmark}${1}_groupsize_${group_size}.json"
    }

    echo "Reduction baseline:"
    ${futhark_bench} benchmarks/${benchmark}.fut --json $(json)

    echo "With segmented scan:"
    ${futhark_bench} benchmarks/${benchmark}_segmented_scan.fut --json $(json _segmented_scan)

    echo "With sequential loop:"
    FUTHARK_MAP_WITH_LOOP=1 ${futhark_bench} benchmarks/${benchmark}_segmented.fut --json $(json _segmented_map_with_loop)

    # The large and small kernel variants are not expected to work for all
    # data sets, so we add a '|| true' to ${futhark_bench}.

    echo "With large-segments kernel:"
    FUTHARK_LARGE_KERNEL=1 ${futhark_bench} benchmarks/${benchmark}_segmented.fut --json $(json _segmented_large) || true

    echo "With small-segments kernel:"
    FUTHARK_SMALL_KERNEL=1 ${futhark_bench} benchmarks/${benchmark}_segmented.fut --json $(json _segmented_small) || true

    echo "Automatic:"
    FUTHARK_VERSIONED_CODE=1 ${futhark_bench} benchmarks/${benchmark}_segmented.fut --json $(json _segmented_auto)
done
