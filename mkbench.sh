#!/bin/sh

set -e
set -x

benchmark=$1
shift
group_sizes=$*

OPENCL_PLATFORM=${OPENCL_PLATFORM:-NVIDIA}
timeout=45 # seconds

echo "Benchmarking $1"

mkdir -p results

for group_size in $group_sizes; do
    echo "With group size $group_size:"

    futhark_bench="futhark-bench --pass-option=-p${OPENCL_PLATFORM} --pass-option=--group-size=${group_size} --compiler=${FUTHARK_OPENCL} --timeout=$timeout"

    json() {
        echo "results/${benchmark}${1}_groupsize_${group_size}.json"
    }

    echo "Non-segmented baseline:"
    ${futhark_bench} benchmarks/${benchmark}.fut --json $(json)

    echo "With segmented scan:"
    ${futhark_bench} benchmarks/${benchmark}_segmented_scan.fut --json $(json _segmented_scan)

    # Some of the forced variants may time out or fail, so we add a || true.

    echo "With sequential loop:"
    FUTHARK_MAP_WITH_LOOP=1 ${futhark_bench} benchmarks/${benchmark}_segmented.fut --json $(json _segmented_map_with_loop) || true

    echo "With large-segments kernel:"
    FUTHARK_LARGE_KERNEL=1 ${futhark_bench} benchmarks/${benchmark}_segmented.fut --json $(json _segmented_large) || true

    echo "With small-segments kernel:"
    FUTHARK_SMALL_KERNEL=1 ${futhark_bench} benchmarks/${benchmark}_segmented.fut --json $(json _segmented_small) || true

    echo "Automatic:"
    FUTHARK_VERSIONED_CODE=1 ${futhark_bench} benchmarks/${benchmark}_segmented.fut --json $(json _segmented_auto)
done
