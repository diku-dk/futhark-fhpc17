#!/bin/sh

set -e

generate() {
    k=$1
    logsegments=0
    while [ $logsegments -le $k ]; do
        logsegsize=$(($k-$logsegments))
        file="benchmarks/data/i32_2pow${logsegments}_2pow${logsegsize}"
        echo "Generating $file"
        segments=$((2**$logsegments))
        segsize=$((2**$logsegsize))
        futhark-dataset --binary -g "[$segments][$segsize]i32" > "$file"
        logsegments=$(($logsegments + 2))
    done

    file="benchmarks/data/i32_2pow${k}"
    echo "Generating $file"
    n=$((2**$k))
    futhark-dataset --binary -g "[$n]i32" > "$file"
}

generate 26

generate 18

blackscholes() {
    k=$1
    logsegments=0
    while [ $logsegments -le $k ]; do
        logsegsize=$(($k-$logsegments))
        file="benchmarks/data/blackscholes_2pow${logsegments}_2pow${logsegsize}"
        echo "Generating $file"
        segments=$((2**$logsegments))
        segsize=$((2**$logsegsize))
        (futhark-dataset --binary -g "[$segments]f32" -g "[$segments]f32"; echo "$segsize") > "$file"
        logsegments=$(($logsegments + 2))
    done
}

blackscholes 26

blackscholes 18
