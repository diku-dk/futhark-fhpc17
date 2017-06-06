#!/bin/sh

set -e

i32bounds="-5:5"

generate() {
    k=$1
    logsegments=0
    while [ $logsegments -le $k ]; do
        logsegsize=$(($k-$logsegments))
        file="benchmarks/inputs/i32_2pow${logsegments}_2pow${logsegsize}"
        echo "Generating $file"
        segments=$(echo 2^$logsegments | bc)
        segsize=$(echo 2^$logsegsize | bc)
        futhark-dataset --binary --i32-bounds="$i32bounds" -g "[$segments][$segsize]i32" > "$file"
        logsegments=$(($logsegments + 2))
    done

    file="benchmarks/inputs/i32_2pow${k}"
    echo "Generating $file"
    n=$(echo 2^$k | bc)
    futhark-dataset --binary --i32-bounds="$i32bounds" -g "[$n]i32" > "$file"
}

generate 26

generate 18

blackscholes() {
    k=$1
    logsegments=0
    while [ $logsegments -le $k ]; do
        logsegsize=$(($k-$logsegments))
        file="benchmarks/inputs/blackscholes_2pow${logsegments}_2pow${logsegsize}"
        echo "Generating $file"
        segments=$(echo 2^$logsegments | bc)
        segsize=$(echo 2^$logsegsize | bc)
        (futhark-dataset --binary -g "[$segments]f64" -g "[$segments]f64"; echo "$segsize") > "$file"
        logsegments=$(($logsegments + 2))
    done

    file="benchmarks/inputs/blackscholes_2pow${k}"
    echo "Generating $file"
    n=$(echo 2^$k | bc)
    (futhark-dataset --binary -g f64 -g f64 --i32-bounds "$n:$n" -g i32) > "$file"
}

blackscholes 26

blackscholes 18
