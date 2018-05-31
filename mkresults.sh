#!/bin/sh

set -e

benchmark=$1
data=$2

bin/futhark-c benchmarks/${benchmark}.fut
bin/futhark-c benchmarks/${benchmark}_segmented.fut

mkdir -p benchmarks/${benchmark}_expected

generate() {
    k=$1
    logsegments=0
    while [ $logsegments -le $k ]; do
        logsegsize=$(($k-$logsegments))
        infile="benchmarks/inputs/${data}_2pow${logsegments}_2pow${logsegsize}"
        outfile="benchmarks/${benchmark}_expected/${data}_2pow${logsegments}_2pow${logsegsize}"
        echo "Generating $outfile"
        segments=$(echo 2^$logsegments | bc)
        segsize=$(echo 2^$logsegsize | bc)
        cat "$infile" | benchmarks/${benchmark}_segmented -b > "$outfile"
        logsegments=$(($logsegments + 2))
    done

    infile="benchmarks/inputs/${data}_2pow${k}"
    outfile="benchmarks/${benchmark}_expected/${data}_2pow${k}"
    echo "Generating $outfile"
    cat "$infile" | benchmarks/${benchmark} -b > "$outfile"
}

generate 26

generate 18
