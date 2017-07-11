#!/bin/sh
#
# This is a huge hack, but I don't know of any quick and easy way to
# generate JSON in a shell script (thi is a sentence I hoped I would
# never have cause to write).

set -e

benchmark=$1

generate() {
    k=$1
    logsegments=0
    while [ $logsegments -le $k ]; do
        logsegsize=$(($k-$logsegments))
        echo -n "\"inputs/i32_2pow${logsegments}_2pow${logsegsize}\":\""
        cub/$benchmark $logsegments $logsegsize || echo "Failed" >&2
        echo "\","
        logsegments=$(($logsegments + 2))
    done
}

exec > results/${benchmark}_cub.json

echo '{'

generate 26
generate 18

echo '"dummy":0'

echo '}'
