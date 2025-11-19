#!/bin/bash

M=2048
K=2048
N=2048
BLUST_BENCH=BlustBench
VERSION=$(git rev-parse --short HEAD)
TYPE=blust

function usage() {
    echo "Usage: $0 [blust|baseline] [M K N]"
    echo "If M, K, N are not provided, defaults to $M $K $N"
}

function setup_params() {
    if [ $# -eq 3 ]; then
        M=$1
        K=$2
        N=$3
        TYPE=blust
    elif [ $# -eq 4 ]; then
        M=$2
        K=$3
        N=$4
        TYPE=$1
        if [ "$TYPE" != "blust" ] && [ "$TYPE" != "baseline" ]; then
            echo "Invalid type: $TYPE"
            usage
            exit 1
        fi
    elif [ $# -ne 0 ]; then
        usage
        exit 1
    fi
}

setup_params "$@"
./build/bin/BlustBench "$TYPE" "$M" "$K" "$N" > ./bench/results/bench_results_${VERSION}.csv
