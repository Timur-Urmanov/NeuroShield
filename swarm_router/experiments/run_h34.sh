#!/bin/bash
# H34 main test: A = NI, B = ES, C = HC (hybrid), D = HD (hybrid, mixed generation), T = EC; three environments, seeds 130-137
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 H26_FORKS= H32_PARAMS=1,3,0.1,300,crate,equal,config,1 H34_GEN=0,0.1
{ for s in 130 131 132 133 134 135 136 137; do for v in NI ES HC HD EC; do
    echo "env H26_DIR=h34/stat python3 h26_freeze.py $v $s"
    echo "env H26_DIR=h34/mild H33_TSHIFT=800000,3,1 python3 h26_freeze.py $v $s"
    echo "env H26_DIR=h34/hard H33_TSHIFT=800000,3 python3 h26_freeze.py $v $s"; done; done
} | xargs -P 4 -I{} sh -c "{}" >> ../.cache/h34.log 2>&1
echo "ALL DONE" >> ../.cache/h34.log
