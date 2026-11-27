#!/bin/bash
# H33 main test: controllers and controls branched from a common NI trajectory (notes/hypotheses.md)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 H26_BASE=NI H26_FORKS=400000,600000 H33_KMAP=400000:HAZQ,600000:F H33_SET=0.01,0.1,1,0.98
{ for s in 120 121 122 123 124 125; do
    echo "env H26_DIR=h33/stat python3 h26_freeze.py NI $s"
    echo "env H26_DIR=h33/hard H33_TSHIFT=800000,3 python3 h26_freeze.py NI $s"
    echo "env H26_DIR=h33/mild H33_TSHIFT=800000,3,1 python3 h26_freeze.py NI $s"; done
} | xargs -P 2 -I{} sh -c "{}" >> ../.cache/h33.log 2>&1
echo "ALL DONE" >> ../.cache/h33.log
