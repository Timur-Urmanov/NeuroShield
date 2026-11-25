#!/bin/bash
# H28: A, C (contrib), NI in four other environments, seeds 50-54 (notes/hypotheses.md)
cd "$(dirname "$0")"
export H26_FORKS= OMP_NUM_THREADS=1
for cfg in "b24 H26_BUSY=24" "b96 H26_BUSY=96" "n320 H26_N=320" "l7 H26_L=7"; do
  set -- $cfg
  for s in 50 51 52 53 54; do for v in C NI A; do echo "env H26_DIR=h28_$1 $2 python3 h26_freeze.py $v $s"; done; done
done | xargs -P 4 -I{} sh -c "{}" >> ../.cache/h28.log 2>&1
echo "ALL DONE" >> ../.cache/h28.log
