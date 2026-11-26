#!/bin/bash
# H29: NI and random search in the original environment (seeds 20-29, paired with H25), random search and
# pattern+novelty in two H28 environments (seeds 50-54) (notes/hypotheses.md)
cd "$(dirname "$0")"
export H26_FORKS= OMP_NUM_THREADS=1
{ for s in 20 21 22 23 24 25 26 27 28 29; do for v in NI R; do echo "env H26_DIR=h29 python3 h26_freeze.py $v $s"; done; done
  for cfg in "b24 H26_BUSY=24" "l7 H26_L=7"; do set -- $cfg
    for s in 50 51 52 53 54; do for v in R PN; do echo "env H26_DIR=h28_$1 $2 python3 h26_freeze.py $v $s"; done; done; done
} | xargs -P 4 -I{} sh -c "{}" >> ../.cache/h29.log 2>&1
echo "ALL DONE" >> ../.cache/h29.log
