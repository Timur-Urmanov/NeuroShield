#!/bin/bash
# H32 stage 1, round 2: niching of parents and configuration-based substitution (dev seeds 100-102, T = 400k)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 H26_FORKS= H26_T=400000 H26_SHIFT=100000
{ echo "env H26_DIR=h32dev/repro H32_PARAMS=3,3,0,300,lcb python3 h26_freeze.py ES 100"
  for q in flat equal; do for u in parent config; do for e in lcb mean; do for n in 1 3; do for s in 100 101 102; do
    echo "env H26_DIR=h32dev/r2_${q}_${u}_${e}_${n} H32_PARAMS=$n,3,0.1,300,$e,$q,$u python3 h26_freeze.py ES $s"; done; done; done; done; done
} | xargs -P 4 -I{} sh -c "{}" >> ../.cache/h32dev2.log 2>&1
echo "ALL DONE" >> ../.cache/h32dev2.log
