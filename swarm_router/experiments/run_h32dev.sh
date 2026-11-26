#!/bin/bash
# H32 stage 1: tuning grid of the ES baseline on dev seeds 100-102, T = 400k (notes/hypotheses.md)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 H26_FORKS= H26_T=400000 H26_SHIFT=100000
{ for s in 100 101 102; do for v in NI A; do echo "env H26_DIR=h32dev/ref python3 h26_freeze.py $v $s"; done; done
  for n in 1 2 3; do for m in 1 3; do for e in lcb mean; do for r in 0 0.1; do for s in 100 101 102; do
    echo "env H26_DIR=h32dev/${n}_${m}_${e}_${r} H32_PARAMS=$n,$m,$r,300,$e python3 h26_freeze.py ES $s"; done; done; done; done; done
} | xargs -P 4 -I{} sh -c "{}" >> ../.cache/h32dev.log 2>&1
echo "ALL DONE" >> ../.cache/h32dev.log
