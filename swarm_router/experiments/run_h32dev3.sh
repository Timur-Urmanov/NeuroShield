#!/bin/bash
# H32 stage 1, round 3: cluster-relative own rate as the eviction score of ES (dev seeds 100-102, T = 400k)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 H26_FORKS= H26_T=400000 H26_SHIFT=100000
{ for n in 1 3; do for k in 1 4; do for s in 100 101 102; do
    echo "env H26_DIR=h32dev/r3_${n}_${k} H32_PARAMS=$n,3,0.1,300,crate,equal,config,$k python3 h26_freeze.py ES $s"; done; done; done
  for s in 100 101 102; do echo "env H26_DIR=h32dev/nt3 H32_PARAMS=3,3,0.1,300,crate,equal,config,1 python3 h26_freeze.py NT $s"; done
} | xargs -P 4 -I{} sh -c "{}" >> ../.cache/h32dev4.log 2>&1
echo "ALL DONE" >> ../.cache/h32dev4.log
