#!/bin/bash
# H33 tuning on dev seeds 100-102: T = 1.2M, targets of 3 niches redrawn at 700k (notes/hypotheses.md)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 H26_T=1200000 H26_SHIFT=100000 H33_TSHIFT=700000,3
{ for s in 100 101 102; do echo "env H26_DIR=h33dev/ref H26_BASE=NI H26_FORKS=500000 H26_KINDS=F python3 h26_freeze.py NI $s"; echo "env H26_DIR=h33dev/ref H26_FORKS= python3 h26_freeze.py N4 $s"; done
  for e in 0.005 0.01; do for h in 0.02 0.1; do for s in 100 101 102; do echo "env H26_DIR=h33dev/g_${e}_${h} H26_FORKS= H33_CTL=g,$e,$h python3 h26_freeze.py NI $s"; done; done; done
  for z in 1 2; do for g in 0.98 0.995; do for s in 100 101 102; do echo "env H26_DIR=h33dev/z_${z}_${g} H26_FORKS= H33_CTL=z,$z,$g python3 h26_freeze.py NI $s"; done; done; done
} | xargs -P 4 -I{} sh -c "{}" >> ../.cache/h33dev.log 2>&1
echo "ALL DONE" >> ../.cache/h33dev.log
