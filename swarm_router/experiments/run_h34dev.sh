#!/bin/bash
# H34 dev (seeds 100-102, T = 1.2M, hard shift at 700k): choice of the search generator by variant B, then a viability
# check of the hybrids and the factor control with the chosen generator (second call: run_h34dev.sh XO,PM)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 H26_T=1200000 H26_SHIFT=100000 H33_TSHIFT=700000,3 H26_FORKS= H32_PARAMS=1,3,0.1,300,crate,equal,config,1
if [ -z "$1" ]; then
  for g in 0,0.1 0.3,0.1 0.3,0.3; do for s in 100 101 102; do echo "env H26_DIR=h34dev/B_$g H34_GEN=$g python3 h26_freeze.py ES $s"; done; done
else
  for s in 100 101 102; do for v in HC HD EC; do echo "env H26_DIR=h34dev/chosen H34_GEN=$1 python3 h26_freeze.py $v $s"; done; done
fi | xargs -P 4 -I{} sh -c "{}" >> ../.cache/h34dev.log 2>&1
echo "ALL DONE $1" >> ../.cache/h34dev.log
