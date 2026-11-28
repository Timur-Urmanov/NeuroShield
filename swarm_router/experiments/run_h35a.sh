#!/bin/bash
# H35 diagnostic stage (A: recovery of routing after controlled perturbations of a frozen mature colony;
# B: lifetimes and turnover on the NI trajectory). Seeds 140-145, T = 1.2M, fork at 800k.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 H26_T=1200000 H26_SHIFT=100000 H26_BASE=NI H26_FORKS= H35_FORK=800000 H35_POP=1 H26_DIR=h35a
export H35_KINDS=C0,R05,R20,R50,G05,G20,G50,S05,S20,S50
for s in 140 141 142 143 144 145; do echo "NI $s"; done | xargs -P 3 -L 1 python3 h26_freeze.py >> ../.cache/h35a.log 2>&1
echo "ALL DONE" >> ../.cache/h35a.log
