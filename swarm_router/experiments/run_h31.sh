#!/bin/bash
# H31: who is evicted under the old and the new rule, on the same states (C and NI trajectories, seeds 70-74)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 H26_FORKS= H26_VICT=1 H26_DIR=h31
for s in 70 71 72 73 74; do echo "C $s"; echo "NI $s"; done | xargs -P 4 -L 1 python3 h26_freeze.py >> ../.cache/h31.log 2>&1
echo "ALL DONE" >> ../.cache/h31.log
