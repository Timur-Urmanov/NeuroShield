#!/bin/bash
# H27: C with freeze and rule-switch branches at 400k/800k, and NI, on seeds 40-47 (notes/hypotheses.md)
cd "$(dirname "$0")"
export H26_FORKS=400000,800000 H26_SWITCH=1 H26_DIR=h27 OMP_NUM_THREADS=1
for s in 40 41 42 43 44 45 46 47; do echo "C $s"; done | xargs -P 2 -L 1 python3 h26_freeze.py >> ../.cache/h27.log 2>&1
for s in 40 41 42 43 44 45 46 47; do echo "NI $s"; done | xargs -P 4 -L 1 python3 h26_freeze.py >> ../.cache/h27.log 2>&1
echo "ALL DONE" >> ../.cache/h27.log
