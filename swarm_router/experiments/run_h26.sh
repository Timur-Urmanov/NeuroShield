#!/bin/bash
cd "$(dirname "$0")"
for s in 30 31 32 33 34 35 36 37; do echo "C $s"; echo "NI $s"; done | \
  OMP_NUM_THREADS=1 xargs -P 3 -L 1 python3 h26_freeze.py >> ../.cache/h26.log 2>&1
echo "ALL DONE" >> ../.cache/h26.log
