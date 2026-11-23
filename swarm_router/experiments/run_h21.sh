#!/usr/bin/env bash
# H21 driver (detached; no checkpoint: ~1 h)
set -u
cd "$(dirname "$0")"
OMP_NUM_THREADS=1 python3 h20_cycle.py T=200000 genes_L=5 genes_pow=1 > ../.cache/h21.log 2>&1
echo "ALL DONE" >> ../.cache/h21.log
