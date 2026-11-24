#!/usr/bin/env bash
# H21 driver (detached; resumes from .cache/h20_T200000_genes_L5_genes_pow1.jsonl)
set -u
cd "$(dirname "$0")"
OMP_NUM_THREADS=1 python3 h20_cycle.py T=200000 genes_L=5 genes_pow=1 >> ../.cache/h21.log 2>&1
echo "ALL DONE" >>> ../.cache/h21.log
