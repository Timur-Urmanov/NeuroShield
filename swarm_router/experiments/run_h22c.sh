#!/bin/bash
cd "$(dirname "$0")"
OMP_NUM_THREADS=1 python3 h22_confirm.py T=200000 genes_L=5 genes_pow=1 nov=0.5,2,8 >> ../.cache/h22c.log 2>&1
echo "ALL DONE" >> ../.cache/h22c.log
