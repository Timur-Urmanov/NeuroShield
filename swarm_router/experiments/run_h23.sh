#!/bin/bash
cd "$(dirname "$0")"
OMP_NUM_THREADS=1 python3 h23_colony.py T=200000 genes_L=5 genes_pow=1 nov=0.5,2,8 >> ../.cache/h23.log 2>&1
echo "ALL DONE" >> ../.cache/h23.log
