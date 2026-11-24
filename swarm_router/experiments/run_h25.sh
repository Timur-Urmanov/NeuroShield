#!/bin/bash
cd "$(dirname "$0")"
until grep -q "ALL DONE" ../.cache/h24.log; do sleep 30; done
OMP_NUM_THREADS=1 python3 h25_long.py >> ../.cache/h25.log 2>&1
echo "ALL DONE" >> ../.cache/h25.log
