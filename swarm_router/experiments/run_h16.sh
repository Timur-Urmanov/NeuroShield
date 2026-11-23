#!/usr/bin/env bash
# H16 driver (detached; resumes from .cache/h16_T8000.jsonl)
set -u
cd "$(dirname "$0")"
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 h16_shift.py --workers 4 >> ../.cache/h16.log 2>&1 && echo "TEST DONE" >> ../.cache/h16.log
echo "ALL DONE" >> ../.cache/h16.log
