#!/usr/bin/env bash
# H16 addendum driver (detached): waits for the H16 run, then context-aware bandits (resumes from .cache/h16b.jsonl)
set -u
cd "$(dirname "$0")"
while ! grep -q "ALL DONE" ../.cache/h16.log; do sleep 30; done
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python3 h16b_ctx.py >> ../.cache/h16b.log 2>&1
echo "ALL DONE" >> ../.cache/h16b.log
