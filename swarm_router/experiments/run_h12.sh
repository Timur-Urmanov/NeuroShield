#!/usr/bin/env bash
# H12 test driver (detached; resumes from .cache/h12_test_T8000.jsonl)
set -u
cd "$(dirname "$0")/.."
python3 experiments/h12_inherit.py --stage test --rho 0.05 --workers 2 >> .cache/h12.log 2>&1 && echo "TEST DONE" >> .cache/h12.log
echo "ALL DONE" >> .cache/h12.log
