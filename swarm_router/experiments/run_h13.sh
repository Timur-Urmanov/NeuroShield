#!/usr/bin/env bash
# H13 test driver (detached; resumes from .cache/h13_test_T8000.jsonl)
set -u
cd "$(dirname "$0")/.."
python3 experiments/h13_team_selection.py --stage test --rho 0.02 --workers 2 >> .cache/h13.log 2>&1 && echo "TEST DONE" >> .cache/h13.log
echo "ALL DONE" >> .cache/h13.log
