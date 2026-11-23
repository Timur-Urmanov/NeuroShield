#!/usr/bin/env bash
# H15 driver (detached; resumes from .cache/h15_T8000.jsonl)
set -u
cd "$(dirname "$0")"
python3 h15_capacity.py --workers 2 >> ../.cache/h15.log 2>&1 && echo "TEST DONE" >> ../.cache/h15.log
echo "ALL DONE" >> ../.cache/h15.log
