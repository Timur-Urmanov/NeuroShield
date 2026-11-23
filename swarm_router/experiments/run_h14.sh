#!/usr/bin/env bash
# H14 driver (detached; resumes from .cache/h14_T8000.jsonl)
set -u
cd "$(dirname "$0")/.."
cd experiments && python3 h14_recruit.py --workers 2 >> ../.cache/h14.log 2>&1 && echo "TEST DONE" >> ../.cache/h14.log
echo "ALL DONE" >> ../.cache/h14.log
