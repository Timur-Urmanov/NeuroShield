#!/usr/bin/env bash
# H17 driver (detached; resumes from .cache/h17.jsonl)
set -u
cd "$(dirname "$0")"
python3 h17_cost.py >> ../.cache/h17.log 2>&1
echo "ALL DONE" >> ../.cache/h17.log
