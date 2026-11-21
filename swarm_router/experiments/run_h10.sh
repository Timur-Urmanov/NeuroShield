#!/usr/bin/env bash
# H10 driver: main grid, then the long-horizon check. Both resume from .cache/h10_jobs*.jsonl.
set -u
cd "$(dirname "$0")/.."
LOG=.cache/h10.log
python3 experiments/h10_scale.py --workers 4 >> "$LOG" 2>&1 && echo "MAIN DONE" >> "$LOG"
python3 experiments/h10_scale.py --workers 4 --long >> "$LOG" 2>&1 && echo "LONG DONE" >> "$LOG"
echo "ALL DONE" >> "$LOG"
