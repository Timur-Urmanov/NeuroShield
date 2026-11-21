#!/usr/bin/env bash
# P3-local driver (notes/hypotheses.md). Re-running resumes from the response cache.
set -u
cd "$(dirname "$0")/.."
LOG=${LOG:-.cache/p3_local.log}
echo "$(date -u +%FT%T) start" >> "$LOG"
python3 experiments/llm_run.py --config configs/llm.p3.yaml --real --yes --budget-usd 500 \
  --n-tasks 300 --seeds 5 --variants swarm,swarm_solo,linucb,single,pipeline --pipeline-n 100 \
  --name p3_local >> "$LOG" 2>&1
echo "$(date -u +%FT%T) ALL DONE rc=$?" >> "$LOG"
