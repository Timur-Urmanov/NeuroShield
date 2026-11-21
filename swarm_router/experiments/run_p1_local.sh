#!/usr/bin/env bash
# P1-local driver (notes/hypotheses.md): one invocation per seed (progress survives restarts through the
# response cache), then one aggregate run over all seeds served from the cache -> results/llm_role_spread_real.json
set -u
cd "$(dirname "$0")/.."
LOG=${LOG:-.cache/p1_local.log}
ARGS="--config configs/llm.local.yaml --real --yes --budget-usd 200 --models qwen05,qwen15,qwen3 --executor-model qwen05 --n-tasks 20"
for sd in 0 1 2 3 4; do
  echo "$(date -u +%FT%T) start seed $sd" >> "$LOG"
  python3 experiments/llm_role_spread.py $ARGS --seed0 $sd --seeds 1 --out .cache/p1_chunks >> "$LOG" 2>&1
  echo "$(date -u +%FT%T) done seed $sd" >> "$LOG"
done
python3 experiments/llm_role_spread.py $ARGS --seed0 0 --seeds 5 >> "$LOG" 2>&1
echo "$(date -u +%FT%T) ALL DONE" >> "$LOG"
