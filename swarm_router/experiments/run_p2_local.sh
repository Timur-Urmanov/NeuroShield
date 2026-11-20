#!/usr/bin/env bash
# P2-local driver (notes/hypotheses.md): one invocation per (model, seed) so that progress survives
# restarts (responses are cached in .cache/llm), then one final aggregate run over all seeds that is
# served from the cache and writes results/llm_context_degradation_real.json.
set -u
cd "$(dirname "$0")/.."
LOG=${LOG:-.cache/p2_local.log}
ARGS="--config configs/llm.local.yaml --real --yes --budget-usd 50 --k 0,8,32,64 --n-tasks 30"
for m in qwen15 qwen05; do
  for sd in 0 1 2 3 4; do
    echo "$(date -u +%FT%T) start $m seed $sd" >> "$LOG"
    python3 experiments/llm_context_degradation.py $ARGS --models $m --seed0 $sd --seeds 1 \
      --out .cache/p2_chunks >> "$LOG" 2>&1
    echo "$(date -u +%FT%T) done $m seed $sd" >> "$LOG"
  done
done
python3 experiments/llm_context_degradation.py $ARGS --models qwen15,qwen05 --seed0 0 --seeds 5 >> "$LOG" 2>&1
echo "$(date -u +%FT%T) ALL DONE" >> "$LOG"
