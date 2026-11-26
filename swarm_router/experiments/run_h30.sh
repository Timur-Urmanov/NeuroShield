#!/bin/bash
# H29r: repaired random search RC in three environments; H30: NI with freeze / true-type / both branches at 800k
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1
{ for s in 60 61 62 63 64 65 66 67; do echo "env H26_DIR=h30 H26_BASE=NI H26_FORKS=800000 H26_KINDS=FOG python3 h26_freeze.py NI $s"; done
  for s in 20 21 22 23 24 25 26 27 28 29; do echo "env H26_FORKS= H26_DIR=h29 python3 h26_freeze.py RC $s"; done
  for cfg in "b24 H26_BUSY=24" "l7 H26_L=7"; do set -- $cfg
    for s in 50 51 52 53 54; do echo "env H26_FORKS= H26_DIR=h28_$1 $2 python3 h26_freeze.py RC $s"; done; done
} | xargs -P 3 -I{} sh -c "{}" >> ../.cache/h30.log 2>&1
echo "ALL DONE" >> ../.cache/h30.log
