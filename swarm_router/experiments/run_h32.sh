#!/bin/bash
# H32 stage 2: NI, ES, ES with the colony's eviction (EC), NI with trials (NT); frozen trial parameters (notes/hypotheses.md)
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 H26_FORKS= H32_PARAMS=1,3,0.1,300,crate,equal,config,1
{ for s in 80 81 82 83 84 85 86 87; do for v in NI ES EC NT; do echo "env H26_DIR=h32/orig python3 h26_freeze.py $v $s"; done; done
  for cfg in "b24 H26_BUSY=24" "l7 H26_L=7"; do set -- $cfg
    for s in 80 81 82 83 84; do for v in NI ES EC NT; do echo "env H26_DIR=h32/$1 $2 python3 h26_freeze.py $v $s"; done; done; done
} | xargs -P 4 -I{} sh -c "{}" >> ../.cache/h32.log 2>&1
echo "ALL DONE" >> ../.cache/h32.log
