#!/bin/bash
# H35 stage C: interval of generations K and packet size n (notes/hypotheses.md). Part 1: branches of a mature NI colony
# at 400k; part 2: the same regimes from the start. Seeds 150-155, T = 1.6M.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 H26_BASE=NI H26_FORKS=
{ for s in 150 151 152 153 154 155; do
    echo "env H26_DIR=h35c/stat H35_FORK=400000 H35_KINDS=C0,T27x9,T300x99,T50x33,T367x33,T100x9,T100x3 python3 h26_freeze.py NI $s"
    echo "env H26_DIR=h35c/mild H33_TSHIFT=800000,3,1 H35_FORK=400000 H35_KINDS=T27x9,T300x99,T100x9 python3 h26_freeze.py NI $s"
    echo "env H26_DIR=h35c/hard H33_TSHIFT=800000,3 H35_FORK=400000 H35_KINDS=T27x9,T300x99,T100x9 python3 h26_freeze.py NI $s"; done
} | xargs -P 2 -I{} sh -c "{}" >> ../.cache/h35c.log 2>&1
echo "PART 1 DONE" >> ../.cache/h35c.log
{ for s in 150 151 152 153 154 155; do for r in 27,9 300,99 50,33 200,33; do
    echo "env H26_DIR=h35c/start_$(echo $r | tr , x) H35_RATE=$r python3 h26_freeze.py NI $s"; done; done
} | xargs -P 4 -I{} sh -c "{}" >> ../.cache/h35c.log 2>&1
echo "ALL DONE" >> ../.cache/h35c.log
