#!/usr/bin/env bash
# H11 test driver (detached; resumes from .cache/h11_test_T8000.jsonl)
set -u
cd "$(dirname "$0")/.."
python3 experiments/h11_factor_links.py --stage test --lr factor=0.02,factor_ctx=0.02,mf=0.02,mf_ctx=0.05 --workers 3 >> .cache/h11.log 2>&1 && echo "TEST DONE" >> .cache/h11.log
echo "ALL DONE" >> .cache/h11.log
