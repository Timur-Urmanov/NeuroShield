"""Shared CLI plumbing for the small LLM measurement scripts."""
from __future__ import annotations

import os
import time
from functools import partial

from .client import build_client
from .config import ROOT, load_config
from .tasks import load_pool
from .tasks.checkers import check


def common_args(ap, n_tasks=30, seeds=5, models="all"):
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "llm.example.yaml"))
    ap.add_argument("--dry-run", action="store_true", default=True, help="SimProvider (default)")
    ap.add_argument("--real", action="store_true", help="call the real providers (spends money)")
    ap.add_argument("--yes", action="store_true", help="skip the interactive confirmation for --real")
    ap.add_argument("--estimate", action="store_true", help="print a USD estimate and exit (no calls)")
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--n-tasks", type=int, default=n_tasks)
    ap.add_argument("--seeds", type=int, default=seeds)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--models", default=models, help="comma-separated model names, 'all' or 'cheapest'")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--cache-dir", default=None)
    return ap


def setup(a, name):
    cfg = load_config(a.config)
    pool = load_pool(cfg.tasks, ROOT)
    budget = a.budget_usd if a.budget_usd is not None else cfg.run["budget_usd"]
    if a.models == "all":
        models = cfg.names
    elif a.models == "cheapest":
        models = [min(cfg.models, key=lambda m: m.price_out_per_1m).name]
    else:
        models = [m.strip() for m in a.models.split(",")]
    for m in models:
        cfg.model(m)
    return cfg, pool, budget, models


def make_client(a, cfg, pool, budget, name):
    dry = not a.real
    log_dir = (cfg.resolve(cfg.run["log_dir"]) if a.out == os.path.join(ROOT, "results")
               else os.path.join(a.out, "llm_calls"))
    log_path = os.path.join(log_dir, f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    client = build_client(cfg, pool, dry, budget, a.cache_dir, log_path)
    check_fn = partial(check, timeout=float(cfg.tasks["code_timeout_s"]), mem_mb=int(cfg.tasks["code_mem_mb"]))
    return client, check_fn, log_path
