"""Phase 2 main comparison on LLM tasks: swarm (SwarmConfig() defaults) vs
  (a) every single model (best one marked), (b) LinUCB router over models,
  (c) fixed plan -> execute -> verify pipeline on the best single model.
Reports reward and USD cost per task, mean ± sd over seeds.

  python3 experiments/llm_run.py --dry-run --n-tasks 60 --seeds 2       # offline, SimProvider
  python3 experiments/llm_run.py --estimate --n-tasks 300 --seeds 5     # cost estimate, no calls
  python3 experiments/llm_run.py --real --budget-usd 5 --n-tasks 300 --seeds 5   # real API (asks to confirm)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from swarm import SwarmConfig, Swarm  # noqa: E402
from swarm.baselines import LinUCB  # noqa: E402
from swarm.config import to_dict  # noqa: E402
from swarm.envs.llm import LLMEnv  # noqa: E402
from swarm.llm import BudgetExceeded, build_client, load_config  # noqa: E402
from swarm.llm.embed import make_embedder  # noqa: E402
from swarm.llm.estimate import Estimator, confirm_real  # noqa: E402
from swarm.llm.results import config_dict, mean_sd, save  # noqa: E402
from swarm.llm.tasks import load_pool, make_stream  # noqa: E402

METRICS = ("reward", "success", "usd_per_task", "charged_usd", "calls_per_task", "cache_hit_rate", "team_size")
ALL_VARIANTS = ("single", "pipeline", "linucb", "swarm")


def run_one(env: LLMEnv, variant: str, seed: int, m: int | None = None) -> dict:
    R, TS = [], []
    status = "ok"
    try:
        if variant == "swarm":
            s = Swarm(SwarmConfig(), seed, env.M, env.D)
            for t in range(env.T):
                r, info = s.step(t, env)
                R.append(r); TS.append(len(info["members"]))
        elif variant == "linucb":
            lin = LinUCB(env.M, env.D)
            for t in range(env.T):
                mm = lin.choose(env.X[t])
                r = env.step_reward(env.execute(t, mm, 1, "linucb"))
                lin.update(env.X[t], mm, r)
                R.append(r); TS.append(1)
        else:                                    # single / pipeline on model m
            k = 1 if variant == "single" else 3
            for t in range(env.T):
                R.append(env.step_reward(env.execute(t, m, k, variant))); TS.append(k)
    except BudgetExceeded as e:
        status = f"stopped_budget: {e}"
    recs = env.records[:len(R)]
    n = max(len(R), 1)
    share = {name: sum(r["model"] == name for r in recs) / n for name in env.models}
    return dict(seed=seed, status=status, n_done=len(R), reward=float(np.mean(R)) if R else None,
                success=float(np.mean([r["success"] for r in recs])) if recs else None,
                usd_per_task=float(np.sum([r["usd"] for r in recs]) / n) if recs else None,
                charged_usd=float(np.sum([r["charged"] for r in recs])),
                calls_per_task=float(np.sum([r["calls"] for r in recs]) / n) if recs else None,
                cache_hit_rate=float(np.sum([r["hits"] for r in recs]) / max(1, np.sum([r["calls"] for r in recs]))),
                team_size=float(np.mean(TS)) if TS else None, model_share=share,
                success_by_kind={k: float(np.mean([r["success"] for r in recs if r["kind"] == k]))
                                 for k in ("math", "code", "text") if any(r["kind"] == k for r in recs)})


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "llm.example.yaml"))
    ap.add_argument("--dry-run", action="store_true", default=True, help="SimProvider (default)")
    ap.add_argument("--real", action="store_true", help="call the real providers (spends money)")
    ap.add_argument("--yes", action="store_true", help="skip the interactive confirmation for --real")
    ap.add_argument("--estimate", action="store_true", help="print a USD estimate and exit (no calls)")
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--n-tasks", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS))
    ap.add_argument("--pipeline-model", default="auto", help="model name or 'auto' (= best single)")
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--name", default=None)
    a = ap.parse_args(argv)
    dry = not a.real
    cfg = load_config(a.config)
    budget_usd = a.budget_usd if a.budget_usd is not None else cfg.run["budget_usd"]
    pool = load_pool(cfg.tasks, ROOT)
    variants = [v.strip() for v in a.variants.split(",") if v.strip()]
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    est = Estimator(cfg, pool)
    if a.estimate:
        est.print_llm_run(a.n_tasks, len(seeds), variants, budget_usd, a.pipeline_model)
        return None
    if not dry and not confirm_real(lambda: est.print_llm_run(a.n_tasks, len(seeds), variants, budget_usd,
                                                              a.pipeline_model), a.yes):
        return None

    name = a.name or f"llm_run_{'dry' if dry else 'real'}_n{a.n_tasks}_s{len(seeds)}"
    log_path = os.path.join(cfg.resolve(cfg.run["log_dir"]) if a.out == os.path.join(ROOT, "results")
                            else os.path.join(a.out, "llm_calls"), f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    client = build_client(cfg, pool, dry, budget_usd, a.cache_dir, log_path)
    embedder = make_embedder(cfg.embeddings, dry, client.budget, client.cache, client.log)
    include_seed = bool(cfg.run["include_seed_in_cache_key"])
    streams = {sd: make_stream(pool, a.n_tasks, sd) for sd in seeds}

    def env_for(sd):
        return LLMEnv(cfg, client, streams[sd], embedder, sd, sample_id=sd if include_seed else 0)

    t0 = time.time()
    res, stopped = {}, False

    def run_variant(label, variant, m=None):
        nonlocal stopped
        rows = []
        for sd in seeds:
            if stopped:
                break
            row = run_one(env_for(sd), variant, sd, m)
            rows.append(row)
            if row["status"] != "ok":
                stopped = True
        res[label] = dict(variant=variant, model=None if m is None else cfg.names[m], per_seed=rows,
                          **mean_sd([r for r in rows if r["status"] == "ok"], METRICS))
        print(f"  {label:<22} done ({time.time() - t0:.0f}s, spent {client.budget.spent:.4f} USD)", flush=True)

    if "single" in variants or ("pipeline" in variants and a.pipeline_model == "auto"):
        for m, nm in enumerate(cfg.names):
            run_variant(f"single:{nm}", "single", m)
    best = None
    singles = {k: v for k, v in res.items() if k.startswith("single:") and "reward" in v}
    if singles:
        best = max(singles, key=lambda k: singles[k]["reward"]).split(":", 1)[1]
    if "pipeline" in variants and not stopped:
        pm = best if a.pipeline_model == "auto" else a.pipeline_model
        if pm is not None:
            run_variant(f"pipeline:{pm}", "pipeline", cfg.names.index(pm))
    if "linucb" in variants and not stopped:
        run_variant("linucb", "linucb")
    if "swarm" in variants and not stopped:
        run_variant("swarm", "swarm")

    print_table(res, best)
    payload = dict(mode="dry_run" if dry else "real", status="stopped_budget" if stopped else "ok",
                   seeds=seeds, n_tasks=a.n_tasks, best_single=best, variants=res,
                   budget=dict(limit_usd=budget_usd, spent_usd=client.budget.spent, n_api_calls=client.budget.n_calls),
                   cache=dict(path=client.cache.path, calls=client.n_calls, hits=client.n_hits),
                   call_log=log_path, swarm_config=to_dict(SwarmConfig()), config=config_dict(cfg),
                   pool=dict(n=len(pool), by_kind={k: sum(t.kind == k for t in pool) for k in ("math", "code", "text")}),
                   elapsed_s=time.time() - t0,
                   note=("dry-run: SimProvider with invented accuracies from the config `sim:` blocks; "
                         "numbers validate the pipeline only, not the models") if dry else "")
    path = save(name, payload, a.out, argv=sys.argv if argv is None else ["llm_run.py", *argv])
    client.log.close()
    return path


def print_table(res, best):
    cols = ("reward", "success", "usd_per_task", "calls_per_task", "team_size")
    print(f"\n| {'variant':<22} | " + " | ".join(f"{c:<17}" for c in cols) + " |")
    print("|" + "-" * 24 + "|" + "|".join("-" * 19 for _ in cols) + "|")
    for n, r in res.items():
        cells = []
        for c in cols:
            if c not in r:
                cells.append(f"{'—':<17}")
            elif c == "usd_per_task":
                cells.append(f"{r[c]:.5f} ± {r[c + '_sd']:.5f}"[:17].ljust(17))
            else:
                cells.append(f"{r[c]:.3f} ± {r[c + '_sd']:.3f}".ljust(17))
        mark = " *" if best and n == f"single:{best}" else ""
        print(f"| {n + mark:<22} | " + " | ".join(cells) + " |")
    print("(* = best single model, chosen post hoc on mean reward)")


if __name__ == "__main__":
    main()
