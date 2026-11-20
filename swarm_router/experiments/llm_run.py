"""Phase 2 main comparison on LLM tasks: swarm (SwarmConfig() defaults) vs
  (a) every single model (best one marked), (b) LinUCB router over models,
  (c) fixed plan -> execute -> verify pipeline on the best single model.
Reports reward and USD cost per task: mean ± sd over seeds, plus the SEM over items.
Variant order: swarm, linucb, singles, pipeline; each variant gets its own budget share.

  python3 experiments/llm_run.py --dry-run --n-tasks 60 --seeds 2       # offline, SimProvider
  python3 experiments/llm_run.py --estimate --n-tasks 300 --seeds 5     # cost estimate, no calls
  python3 experiments/llm_run.py --real --budget-usd 5 --pilot 6 --pilot-only   # measure output tokens
  python3 experiments/llm_run.py --real --budget-usd 5 --n-tasks 300 --seeds 5  # real API (asks to confirm)
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from swarm import Swarm, SwarmConfig  # noqa: E402
from swarm.baselines import LinUCB  # noqa: E402
from swarm.config import to_dict  # noqa: E402
from swarm.envs.llm import LLMEnv  # noqa: E402
from swarm.llm import BudgetExceeded, build_client, load_config  # noqa: E402
from swarm.llm.embed import make_embedder  # noqa: E402
from swarm.llm.estimate import Estimator, confirm_real  # noqa: E402
from swarm.llm.providers import SimOracle, hash_uniform  # noqa: E402
from swarm.llm.results import config_dict, mean_sd, save  # noqa: E402
from swarm.llm.roles import ROLE_PROMPTS, executor_message, planner_message, verifier_message  # noqa: E402
from swarm.llm.tasks import load_pool, make_stream  # noqa: E402

METRICS = ("reward", "success", "usd_per_task", "charged_usd", "calls_per_task", "cache_hit_rate", "team_size")
ALL_VARIANTS = ("swarm", "linucb", "single", "pipeline")


def run_one(env: LLMEnv, variant: str, seed: int, m: int | None = None) -> dict:
    """One (variant, seed). Never raises: budget stops and errors give a partial row with a status."""
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
        status = f"stopped_budget[{e.scope}]: {e}"
    except Exception as e:                                       # noqa: BLE001 - keep the partial seed
        status = f"error: {type(e).__name__}: {e}"
        traceback.print_exc()
    recs = env.records[:len(R)]
    n = max(len(R), 1)
    calls = int(np.sum([r["calls"] for r in recs])) if recs else 0
    verd = [r["verdict"] for r in recs if r["verdict"] is not None]
    return dict(seed=seed, status=status, n_done=len(R), reward=float(np.mean(R)) if R else None,
                success=float(np.mean([r["success"] for r in recs])) if recs else None,
                usd_per_task=float(np.sum([r["usd"] for r in recs]) / n) if recs else None,
                charged_usd=float(np.sum([r["charged"] for r in recs])),
                calls_per_task=calls / n if recs else None,
                cache_hit_rate=float(np.sum([r["hits"] for r in recs]) / max(1, calls)),
                team_size=float(np.mean(TS)) if TS else None,
                model_share={name: sum(r["model"] == name for r in recs) / n for name in env.models},
                success_by_kind={k: float(np.mean([r["success"] for r in recs if r["kind"] == k]))
                                 for k in ("math", "code", "text") if any(r["kind"] == k for r in recs)},
                max_tokens_stops=int(sum(r["max_tokens_stops"] for r in recs)),
                refusals=int(sum(r["refusals"] for r in recs)),
                verdicts={v: verd.count(v) for v in ("accept", "revise", "none")},
                items=[(r["task"], r["reward"], float(r["success"])) for r in recs])


def item_sem(rows):
    """SEM over items: per-item reward averaged over all its occurrences (seeds, repeats), then
    sd/sqrt(n_items). Complements the sd over seeds, which ignores the item-sampling error."""
    by = {}
    for row in rows:
        for tid, r, s in row["items"]:
            by.setdefault(tid, []).append((r, s))
    if len(by) < 2:
        return dict(n_items=len(by))
    rr = np.array([np.mean([x[0] for x in v]) for v in by.values()])
    ss = np.array([np.mean([x[1] for x in v]) for v in by.values()])
    k = np.sqrt(len(by))
    return dict(n_items=len(by), reward_sem_items=float(rr.std(ddof=1) / k), success_sem_items=float(ss.std(ddof=1) / k))


def pilot(client, cfg, pool, n, sample_id=0):
    """n calls per model, round-robin over roles (executor, planner, verifier) on the first pool items.
    Returns measured output tokens {model: {role: {mean, max, n}}}."""
    roles = ("executor", "planner", "verifier")
    mt = cfg.run["max_tokens"]
    out = {}
    for spec in cfg.models:
        meas = {}
        for i in range(n):
            t = pool[(i * 7) % len(pool)]
            role = roles[i % 3]
            if role == "executor":
                um = executor_message(t)
            elif role == "planner":
                um = planner_message(t)
            else:
                um = verifier_message(t, SimOracle.render(t, hash_uniform("pilot", t.id) < 0.5, 0.3))
            c = client.call(spec.name, ROLE_PROMPTS[role], [{"role": "user", "content": um}], mt[role],
                            sample_id=sample_id, tag=f"pilot_{role}")
            meas.setdefault(role, []).append(c.output_tokens)
        out[spec.name] = {r: dict(mean=float(np.mean(v)), max=float(np.max(v)), n=len(v)) for r, v in meas.items()}
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=os.path.join(ROOT, "configs", "llm.example.yaml"))
    ap.add_argument("--dry-run", action="store_true", default=True, help="SimProvider (default)")
    ap.add_argument("--real", action="store_true", help="call the real providers (spends money)")
    ap.add_argument("--yes", action="store_true", help="skip the interactive confirmation (needs --budget-usd "
                                                       "or SWARM_ALLOW_SPEND_USD as well)")
    ap.add_argument("--estimate", action="store_true", help="print a USD estimate and exit (no calls)")
    ap.add_argument("--budget-usd", type=float, default=None)
    ap.add_argument("--n-tasks", type=int, default=60)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--variants", default=",".join(ALL_VARIANTS))
    ap.add_argument("--pipeline-model", default="auto", help="model name or 'auto' (= best single)")
    ap.add_argument("--pilot", type=int, default=0, help="N calls per model to measure output tokens, then "
                                                         "re-estimate (real calls only with --real)")
    ap.add_argument("--pilot-only", action="store_true", help="stop after the pilot")
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
    show = lambda e, title="": e.print_llm_run(a.n_tasks, len(seeds), variants, budget_usd, a.pipeline_model, title)
    if a.estimate:
        show(est)
        return None
    if not dry and not confirm_real(lambda: show(est), a.yes, budget_usd, a.budget_usd is not None):
        return None

    name = a.name or f"llm_run_{'dry' if dry else 'real'}_n{a.n_tasks}_s{len(seeds)}"
    log_dir = (cfg.resolve(cfg.run["log_dir"]) if a.out == os.path.join(ROOT, "results")
               else os.path.join(a.out, "llm_calls"))
    log_path = os.path.join(log_dir, f"{name}_{time.strftime('%Y%m%d-%H%M%S')}.jsonl")
    client = build_client(cfg, pool, dry, budget_usd, a.cache_dir, log_path)
    run_budget = client.run_budget
    include_seed = bool(cfg.run["include_seed_in_cache_key"])
    t0 = time.time()
    res, run_status, measured = {}, "ok", None
    labels = est.llm_run_labels(variants, a.pipeline_model)

    def finish():
        print_table(res, best)
        payload = dict(mode="dry_run" if dry else "real", status=run_status, seeds=seeds, n_tasks=a.n_tasks,
                       best_single=best, variants=res, pilot=measured,
                       budget=dict(limit_usd=budget_usd, spent_usd=run_budget.spent, n_api_calls=run_budget.n_calls,
                                   shares=shares),
                       cache=dict(path=client.cache.path, calls=client.n_calls, hits=client.n_hits,
                                  include_seed_in_cache_key=include_seed),
                       pool=dict(n=len(pool), by_kind={k: sum(t.kind == k for t in pool) for k in ("math", "code", "text")},
                                 distinct_per_seed=min(a.n_tasks, len(pool))),
                       call_log=log_path, swarm_config=to_dict(SwarmConfig()), config=config_dict(cfg),
                       elapsed_s=time.time() - t0,
                       note=("dry-run: SimProvider with invented accuracies from the config `sim:` blocks; "
                             "numbers validate the pipeline only, not the models") if dry else "")
        path = save(name, payload, a.out, argv=sys.argv if argv is None else ["llm_run.py", *argv])
        client.log.close()
        return path

    best, shares = None, {}
    try:
        if a.pilot > 0:
            measured = pilot(client, cfg, pool, a.pilot)
            print(f"pilot: {a.pilot} calls/model, spent {run_budget.spent:.4f} USD; measured output tokens:")
            for mname, d in measured.items():
                print(f"  {mname}: " + ", ".join(f"{r} mean {v['mean']:.0f} max {v['max']:.0f}" for r, v in d.items()))
            est = Estimator(cfg, pool, measured)
            show(est, " after pilot")
            if a.pilot_only:
                run_status = "pilot_only"
                return finish()
            if not dry and not a.yes and not confirm_real(lambda: None, False, budget_usd, a.budget_usd is not None):
                run_status = "aborted_after_pilot"
                return finish()
        embedder = make_embedder(cfg.embeddings, dry, run_budget, client.cache, client.log)
        streams = {sd: make_stream(pool, a.n_tasks, sd) for sd in seeds}
        X = {sd: embedder.embed([t.prompt for t in streams[sd]]) for sd in seeds}
        shares = est.variant_shares(labels, run_budget.remaining)

        class _Fixed:              # embeddings computed once per seed, reused by every variant
            def __init__(self, sd): self.sd = sd
            def embed(self, texts): return X[self.sd]

        def run_variant(label, variant, m=None, share_key=None):
            nonlocal run_status
            share = shares.get(share_key or label, run_budget.remaining)
            client.budget = run_budget.share(share, label)
            rows = []
            for sd in seeds:
                env = LLMEnv(cfg, client, streams[sd], _Fixed(sd), sd, sample_id=sd if include_seed else 0)
                row = run_one(env, variant, sd, m)
                rows.append(row)
                if row["status"].startswith("stopped_budget[run]"):
                    run_status = "stopped_budget"
                    break
                if row["status"] != "ok":          # share exhausted or error: next variant
                    if run_status == "ok":
                        run_status = "partial"
                    break
            client.budget = run_budget
            ok = [r for r in rows if r["status"] == "ok"]
            partial = not ok
            agg_rows = ok or [r for r in rows if r["n_done"] > 0]      # no complete seed: report the partial one
            res[label] = dict(variant=variant, model=None if m is None else cfg.names[m], per_seed=rows,
                              share_usd=share, n_seeds_ok=len(ok), partial=partial,
                              **mean_sd(agg_rows, METRICS), **item_sem(rows),
                              max_tokens_stops=sum(r["max_tokens_stops"] for r in rows),
                              refusals=sum(r["refusals"] for r in rows),
                              verdicts={v: sum(r["verdicts"][v] for r in rows) for v in ("accept", "revise", "none")})
            print(f"  {label:<22} {rows[-1]['status'][:60] if rows else ''} ({time.time() - t0:.0f}s, "
                  f"spent {run_budget.spent:.4f} USD)", flush=True)

        for label in labels:
            if run_status == "stopped_budget":
                break
            if label in ("swarm", "linucb"):
                run_variant(label, label)
            elif label.startswith("single:"):
                run_variant(label, "single", cfg.names.index(label.split(":", 1)[1]))
            elif label.startswith("pipeline:"):
                singles = {k: v for k, v in res.items() if k.startswith("single:") and "reward" in v}
                if singles:
                    best = max(singles, key=lambda k: singles[k]["reward"]).split(":", 1)[1]
                pm = best if a.pipeline_model == "auto" else a.pipeline_model
                if pm is not None:
                    run_variant(f"pipeline:{pm}", "pipeline", cfg.names.index(pm), share_key=label)
        singles = {k: v for k, v in res.items() if k.startswith("single:") and "reward" in v}
        if singles and best is None:
            best = max(singles, key=lambda k: singles[k]["reward"]).split(":", 1)[1]
    except BudgetExceeded as e:
        run_status = f"stopped_budget: {e}"
    except Exception as e:                                   # noqa: BLE001 - always save what we have
        run_status = f"error: {type(e).__name__}: {e}"
        traceback.print_exc()
    return finish()


def print_table(res, best):
    cols = ("reward", "success", "usd_per_task", "calls_per_task", "team_size")
    print(f"\n| {'variant':<30} | " + " | ".join(f"{c:<17}" for c in cols)
          + " | SEM items (n) | trunc/refus/no-verdict |")
    print("|" + "-" * 32 + "|" + "|".join("-" * 19 for _ in cols) + "|" + "-" * 15 + "|" + "-" * 24 + "|")
    for n, r in res.items():
        cells = []
        for c in cols:
            if c not in r:
                cells.append(f"{'—':<17}")
            elif c == "usd_per_task":
                cells.append(f"{r[c]:.5f} ± {r[c + '_sd']:.5f}"[:17].ljust(17))
            else:
                cells.append(f"{r[c]:.3f} ± {r[c + '_sd']:.3f}".ljust(17))
        sem = f"{r['reward_sem_items']:.3f} ({r['n_items']})" if "reward_sem_items" in r else "—"
        mark = (" *" if best and n == f"single:{best}" else "") + (" (partial)" if r.get("partial") else "")
        cnt = f"{r['max_tokens_stops']}/{r['refusals']}/{r['verdicts']['none']}"
        print(f"| {n + mark:<30} | " + " | ".join(cells) + f" | {sem:<13} | {cnt:<22} |")
    print("(± = sd over seeds; SEM items = sd of per-item mean reward / sqrt(n distinct items); "
          "* = best single, chosen post hoc; (partial) = no complete seed, numbers from the stopped one; trunc = calls stopped at max_tokens; no-verdict counted as accept)")


if __name__ == "__main__":
    main()
