"""Measure single-agent quality vs accumulated context length (CLAUDE.md, phase 2).

Default --models cheapest (long contexts are expensive: see --estimate before adding others).
Protocol: for each model separately, a seeded stream of tasks. Target task i is solved by the
executor with its conversation prefixed by the k previous tasks of the stream as transcripts
(user = task message, assistant = the same model's own k=0 reply to it), k in K. Same targets for
every k (paired). Output: success and input tokens per k (mean ± sd over seeds), paired difference
k_max - 0, and the per-seed least-squares slope of success vs k.
This script only MEASURES degradation; it does not implement fatigue/sleep.

  python3 experiments/llm_context_degradation.py --dry-run --n-tasks 30 --seeds 5
  python3 experiments/llm_context_degradation.py --estimate --n-tasks 60 --seeds 5
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from swarm.llm import BudgetExceeded  # noqa: E402
from swarm.llm.cli import common_args, make_client, setup  # noqa: E402
from swarm.llm.estimate import Estimator, confirm_real  # noqa: E402
from swarm.llm.results import config_dict, save  # noqa: E402
from swarm.llm.roles import ROLE_PROMPTS, executor_message  # noqa: E402
from swarm.llm.tasks import make_stream  # noqa: E402


def estimate(cfg, pool, models, K, n, seeds, budget):
    e = Estimator(cfg, pool)
    rows = []
    for name in models:
        spec = cfg.model(name)
        usd = {}
        for mode in ("opt", "cons"):
            out = e.out_tokens(spec, "executor", mode)
            usd[mode] = []
            for k in K:
                in_tok = e.in_exec + k * (e.in_exec - len(ROLE_PROMPTS["executor"]) / 4 + out)
                calls = n * seeds + (max(K) * seeds if k == 0 else 0)     # k=0 also pays the prefix tasks
                usd[mode].append((calls, calls * spec.usd(in_tok, out)))
        for i, k in enumerate(K):
            rows.append((f"{name} k={k}", usd["opt"][i][0], usd["opt"][i][1], usd["cons"][i][1]))
    return e.print_rows(f"Cost estimate: context degradation, {n} targets x {seeds} seeds x K={K}, models={models}"
                        f" (history replies are as long as the output assumption; no embeddings used)",
                        rows, budget)


def main(argv=None):
    ap = common_args(argparse.ArgumentParser(description=__doc__,
                                             formatter_class=argparse.RawDescriptionHelpFormatter),
                     n_tasks=30, seeds=5, models="cheapest")
    ap.add_argument("--k", default="0,2,4,8,16")
    a = ap.parse_args(argv)
    K = sorted({int(x) for x in a.k.split(",")})
    name = f"llm_context_degradation_{'real' if a.real else 'dry'}"
    cfg, pool, budget, models = setup(a, name)
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    if a.estimate:
        estimate(cfg, pool, models, K, a.n_tasks, len(seeds), budget); return None
    if a.real and not confirm_real(lambda: estimate(cfg, pool, models, K, a.n_tasks, len(seeds), budget), a.yes,
                                   budget, a.budget_usd is not None):
        return None
    client, check_fn, log_path = make_client(a, cfg, pool, budget, name)
    sid = lambda sd: sd if cfg.run["include_seed_in_cache_key"] else 0
    mt = cfg.run["max_tokens"]["executor"]
    sysp = ROLE_PROMPTS["executor"]
    kmax = max(K)
    res, status = {}, "ok"
    for mname in models:
        if status.startswith("stopped_budget"):
            break
        per_seed = []
        for sd in seeds:
            row, seed_status = {}, "ok"
            try:
                # extra head room so that every target still has kmax non-repeat predecessors
                stream = make_stream(pool, 2 * kmax + a.n_tasks, sd)
                solo = {}

                def reply0(j):
                    if j not in solo:
                        um = executor_message(stream[j])
                        solo[j] = client.call(mname, sysp, [{"role": "user", "content": um}], mt,
                                              sample_id=sid(sd), tag="ctx_k0").text or "(empty)"
                    return solo[j]
                for k in K:
                    succ, toks, trunc, refus = [], [], 0, 0
                    row[str(k)] = dict(success=None, input_tokens=None, n=0, max_tokens_stops=0, refusals=0)
                    for i in range(2 * kmax, 2 * kmax + a.n_tasks):
                        hist = []
                        # previous tasks of the stream, skipping repeats of the target item (the pool is
                        # small; a repeat would put the target's own answer in the context)
                        prev = [j for j in range(i) if stream[j].id != stream[i].id][-k:] if k else []
                        for j in prev:
                            hist += [{"role": "user", "content": executor_message(stream[j])},
                                     {"role": "assistant", "content": reply0(j)}]
                        msgs = hist + [{"role": "user", "content": executor_message(stream[i])}]
                        c = client.call(mname, sysp, msgs, mt, sample_id=sid(sd), tag=f"ctx_k{k}")
                        succ.append(float(check_fn(stream[i], c.text))); toks.append(c.input_tokens)
                        trunc += int(c.stop_reason == "max_tokens"); refus += int(c.stop_reason == "refusal")
                        row[str(k)] = dict(success=float(np.mean(succ)), input_tokens=float(np.mean(toks)),
                                           n=len(succ), max_tokens_stops=trunc, refusals=refus)
                ks = np.array(K, float); ys = np.array([row[str(k)]["success"] for k in K])
                row["slope_per_k"] = float(np.polyfit(ks, ys, 1)[0]) if len(K) > 1 else 0.0
            except BudgetExceeded as e:
                seed_status = status = f"stopped_budget: {e}"
            except Exception as e:                              # noqa: BLE001 - keep the partial seed
                seed_status = f"error: {type(e).__name__}: {e}"
                status = status if status != "ok" else "partial"
                traceback.print_exc()
            per_seed.append(dict(seed=sd, status=seed_status, **row))
            if seed_status != "ok":
                print(f"{mname} seed {sd}: {seed_status}")
                if seed_status.startswith("stopped_budget"):
                    break
        complete = [r for r in per_seed if r["status"] == "ok"]
        res[mname] = dict(per_seed=per_seed, n_seeds_ok=len(complete),
                          summary=summarize(complete, K) if complete else None)
    for mname, r in res.items():
        if not r["summary"]:
            print(f"\n{mname}: no complete seed (see per_seed in the results file)")
            continue
        print(f"\n{mname}:  k | success (mean ± sd) | input tokens | truncated")
        for k in K:
            s = r["summary"][str(k)]
            print(f"   {k:>3} | {s['success']:.3f} ± {s['success_sd']:.3f}       | {s['input_tokens']:>12.0f} | "
                  f"{s['max_tokens_stops']}")
        d = r["summary"]["diff_kmax_k0"]
        print(f"   paired diff k={kmax} - k=0: {d['diff']:+.3f} ± {d['sd']:.3f} -> "
              f"{'EFFECT' if d['effect'] else 'no effect detected'} (|diff| > 2 sd); "
              f"slope/k {r['summary']['slope_per_k']:+.4f} ± {r['summary']['slope_per_k_sd']:.4f}")
    payload = dict(mode="real" if a.real else "dry_run", status=status, seeds=seeds, K=K, n_tasks=a.n_tasks,
                   models=models, results=res, budget=dict(limit_usd=budget, spent_usd=client.run_budget.spent),
                   call_log=log_path, config=config_dict(cfg),
                   note="" if a.real else "dry-run: degradation is the invented sim.context_decay_per_1k")
    path = save(name, payload, a.out, argv=sys.argv if argv is None else [os.path.basename(__file__), *argv])
    client.log.close()
    return path


def summarize(per_seed, K):
    out = {}
    for k in K:
        s = [r[str(k)]["success"] for r in per_seed]; t = [r[str(k)]["input_tokens"] for r in per_seed]
        out[str(k)] = dict(success=float(np.mean(s)), success_sd=float(np.std(s, ddof=1)) if len(s) > 1 else 0.0,
                           input_tokens=float(np.mean(t)),
                           max_tokens_stops=int(sum(r[str(k)]["max_tokens_stops"] for r in per_seed)),
                           refusals=int(sum(r[str(k)]["refusals"] for r in per_seed)))
    d = np.array([r[str(max(K))]["success"] - r[str(min(K))]["success"] for r in per_seed])
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    out["diff_kmax_k0"] = dict(diff=float(d.mean()), sd=sd, effect=bool(len(d) > 1 and abs(d.mean()) > 2 * sd))
    sl = [r["slope_per_k"] for r in per_seed]
    out["slope_per_k"] = float(np.mean(sl)); out["slope_per_k_sd"] = float(np.std(sl, ddof=1)) if len(sl) > 1 else 0.0
    return out


if __name__ == "__main__":
    main()
