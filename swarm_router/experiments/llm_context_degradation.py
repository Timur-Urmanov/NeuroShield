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
    print(f"Cost estimate: context degradation, {n} targets x {seeds} seeds x K={K}, models={models}")
    print(f"| {'model':<8} | {'k':>3} | {'in tok/call':>11} | {'calls':>6} | {'USD':>9} |")
    print("|" + "-" * 10 + "|" + "-" * 5 + "|" + "-" * 13 + "|" + "-" * 8 + "|" + "-" * 11 + "|")
    tot = 0.0
    for name in models:
        spec = cfg.model(name)
        out = spec.sim["out_tokens"].get("executor", 300)
        for k in K:
            in_tok = e.in_exec + k * (e.in_exec - len(ROLE_PROMPTS["executor"]) / 4 + out)
            calls = n * seeds + (max(K) * seeds if k == 0 else 0)     # k=0 also pays the prefix tasks
            usd = calls * spec.usd(in_tok, out)
            tot += usd
            print(f"| {name:<8} | {k:>3} | {in_tok:>11.0f} | {calls:>6} | {usd:>9.4f} |")
    print(f"TOTAL ≈ {tot:.4f} USD (no cache; output = sim.out_tokens assumption); budget limit {budget:.2f} USD")
    return tot


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
    if a.real and not confirm_real(lambda: estimate(cfg, pool, models, K, a.n_tasks, len(seeds), budget), a.yes):
        return None
    client, check_fn, log_path = make_client(a, cfg, pool, budget, name)
    sid = lambda sd: sd if cfg.run["include_seed_in_cache_key"] else 0
    mt = cfg.run["max_tokens"]["executor"]
    sysp = ROLE_PROMPTS["executor"]
    kmax = max(K)
    res, status = {}, "ok"
    try:
        for mname in models:
            per_seed = []
            for sd in seeds:
                stream = make_stream(pool, kmax + a.n_tasks, sd)
                solo = {}

                def reply0(j):
                    if j not in solo:
                        um = executor_message(stream[j])
                        solo[j] = client.call(mname, sysp, [{"role": "user", "content": um}], mt,
                                              sample_id=sid(sd), tag="ctx_k0").text or "(empty)"
                    return solo[j]
                row = {}
                for k in K:
                    succ, toks = [], []
                    for i in range(kmax, kmax + a.n_tasks):
                        hist = []
                        for j in range(i - k, i):
                            hist += [{"role": "user", "content": executor_message(stream[j])},
                                     {"role": "assistant", "content": reply0(j)}]
                        msgs = hist + [{"role": "user", "content": executor_message(stream[i])}]
                        c = client.call(mname, sysp, msgs, mt, sample_id=sid(sd), tag=f"ctx_k{k}")
                        succ.append(float(check_fn(stream[i], c.text))); toks.append(c.input_tokens)
                    row[str(k)] = dict(success=float(np.mean(succ)), input_tokens=float(np.mean(toks)))
                ks = np.array(K, float); ys = np.array([row[str(k)]["success"] for k in K])
                row["slope_per_k"] = float(np.polyfit(ks, ys, 1)[0]) if len(K) > 1 else 0.0
                per_seed.append(dict(seed=sd, **row))
            res[mname] = dict(per_seed=per_seed, summary=summarize(per_seed, K))
    except BudgetExceeded as e:
        status = f"stopped_budget: {e}"
        print(status)
    for mname, r in res.items():
        print(f"\n{mname}:  k | success (mean ± sd) | input tokens")
        for k in K:
            s = r["summary"][str(k)]
            print(f"   {k:>3} | {s['success']:.3f} ± {s['success_sd']:.3f}       | {s['input_tokens']:.0f}")
        d = r["summary"]["diff_kmax_k0"]
        print(f"   paired diff k={kmax} - k=0: {d['diff']:+.3f} ± {d['sd']:.3f} -> "
              f"{'EFFECT' if d['effect'] else 'no effect detected'} (|diff| > 2 sd); "
              f"slope/k {r['summary']['slope_per_k']:+.4f} ± {r['summary']['slope_per_k_sd']:.4f}")
    payload = dict(mode="real" if a.real else "dry_run", status=status, seeds=seeds, K=K, n_tasks=a.n_tasks,
                   models=models, results=res, budget=dict(limit_usd=budget, spent_usd=client.budget.spent),
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
                           input_tokens=float(np.mean(t)))
    d = np.array([r[str(max(K))]["success"] - r[str(min(K))]["success"] for r in per_seed])
    sd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
    out["diff_kmax_k0"] = dict(diff=float(d.mean()), sd=sd, effect=bool(len(d) > 1 and abs(d.mean()) > 2 * sd))
    sl = [r["slope_per_k"] for r in per_seed]
    out["slope_per_k"] = float(np.mean(sl)); out["slope_per_k_sd"] = float(np.std(sl, ddof=1)) if len(sl) > 1 else 0.0
    return out


if __name__ == "__main__":
    main()
