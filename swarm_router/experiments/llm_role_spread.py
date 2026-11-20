"""Is there any quality spread over (role prompt x model)? Checked before building on roles (CLAUDE.md).

Each model in each role on a fixed small task set, checker-scored:
  none      - solver without a system prompt (control for the executor prompt)
  executor  - solver with the EXECUTOR prompt: success rate
  planner   - the model writes a plan, a FIXED executor model (--executor-model, default cheapest)
              solves with it: success rate (compare with that executor's own 'executor' cell = no plan)
  verifier  - the model judges fixed candidates with known labels (one correct, one wrong per task,
              rendered deterministically): balanced accuracy = (P(accept|correct)+P(reject|wrong))/2
Caveat: max/min models are picked post hoc on the means, so the spread is biased upwards for
near-equal models; treat "detected" with that in mind (or confirm on a fresh seed set).
Per role: spread = max - min over models, with the sd of that difference over seeds (task subsets);
"spread detected" iff spread > 2 sd. Model ranks per role are reported to see role x model interaction.

  python3 experiments/llm_role_spread.py --dry-run --n-tasks 20 --seeds 5
  python3 experiments/llm_role_spread.py --estimate --n-tasks 20 --seeds 5
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
from swarm.llm.providers import SimOracle, hash_uniform  # noqa: E402
from swarm.llm.results import config_dict, save  # noqa: E402
from swarm.llm.roles import (ROLE_PROMPTS, executor_message, parse_verdict, planner_message,  # noqa: E402
                             verifier_message)
from swarm.llm.tasks import make_stream  # noqa: E402

ROLES = ("none", "executor", "planner", "verifier")


def estimate(cfg, pool, models, ex_model, n, seeds, budget):
    e = Estimator(cfg, pool)
    print(f"Cost estimate: role spread, {n} tasks x {seeds} seeds, models={models}, fixed executor={ex_model}")
    exs = cfg.model(ex_model)
    tot = 0.0
    print(f"| {'model':<8} | {'calls':>6} | {'USD':>9} |")
    print("|" + "-" * 10 + "|" + "-" * 8 + "|" + "-" * 11 + "|")
    for name in models:
        s = cfg.model(name); o = s.sim["out_tokens"]
        usd = 2 * s.usd(e.in_exec, o["executor"])                                 # none + executor
        usd += s.usd(e.in_plan, o["planner"]) + exs.usd(e.in_exec + o["planner"], exs.sim["out_tokens"]["executor"])
        usd += 2 * s.usd(e.in_ver0 + o["executor"], o["verifier"])               # 2 candidates
        usd *= n * seeds; tot += usd
        print(f"| {name:<8} | {6 * n * seeds:>6} | {usd:>9.4f} |")
    print(f"TOTAL ≈ {tot:.4f} USD (no cache; output = sim.out_tokens assumption); budget limit {budget:.2f} USD")
    return tot


def main(argv=None):
    ap = common_args(argparse.ArgumentParser(description=__doc__,
                                             formatter_class=argparse.RawDescriptionHelpFormatter),
                     n_tasks=20, seeds=5)
    ap.add_argument("--executor-model", default="cheapest")
    a = ap.parse_args(argv)
    name = f"llm_role_spread_{'real' if a.real else 'dry'}"
    cfg, pool, budget, models = setup(a, name)
    ex_model = (min(cfg.models, key=lambda m: m.price_out_per_1m).name if a.executor_model == "cheapest"
                else a.executor_model)
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    if a.estimate:
        estimate(cfg, pool, models, ex_model, a.n_tasks, len(seeds), budget); return None
    if a.real and not confirm_real(lambda: estimate(cfg, pool, models, ex_model, a.n_tasks, len(seeds), budget),
                                   a.yes):
        return None
    client, check_fn, log_path = make_client(a, cfg, pool, budget, name)
    mt = cfg.run["max_tokens"]
    sid = lambda sd: sd if cfg.run["include_seed_in_cache_key"] else 0
    status, per_seed = "ok", []

    def solve(m, system, task, sd, plan=None, tag=""):
        c = client.call(m, system, [{"role": "user", "content": executor_message(task, plan)}], mt["executor"],
                        sample_id=sid(sd), tag=tag)
        return float(check_fn(task, c.text))

    try:
        for sd in seeds:
            tasks = make_stream(pool, a.n_tasks, sd)
            row = {r: {} for r in ROLES}
            for m in models:
                row["none"][m] = float(np.mean([solve(m, None, t, sd, tag="spread_none") for t in tasks]))
                row["executor"][m] = float(np.mean([solve(m, ROLE_PROMPTS["executor"], t, sd, tag="spread_exec")
                                                    for t in tasks]))
                ps = []
                for t in tasks:
                    plan = client.call(m, ROLE_PROMPTS["planner"], [{"role": "user", "content": planner_message(t)}],
                                       mt["planner"], sample_id=sid(sd), tag="spread_plan").text
                    ps.append(solve(ex_model, ROLE_PROMPTS["executor"], t, sd, plan=plan, tag="spread_plan_exec"))
                row["planner"][m] = float(np.mean(ps))
                acc_ok, rej_bad = [], []
                for t in tasks:
                    u = hash_uniform("cand", t.id)
                    for correct in (True, False):
                        cand = SimOracle.render(t, correct, u)
                        v = client.call(m, ROLE_PROMPTS["verifier"],
                                        [{"role": "user", "content": verifier_message(t, cand)}], mt["verifier"],
                                        sample_id=sid(sd), tag="spread_verify")
                        (acc_ok if correct else rej_bad).append(float(parse_verdict(v.text) == correct))
                row["verifier"][m] = float((np.mean(acc_ok) + np.mean(rej_bad)) / 2)
            per_seed.append(dict(seed=sd, **row))
    except BudgetExceeded as e:
        status = f"stopped_budget: {e}"
        print(status)

    summary = summarize(per_seed, models)
    print(f"\n| {'role':<9} | " + " | ".join(f"{m:<15}" for m in models) + " | spread (max-min)  | ranks")
    for r in ROLES:
        s = summary.get(r)
        if not s:
            continue
        cells = " | ".join(f"{s['mean'][m]:.3f} ± {s['sd'][m]:.3f}".ljust(15) for m in models)
        flag = "SPREAD" if s["spread_detected"] else "none detected"
        print(f"| {r:<9} | {cells} | {s['spread']:.3f} ± {s['spread_sd']:.3f} {flag} | {'>'.join(s['rank'])}")
    print(f"(planner cell = success of fixed executor '{ex_model}' with that model's plan; "
          f"verifier cell = balanced accuracy)")
    payload = dict(mode="real" if a.real else "dry_run", status=status, seeds=seeds, n_tasks=a.n_tasks,
                   models=models, executor_model=ex_model, per_seed=per_seed, summary=summary,
                   rank_changes_across_roles=len({tuple(s["rank"]) for s in summary.values()}) > 1,
                   budget=dict(limit_usd=budget, spent_usd=client.budget.spent), call_log=log_path,
                   config=config_dict(cfg),
                   note="" if a.real else "dry-run: role effects are the invented sim.role_skill/plan_bonus/verify_*")
    path = save(name, payload, a.out, argv=sys.argv if argv is None else [os.path.basename(__file__), *argv])
    client.log.close()
    return path


def summarize(per_seed, models):
    out = {}
    if not per_seed:
        return out
    for r in ROLES:
        M = np.array([[row[r][m] for m in models] for row in per_seed])      # seeds x models
        mean = M.mean(0); sd = M.std(0, ddof=1) if len(M) > 1 else np.zeros(len(models))
        hi, lo = int(mean.argmax()), int(mean.argmin())
        d = M[:, hi] - M[:, lo]
        dsd = float(d.std(ddof=1)) if len(d) > 1 else 0.0
        out[r] = dict(mean=dict(zip(models, mean.tolist())), sd=dict(zip(models, sd.tolist())),
                      spread=float(d.mean()), spread_sd=dsd,
                      spread_detected=bool(len(d) > 1 and d.mean() > 2 * dsd),
                      rank=[models[i] for i in np.argsort(-mean, kind="stable")])
    return out


if __name__ == "__main__":
    main()
