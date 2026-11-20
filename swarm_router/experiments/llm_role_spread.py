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
import traceback

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
    exs = cfg.model(ex_model)
    rows = []
    for name in models:
        spec = cfg.model(name)
        usd = {}
        for mode in ("opt", "cons"):
            o = lambda r: e.out_tokens(spec, r, mode)
            u = 2 * spec.usd(e.in_exec, o("executor"))                                       # none + executor
            u += spec.usd(e.in_plan, o("planner")) + exs.usd(e.in_exec + o("planner"), e.out_tokens(exs, "executor", mode))
            u += 2 * spec.usd(e.in_ver0 + 100, o("verifier"))       # 2 fixed candidates (~100 tokens each)
            usd[mode] = u * n * seeds
        rows.append((name, 6 * n * seeds, usd["opt"], usd["cons"]))
    return e.print_rows(f"Cost estimate: role spread, {n} tasks x {seeds} seeds, models={models}, fixed executor="
                        f"{ex_model} (no embeddings used)", rows, budget)


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
                                   a.yes, budget, a.budget_usd is not None):
        return None
    client, check_fn, log_path = make_client(a, cfg, pool, budget, name)
    mt = cfg.run["max_tokens"]
    sid = lambda sd: sd if cfg.run["include_seed_in_cache_key"] else 0
    status, per_seed = "ok", []

    counts = {m: dict(max_tokens_stops=0, refusals=0, no_verdict=0, verdicts=0) for m in models}

    def call(m, system, um, max_tokens, sd, tag):
        c = client.call(m, system, [{"role": "user", "content": um}], max_tokens, sample_id=sid(sd), tag=tag)
        counts[m]["max_tokens_stops"] += int(c.stop_reason == "max_tokens")
        counts[m]["refusals"] += int(c.stop_reason == "refusal")
        return c

    def solve(m, system, task, sd, plan=None, tag=""):
        return float(check_fn(task, call(m, system, executor_message(task, plan), mt["executor"], sd, tag).text))

    for sd in seeds:
        row, seed_status = {r: {} for r in ROLES}, "ok"
        try:
            tasks = make_stream(pool, a.n_tasks, sd)
            for m in models:
                row["none"][m] = float(np.mean([solve(m, None, t, sd, tag="spread_none") for t in tasks]))
                row["executor"][m] = float(np.mean([solve(m, ROLE_PROMPTS["executor"], t, sd, tag="spread_exec")
                                                    for t in tasks]))
                ps = []
                for t in tasks:
                    plan = call(m, ROLE_PROMPTS["planner"], planner_message(t), mt["planner"], sd, "spread_plan").text
                    ps.append(solve(ex_model, ROLE_PROMPTS["executor"], t, sd, plan=plan, tag="spread_plan_exec"))
                row["planner"][m] = float(np.mean(ps))
                acc_ok, rej_bad = [], []
                for t in tasks:
                    u = hash_uniform("cand", t.id)
                    for correct in (True, False):
                        # plausible wrong candidate: failing AST mutant / off-by-one / distractor entity
                        cand = SimOracle.render(t, correct, u)
                        v = call(m, ROLE_PROMPTS["verifier"], verifier_message(t, cand), mt["verifier"], sd,
                                 "spread_verify")
                        verdict = parse_verdict(v.text)
                        counts[m]["verdicts"] += 1
                        counts[m]["no_verdict"] += int(verdict == "none")
                        accepted = verdict != "revise"            # 'none' -> default action accept
                        (acc_ok if correct else rej_bad).append(float(accepted == correct))
                row["verifier"][m] = float((np.mean(acc_ok) + np.mean(rej_bad)) / 2)
        except BudgetExceeded as e:
            seed_status = status = f"stopped_budget: {e}"
        except Exception as e:                                  # noqa: BLE001 - keep the partial seed
            seed_status = f"error: {type(e).__name__}: {e}"
            status = status if status != "ok" else "partial"
            traceback.print_exc()
        per_seed.append(dict(seed=sd, status=seed_status, **row))
        if seed_status != "ok":
            print(f"seed {sd}: {seed_status}")
            if seed_status.startswith("stopped_budget"):
                break

    summary = summarize([r for r in per_seed if r["status"] == "ok"], models)
    print(f"\n| {'role':<9} | " + " | ".join(f"{m:<15}" for m in models) + " | spread (max-min)  | ranks")
    for r in ROLES:
        s = summary.get(r)
        if not s:
            continue
        cells = " | ".join(f"{s['mean'][m]:.3f} ± {s['sd'][m]:.3f}".ljust(15) for m in models)
        flag = "SPREAD" if s["spread_detected"] else "none detected"
        print(f"| {r:<9} | {cells} | {s['spread']:.3f} ± {s['spread_sd']:.3f} {flag} | {'>'.join(s['rank'])}")
    print(f"(planner cell = success of fixed executor '{ex_model}' with that model's plan; "
          f"verifier cell = balanced accuracy, no-verdict counted as accept)")
    for m in models:
        c = counts[m]
        print(f"  {m}: truncated {c['max_tokens_stops']}, refusals {c['refusals']}, "
              f"no-verdict {c['no_verdict']}/{c['verdicts']} verifier replies")
    payload = dict(mode="real" if a.real else "dry_run", status=status, seeds=seeds, n_tasks=a.n_tasks,
                   models=models, executor_model=ex_model, per_seed=per_seed, counts=counts, summary=summary,
                   rank_changes_across_roles=len({tuple(s["rank"]) for s in summary.values()}) > 1,
                   budget=dict(limit_usd=budget, spent_usd=client.run_budget.spent), call_log=log_path,
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
