"""Offline USD estimates: calls x average tokens x config prices. Nothing is called.

Input tokens come from the real prompts built for the task pool (chars/4). Output lengths are an
ASSUMPTION taken from each model's `sim.out_tokens` (thinking tokens on current Anthropic models are
billed as output, so treat them as a floor); the "worst" column uses max_tokens for every output,
which is exactly what the budget guard assumes before each call.
"""
from __future__ import annotations

import sys

import numpy as np

from .roles import ROLE_PROMPTS, executor_message, planner_message, revision_message, verifier_message

REVISE_RATE = 0.3      # assumed P(verifier asks for a revision) in the "expected" column


class Estimator:
    def __init__(self, cfg, pool):
        self.cfg, self.pool = cfg, pool
        L = lambda s: len(s) / 4
        self.in_exec = float(np.mean([L(executor_message(t)) + L(ROLE_PROMPTS["executor"]) for t in pool]))
        self.in_plan = float(np.mean([L(planner_message(t)) + L(ROLE_PROMPTS["planner"]) for t in pool]))
        self.in_ver0 = float(np.mean([L(verifier_message(t, "")) + L(ROLE_PROMPTS["verifier"]) for t in pool]))
        self.in_rev0 = L(revision_message(""))
        self.mt = cfg.run["max_tokens"]

    def _out(self, spec, role, worst):
        return self.mt[role] if worst else spec.sim["out_tokens"].get(role, 300)

    def team_usd(self, spec, n_members: int, worst=False, revise_rate=REVISE_RATE) -> float:
        """Expected USD of one team execution (per task)."""
        o = lambda r: self._out(spec, r, worst)
        if n_members <= 1:
            return spec.usd(self.in_exec, o("executor"))
        usd = 0.0
        in_exec = self.in_exec
        if n_members >= 3:
            usd += spec.usd(self.in_plan, o("planner"))
            in_exec += o("planner")
        usd += spec.usd(in_exec, o("executor"))
        usd += spec.usd(self.in_ver0 + o("executor"), o("verifier"))
        rr = 1.0 if worst else revise_rate
        usd += rr * spec.usd(in_exec + o("executor") + self.in_rev0 + o("verifier"), o("executor"))
        return usd

    def calls_per_task(self, n_members, worst=False):
        base = {1: 1, 2: 2}.get(n_members, 3)
        return base if n_members <= 1 else base + (1.0 if worst else REVISE_RATE)

    # ------------------------------------------------------------------ llm_run
    def llm_run_table(self, n_tasks, n_seeds, variants, pipeline_model="auto"):
        models = self.cfg.models
        N = n_tasks * n_seeds
        rows = []
        def row(name, calls, exp_lo, exp, exp_hi, worst):
            rows.append(dict(variant=name, calls=calls * N, usd_task=exp, usd_lo=exp_lo * N, usd=exp * N,
                             usd_hi=exp_hi * N, usd_worst=worst * N))
        singles = [(m, self.team_usd(m, 1), self.team_usd(m, 1, True)) for m in models]
        if "single" in variants or ("pipeline" in variants and pipeline_model == "auto"):
            for m, e, w in singles:
                row(f"single:{m.name}", 1, e, e, e, w)
        if "pipeline" in variants:
            cand = models if pipeline_model == "auto" else [self.cfg.model(pipeline_model)]
            ex = [self.team_usd(m, 3) for m in cand]
            row(f"pipeline:{pipeline_model}", self.calls_per_task(3), min(ex), float(np.mean(ex)), max(ex),
                max(self.team_usd(m, 3, True) for m in cand))
        if "linucb" in variants:
            ex = [e for _, e, _ in singles]
            row("linucb", 1, min(ex), float(np.mean(ex)), max(ex), max(w for _, _, w in singles))
        if "swarm" in variants:
            ex1 = [e for _, e, _ in singles]
            ex3 = [self.team_usd(m, 3) for m in models]
            # expected: uniform model mix, team size uniform over {1,2,3}; hi: team 3 on the priciest model
            ex2 = [self.team_usd(m, 2) for m in models]
            mean = float(np.mean(ex1 + ex2 + ex3))
            row("swarm", self.calls_per_task(2), min(ex1), mean, max(ex3), max(self.team_usd(m, 3, True) for m in models))
        return rows

    def cache_cap_usd(self, n_tasks, n_seeds):
        """With caching and include_seed_in_cache_key=false, every (item, model, team shape) is executed
        at most once per cache lifetime: an upper bound on real spend that ignores repetitions."""
        uniq = min(n_tasks, len(self.pool))
        reps = n_seeds if self.cfg.run["include_seed_in_cache_key"] else 1
        per_item = sum(self.team_usd(m, k) for m in self.cfg.models for k in (1, 2, 3))
        return uniq * per_item * reps

    def print_llm_run(self, n_tasks, n_seeds, variants, budget_usd, pipeline_model="auto"):
        rows = self.llm_run_table(n_tasks, n_seeds, variants, pipeline_model)
        print(f"Cost estimate: llm_run, {n_tasks} tasks x {n_seeds} seeds, pool={len(self.pool)} items, "
              f"prices from {self.cfg.path}")
        print(f"avg input tokens: executor {self.in_exec:.0f}, planner {self.in_plan:.0f}, "
              f"verifier {self.in_ver0:.0f}+answer; output tokens = sim.out_tokens (assumption)")
        print(f"\n| {'variant':<18} | {'calls':>7} | {'USD/task':>9} | {'USD expected':>12} | {'USD range':>17} | {'USD worst':>10} |")
        print("|" + "-" * 20 + "|" + "-" * 9 + "|" + "-" * 11 + "|" + "-" * 14 + "|" + "-" * 19 + "|" + "-" * 12 + "|")
        tot = tot_hi = tot_w = 0.0
        for r in rows:
            print(f"| {r['variant']:<18} | {r['calls']:>7.0f} | {r['usd_task']:>9.5f} | {r['usd']:>12.4f} | "
                  f"{r['usd_lo']:>7.4f}–{r['usd_hi']:<9.4f} | {r['usd_worst']:>10.4f} |")
            tot += r["usd"]; tot_hi += r["usd_hi"]; tot_w += r["usd_worst"]
        cap = self.cache_cap_usd(n_tasks, n_seeds)
        print(f"| {'TOTAL':<18} | {sum(r['calls'] for r in rows):>7.0f} | {'':>9} | {tot:>12.4f} | "
              f"{'':>7} {tot_hi:<9.4f} | {tot_w:>10.4f} |")
        print(f"\ncache-aware cap (each item x model x team shape paid once; "
              f"include_seed_in_cache_key={self.cfg.run['include_seed_in_cache_key']}): {cap:.4f} USD")
        print(f"expected spend with cache: <= {min(tot, cap):.4f} USD; budget guard limit: {budget_usd:.2f} USD "
              f"-> {'fits' if min(tot_hi, cap) <= budget_usd else 'MAY STOP EARLY (BudgetExceeded)'}")
        print("'USD range' = cheapest..priciest model mix; 'worst' = every output at max_tokens, every verifier "
              "asks for a revision (no cache).")
        return rows


def confirm_real(print_estimate, yes: bool) -> bool:
    """Show the estimate before any real spend; require --yes or an interactive 'yes'."""
    print("REAL RUN: this calls paid APIs.")
    print_estimate()
    if yes:
        return True
    if not sys.stdin.isatty():
        print("not a TTY and --yes not given: aborting without any API call.")
        return False
    return input("type 'yes' to proceed: ").strip().lower() == "yes"
