"""Offline USD estimates: calls x average tokens x config prices. Nothing is called.

Input tokens come from the real prompts built for the task pool (chars/4). Output tokens:
  optimistic   - each model's `sim.out_tokens[role]` (or the pilot's measured mean, see --pilot)
  conservative - max_tokens[role] for models with `thinking: true` (adaptive thinking is billed as
                 output and is not bounded by anything but max_tokens; this is also exactly what the
                 budget guard assumes before each call); sim.out_tokens otherwise. After a pilot:
                 the largest measured output for that (model, role), capped at max_tokens.
"""
from __future__ import annotations

import os
import sys

import numpy as np

from .roles import ROLE_PROMPTS, executor_message, planner_message, revision_message, verifier_message

REVISE_RATE = 0.3      # assumed P(verifier asks for a revision)
MODES = ("opt", "cons")


class Estimator:
    def __init__(self, cfg, pool, measured: dict | None = None):
        """measured: {model_name: {role: {"mean": x, "max": y, "n": k}}} from a pilot."""
        self.cfg, self.pool, self.measured = cfg, pool, measured or {}
        L = lambda s: len(s) / 4
        self.in_exec = float(np.mean([L(executor_message(t)) + L(ROLE_PROMPTS["executor"]) for t in pool]))
        self.in_plan = float(np.mean([L(planner_message(t)) + L(ROLE_PROMPTS["planner"]) for t in pool]))
        self.in_ver0 = float(np.mean([L(verifier_message(t, "")) + L(ROLE_PROMPTS["verifier"]) for t in pool]))
        self.in_rev0 = L(revision_message(""))
        self.in_embed = float(np.mean([L(t.prompt) for t in pool]))
        self.mt = cfg.run["max_tokens"]
        from .embed import embedding_price
        self.embed_price = embedding_price(cfg.embeddings)      # raises for a real provider without price

    def out_tokens(self, spec, role, mode):
        m = self.measured.get(spec.name, {}).get(role)
        if m:
            return m["mean"] if mode == "opt" else min(m["max"], self.mt[role])
        if mode == "cons" and spec.thinking:
            return self.mt[role]
        return spec.sim["out_tokens"].get(role, 300)

    def team_usd(self, spec, n_members: int, mode="opt", revise_rate=REVISE_RATE) -> float:
        """Expected USD of one team execution (per task)."""
        o = lambda r: self.out_tokens(spec, r, mode)
        if n_members <= 1:
            return spec.usd(self.in_exec, o("executor"))
        usd, in_exec = 0.0, self.in_exec
        if n_members >= 3:
            usd += spec.usd(self.in_plan, o("planner"))
            in_exec += o("planner")
        usd += spec.usd(in_exec, o("executor"))
        usd += spec.usd(self.in_ver0 + o("executor"), o("verifier"))
        usd += revise_rate * spec.usd(in_exec + o("executor") + self.in_rev0 + o("verifier"), o("executor"))
        return usd

    @staticmethod
    def calls_per_task(n_members):
        return 1 if n_members <= 1 else (2 if n_members == 2 else 3) + REVISE_RATE

    def embed_usd(self, n_texts: int) -> float:
        return self.embed_price * self.in_embed * n_texts / 1e6

    # ------------------------------------------------------------------ llm_run
    def llm_run_labels(self, variants, pipeline_model="auto"):
        labels = []
        if "swarm" in variants:
            labels.append("swarm")
        if "linucb" in variants:
            labels.append("linucb")
        if "single" in variants or ("pipeline" in variants and pipeline_model == "auto"):
            labels += [f"single:{m.name}" for m in self.cfg.models]
        if "pipeline" in variants:
            labels.append(f"pipeline:{pipeline_model}")
        return labels

    def per_task(self, label, mode):
        """(calls/task, expected USD/task, max USD/task over model choices) for a variant label."""
        models = self.cfg.models
        if label.startswith("single:"):
            u = self.team_usd(self.cfg.model(label.split(":", 1)[1]), 1, mode)
            return 1, u, u
        if label.startswith("pipeline:"):
            pm = label.split(":", 1)[1]
            cand = models if pm == "auto" else [self.cfg.model(pm)]
            ex = [self.team_usd(m, 3, mode) for m in cand]
            return self.calls_per_task(3), float(np.mean(ex)), max(ex)
        if label == "linucb":
            ex = [self.team_usd(m, 1, mode) for m in models]
            return 1, float(np.mean(ex)), max(ex)
        if label == "swarm":       # expected: uniform model mix x team size uniform on {1,2,3}
            ex = [self.team_usd(m, k, mode) for m in models for k in (1, 2, 3)]
            return self.calls_per_task(2), float(np.mean(ex)), max(self.team_usd(m, 3, mode) for m in models)
        raise ValueError(label)

    def cache_cap_usd(self, n_tasks, n_seeds, mode="opt"):
        """Each (item, model, team shape) is paid at most once per cache lifetime and sample_id; with
        include_seed_in_cache_key=true there is one sample_id per seed."""
        uniq = min(n_tasks, len(self.pool))
        reps = n_seeds if self.cfg.run["include_seed_in_cache_key"] else 1
        per_item = sum(self.team_usd(m, k, mode) for m in self.cfg.models for k in (1, 2, 3))
        return uniq * per_item * reps

    def variant_shares(self, labels, budget_usd):
        spec = self.cfg.run.get("variant_budget_share", "equal")
        if spec in (None, "equal") or not isinstance(spec, dict):
            return {l: budget_usd / len(labels) for l in labels}
        fixed = {l: float(spec[l]) for l in labels if l in spec}
        rest = [l for l in labels if l not in spec]
        left = max(0.0, 1.0 - sum(fixed.values()))
        out = {l: f * budget_usd for l, f in fixed.items()}
        out.update({l: left * budget_usd / len(rest) for l in rest} if rest else {})
        return out

    def print_llm_run(self, n_tasks, n_seeds, variants, budget_usd, pipeline_model="auto", title=""):
        labels = self.llm_run_labels(variants, pipeline_model)
        shares = self.variant_shares(labels, budget_usd)
        N = n_tasks * n_seeds
        tag = " (pilot-measured output tokens)" if self.measured else ""
        print(f"Cost estimate{title}: llm_run, {n_tasks} tasks x {n_seeds} seeds, prices from {self.cfg.path}{tag}")
        uniq = min(n_tasks, len(self.pool))
        print(f"item pool ceiling: {len(self.pool)} distinct items; {uniq} distinct per seed, each used "
              f"{n_tasks / uniq:.1f}x per seed (repeats add no new information)")
        print(f"avg input tokens: executor {self.in_exec:.0f}, planner {self.in_plan:.0f}, verifier "
              f"{self.in_ver0:.0f}+answer. optimistic = sim.out_tokens; conservative = max_tokens for "
              f"thinking models {dict(self.mt)}")
        print(f"\n| {'variant':<16} | {'calls':>6} | {'USD/task opt':>12} | {'USD/task cons':>13} | "
              f"{'USD opt':>9} | {'USD cons':>9} | {'cons max-mix':>12} | {'share':>7} |")
        print("|" + "|".join("-" * w for w in (18, 8, 14, 15, 11, 11, 14, 9)) + "|")
        tot = dict(opt=0.0, cons=0.0, hi=0.0)
        rows = []
        for l in labels:
            c, eo, _ = self.per_task(l, "opt")
            _, ec, hc = self.per_task(l, "cons")
            r = dict(variant=l, calls=c * N, usd_task_opt=eo, usd_task_cons=ec, usd_opt=eo * N, usd_cons=ec * N,
                     usd_cons_hi=hc * N, share=shares[l])
            rows.append(r)
            tot["opt"] += r["usd_opt"]; tot["cons"] += r["usd_cons"]; tot["hi"] += r["usd_cons_hi"]
            flag = "" if r["usd_cons"] <= r["share"] else " !"
            print(f"| {l:<16} | {r['calls']:>6.0f} | {eo:>12.5f} | {ec:>13.5f} | {r['usd_opt']:>9.4f} | "
                  f"{r['usd_cons']:>9.4f} | {r['usd_cons_hi']:>12.4f} | {r['share']:>6.2f}{flag:<1} |")
        emb_n = N
        emb = self.embed_usd(emb_n)
        print(f"| {'embeddings':<16} | {emb_n:>6} | {'':>12} | {'':>13} | {emb:>9.4f} | {emb:>9.4f} | "
              f"{emb:>12.4f} | {'run':>7} |  provider={self.cfg.embeddings.get('provider')}")
        print(f"| {'TOTAL':<16} | {sum(r['calls'] for r in rows):>6.0f} | {'':>12} | {'':>13} | "
              f"{tot['opt'] + emb:>9.4f} | {tot['cons'] + emb:>9.4f} | {tot['hi'] + emb:>12.4f} | {budget_usd:>7.2f} |")
        cap_o, cap_c = self.cache_cap_usd(n_tasks, n_seeds, "opt"), self.cache_cap_usd(n_tasks, n_seeds, "cons")
        print(f"\ncache-aware cap (every item x model x team shape paid once per "
              f"{'seed' if self.cfg.run['include_seed_in_cache_key'] else 'cache lifetime'}): "
              f"optimistic {cap_o:.4f} USD, conservative {cap_c:.4f} USD")
        exp_c = min(tot["cons"], cap_c) + emb
        print(f"expected spend: optimistic <= {min(tot['opt'], cap_o) + emb:.4f} USD, conservative <= {exp_c:.4f} USD;"
              f" budget limit {budget_usd:.2f} USD -> {'fits' if exp_c <= budget_usd else 'MAY STOP EARLY'}")
        starved = [r["variant"] for r in rows if r["usd_cons"] > r["share"]]
        if starved:
            print(f"'!' = conservative cost above the variant's budget share ({', '.join(starved)}): "
                  f"that variant may stop early (status stopped_budget); raise --budget-usd or run --pilot.")
        print("'cons max-mix' = conservative, every call on the priciest model (team of 3 for swarm/pipeline).")
        return rows

    # ------------------------------------------------------------------ generic two-column printer
    def print_rows(self, title, rows, budget_usd):
        """rows: [(label, calls, usd_opt, usd_cons)]"""
        print(title)
        print(f"| {'item':<18} | {'calls':>6} | {'USD opt':>9} | {'USD cons':>9} |")
        print("|" + "|".join("-" * w for w in (20, 8, 11, 11)) + "|")
        to = tc = 0.0
        for lab, calls, uo, uc in rows:
            print(f"| {lab:<18} | {calls:>6.0f} | {uo:>9.4f} | {uc:>9.4f} |")
            to += uo; tc += uc
        print(f"| {'TOTAL':<18} | {sum(r[1] for r in rows):>6.0f} | {to:>9.4f} | {tc:>9.4f} |")
        print(f"optimistic = sim.out_tokens, conservative = max_tokens for thinking models; no cache. "
              f"Budget limit {budget_usd:.2f} USD -> {'fits' if tc <= budget_usd else 'conservative estimate EXCEEDS the limit'}")
        return to, tc


# ---------------------------------------------------------------------- spend authorisation
SPEND_ENV = "SWARM_ALLOW_SPEND_USD"


def spend_authorised(budget_usd: float, budget_explicit: bool):
    """A real run needs the budget stated explicitly: --budget-usd on the command line, or the env var
    SWARM_ALLOW_SPEND_USD >= budget. Returns (ok, reason)."""
    if budget_explicit:
        return True, "--budget-usd given on the command line"
    v = os.environ.get(SPEND_ENV)
    if v is not None:
        try:
            if float(v) >= budget_usd:
                return True, f"{SPEND_ENV}={v} >= budget {budget_usd}"
            return False, f"{SPEND_ENV}={v} is below the budget {budget_usd}"
        except ValueError:
            return False, f"{SPEND_ENV}={v!r} is not a number"
    return False, f"neither --budget-usd nor {SPEND_ENV} is set"


def confirm_real(print_estimate, yes: bool, budget_usd: float = 0.0, budget_explicit: bool = False,
                 isatty=None) -> bool:
    """Gate before any real spend: shows the estimate; needs an explicit budget (flag or env var) AND
    either --yes or an interactive 'yes'. Without both it returns False before any call."""
    print("REAL RUN: this calls paid APIs.")
    print_estimate()
    ok, why = spend_authorised(budget_usd, budget_explicit)
    if not ok:
        print(f"aborting without any API call: {why}. Pass --budget-usd X or set {SPEND_ENV} >= X.")
        return False
    print(f"spend authorised up to {budget_usd:.2f} USD ({why})")
    if yes:
        return True
    tty = sys.stdin.isatty() if isatty is None else isatty
    if not tty:
        print("not a TTY and --yes not given: aborting without any API call.")
        return False
    return input("type 'yes' to proceed: ").strip().lower() == "yes"
