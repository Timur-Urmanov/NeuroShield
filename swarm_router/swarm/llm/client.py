"""LLMClient = providers + response cache + budget guard + call log. All model calls go through here."""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Optional

from .budget import Budget, CallLog, estimate_tokens, prompt_chars
from .cache import ResponseCache, cache_key
from .config import LLMConfig
from .providers import Completion, SimOracle, completion_usd, make_provider


@dataclass
class CallResult:
    text: str
    input_tokens: int
    output_tokens: int
    usd_nominal: float        # cost of this response as billed when it was first produced
    usd_charged: float        # charged to this run's budget (0 on a cache hit)
    cache_hit: bool
    stop_reason: str


class LLMClient:
    def __init__(self, cfg: LLMConfig, budget: Budget, cache: ResponseCache, log: CallLog,
                 dry_run: bool = True, oracle: Optional[SimOracle] = None):
        self.cfg, self.budget, self.cache, self.log, self.dry_run = cfg, budget, cache, log, dry_run
        self.providers = {m.name: make_provider(m, dry_run, oracle) for m in cfg.models}
        self.n_calls = self.n_hits = 0

    def estimate_usd(self, model: str, system, messages, max_tokens: int) -> float:
        spec = self.cfg.model(model)
        return spec.usd(estimate_tokens(prompt_chars(system, messages)), max_tokens)

    def call(self, model: str, system: Optional[str], messages: list, max_tokens: int,
             sample_id: int = 0, tag: str = "") -> CallResult:
        prov = self.providers[model]
        spec = prov.spec
        params = dict(max_tokens=int(max_tokens), sample_id=int(sample_id))
        key = cache_key(prov.identity(), system, messages, params)
        self.n_calls += 1
        hit = self.cache.get(key)
        if hit is not None:
            self.n_hits += 1
            self.log.write(model=model, model_id=spec.model, provider=prov.kind, tag=tag, cache_hit=True,
                           input_tokens=hit["input_tokens"], output_tokens=hit["output_tokens"],
                           usd=0.0, usd_nominal=hit["usd"], spent=self.budget.spent, key=key[:16])
            return CallResult(hit["text"], hit["input_tokens"], hit["output_tokens"], hit["usd"], 0.0, True,
                              hit.get("stop_reason", ""))
        est = self.estimate_usd(model, system, messages, max_tokens)
        self.budget.check(est, f"{model}/{tag}")       # raises BudgetExceeded: the call is not made
        t0 = time.time()
        if prov.kind == "sim":
            c: Completion = prov.complete(system, messages, max_tokens, sample_id=sample_id)
        else:
            c = prov.complete(system, messages, max_tokens)
        usd = completion_usd(spec, c)
        self.budget.charge(usd)
        self.cache.put(key, dict(text=c.text, input_tokens=c.input_tokens, output_tokens=c.output_tokens,
                                 usd=usd, stop_reason=c.stop_reason, model=model, model_id=spec.model))
        self.log.write(model=model, model_id=spec.model, provider=prov.kind, tag=tag, cache_hit=False,
                       input_tokens=c.input_tokens, output_tokens=c.output_tokens, usd=usd, usd_nominal=usd,
                       estimate=est, spent=self.budget.spent, stop_reason=c.stop_reason,
                       latency_s=round(time.time() - t0, 3), key=key[:16])
        return CallResult(c.text, c.input_tokens, c.output_tokens, usd, usd, False, c.stop_reason)


def build_client(cfg: LLMConfig, pool: list, dry_run: bool, budget_usd: Optional[float] = None,
                 cache_dir: Optional[str] = None, log_path: Optional[str] = None,
                 check_fn=None) -> LLMClient:
    budget = Budget(cfg.run["budget_usd"] if budget_usd is None else budget_usd)
    cdir = cache_dir or cfg.resolve(cfg.run["cache_dir"])
    cache = ResponseCache(os.path.join(cdir, "responses_sim.sqlite" if dry_run else "responses.sqlite"))
    oracle = SimOracle(pool, check_fn) if dry_run else None
    return LLMClient(cfg, budget, cache, CallLog(log_path), dry_run, oracle)
