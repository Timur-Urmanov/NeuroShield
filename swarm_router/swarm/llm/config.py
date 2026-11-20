"""YAML config for the LLM phase. Prices and model ids come from the config file only."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass, field
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_CONFIG = os.path.join(ROOT, "configs", "llm.example.yaml")

DEFAULT_RUN = dict(budget_usd=5.0, lam=0.15, cost_norm_usd=0.01,
                   max_tokens=dict(planner=1024, executor=2048, verifier=2048),
                   cache_dir=".cache/llm", log_dir="results/llm_calls",
                   include_seed_in_cache_key=True, team_roles="position",
                   variant_budget_share="equal")
DEFAULT_EMB = dict(dim=16, projection_seed=12345, provider="hashed")
DEFAULT_TASKS = dict(builtin=True, external_dir="data", code_timeout_s=5.0, code_mem_mb=512)
DEFAULT_SIM = dict(acc=dict(math=0.8, code=0.8, text=0.8),
                   role_skill=dict(none=0.0, executor=0.0, planner=-0.3, verifier=-0.3),
                   out_tokens=dict(planner=200, executor=300, verifier=80),
                   plan_bonus=0.0, verify_detect=0.5, false_reject=0.1, no_verdict=0.0,
                   context_decay_per_1k=0.0)


@dataclass
class ModelSpec:
    name: str
    provider: str                     # anthropic | openai_compatible | sim
    model: str
    price_in_per_1m: float
    price_out_per_1m: float
    api_key_env: Optional[str] = None
    base_url: Optional[str] = None
    params: dict = field(default_factory=dict)
    sim: dict = field(default_factory=dict)
    max_tokens_field: str = "max_tokens"
    thinking: bool = False            # model may emit (billed) thinking tokens -> conservative estimate

    def usd(self, input_tokens: float, output_tokens: float) -> float:
        return (input_tokens * self.price_in_per_1m + output_tokens * self.price_out_per_1m) / 1e6


@dataclass
class LLMConfig:
    models: list
    run: dict
    embeddings: dict
    tasks: dict
    path: Optional[str] = None
    raw: dict = field(default_factory=dict)

    def model(self, name: str) -> ModelSpec:
        for m in self.models:
            if m.name == name:
                return m
        raise KeyError(name)

    @property
    def names(self):
        return [m.name for m in self.models]

    def resolve(self, p: str) -> str:
        return p if os.path.isabs(p) else os.path.join(ROOT, p)


def _merge(base: dict, over: Optional[dict]) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def from_dict(d: dict, path: Optional[str] = None) -> LLMConfig:
    models = []
    for m in d.get("models") or []:
        m = dict(m)
        m["sim"] = _merge(DEFAULT_SIM, m.get("sim"))
        m["params"] = m.get("params") or {}
        models.append(ModelSpec(**m))
    if not models:
        raise ValueError("config has no models")
    if len({m.name for m in models}) != len(models):
        raise ValueError("model names must be unique")
    return LLMConfig(models=models, run=_merge(DEFAULT_RUN, d.get("run")),
                     embeddings=_merge(DEFAULT_EMB, d.get("embeddings")),
                     tasks=_merge(DEFAULT_TASKS, d.get("tasks")), path=path, raw=copy.deepcopy(d))


def load_config(path: Optional[str] = None) -> LLMConfig:
    import yaml
    path = path or DEFAULT_CONFIG
    with open(path) as f:
        return from_dict(yaml.safe_load(f), path)
