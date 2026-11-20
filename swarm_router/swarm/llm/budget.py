"""Hard per-run USD budget + jsonl call log.

Guard (before every non-cached call):
    estimate = chars/4 * price_in + max_tokens * price_out        (worst case on output)
    if spent + estimate > limit: raise BudgetExceeded            (call is NOT made)
After the call the *actual* usage reported by the provider is charged.
Because the output side is bounded by max_tokens, the real spend can never exceed the limit
(up to the chars/4 input-token approximation).
"""
from __future__ import annotations

import json
import os
import time
from typing import Optional


class BudgetExceeded(RuntimeError):
    def __init__(self, msg: str, scope: str = "run"):
        super().__init__(msg)
        self.scope = scope            # "run" (whole-run limit) or "share" (a variant's share)


def estimate_tokens(text_chars: int) -> int:
    return int(text_chars / 4) + 1


def prompt_chars(system: Optional[str], messages: list) -> int:
    n = len(system or "")
    for m in messages:
        c = m["content"]
        n += len(c) if isinstance(c, str) else len(json.dumps(c))
    return n


class Budget:
    """Hard limit. A child budget (`share()`) has its own limit and also charges/checks its parent,
    so a variant can be capped at its share without ever exceeding the run limit."""

    def __init__(self, limit_usd: float, parent: "Optional[Budget]" = None, name: str = "run"):
        self.limit = float(limit_usd)
        self.parent = parent
        self.name = name
        self.spent = 0.0
        self.n_calls = 0

    @property
    def remaining(self) -> float:
        own = self.limit - self.spent
        return min(own, self.parent.remaining) if self.parent else own

    def check(self, estimate_usd: float, what: str = ""):
        if self.parent is not None:
            self.parent.check(estimate_usd, what)
        if self.spent + estimate_usd > self.limit:
            raise BudgetExceeded(f"budget '{self.name}' {self.limit:.4f} USD: spent {self.spent:.4f} + estimate "
                                 f"{estimate_usd:.4f} for {what} would exceed the limit",
                                 scope="run" if self.parent is None else "share")

    def charge(self, usd: float):
        self.spent += float(usd)
        self.n_calls += 1
        if self.parent is not None:
            self.parent.charge(usd)

    def share(self, limit_usd: float, name: str) -> "Budget":
        return Budget(min(float(limit_usd), self.remaining), parent=self, name=name)

    @property
    def root(self) -> "Budget":
        return self.parent.root if self.parent else self


class CallLog:
    """Append-only jsonl: one line per call (cached or not)."""

    def __init__(self, path: Optional[str]):
        self.path = path
        self.f = None
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            self.f = open(path, "a", encoding="utf-8")

    def write(self, **rec):
        if self.f:
            rec.setdefault("ts", time.time())
            self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self.f.flush()

    def close(self):
        if self.f:
            self.f.close()
