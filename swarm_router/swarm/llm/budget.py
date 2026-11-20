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
    pass


def estimate_tokens(text_chars: int) -> int:
    return int(text_chars / 4) + 1


def prompt_chars(system: Optional[str], messages: list) -> int:
    n = len(system or "")
    for m in messages:
        c = m["content"]
        n += len(c) if isinstance(c, str) else len(json.dumps(c))
    return n


class Budget:
    def __init__(self, limit_usd: float):
        self.limit = float(limit_usd)
        self.spent = 0.0
        self.n_calls = 0

    @property
    def remaining(self) -> float:
        return self.limit - self.spent

    def check(self, estimate_usd: float, what: str = ""):
        if self.spent + estimate_usd > self.limit:
            raise BudgetExceeded(f"budget {self.limit:.4f} USD: spent {self.spent:.4f} + estimate "
                                 f"{estimate_usd:.4f} for {what} would exceed the limit")

    def charge(self, usd: float):
        self.spent += float(usd)
        self.n_calls += 1


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
