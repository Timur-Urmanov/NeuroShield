"""Task embedding x in R^D (unit norm). The embedder sees only Task.prompt, never the task kind.

hashed (offline, default in dry-run): word unigrams+bigrams and char 3-grams hashed (sha1) into
    2^14 signed buckets, tf-log weighted, then a fixed Gaussian random projection to D, L2-normalised.
openai_compatible: POST {base_url}/embeddings (OpenAI/Voyage-style), then the same fixed projection
    to D. Goes through the response cache and the budget guard like chat calls.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
import numpy as np

from .budget import estimate_tokens
from .cache import cache_key

N_BUCKETS = 2 ** 14


def _features(text: str):
    t = text.lower()
    words = re.findall(r"[a-z0-9_]+", t)
    feats = list(words) + [a + " " + b for a, b in zip(words, words[1:])]
    s = re.sub(r"\s+", " ", t)
    feats += ["#" + s[i:i + 3] for i in range(max(0, len(s) - 2))]
    return feats


def hashed_vector(text: str) -> np.ndarray:
    v = np.zeros(N_BUCKETS)
    counts: dict = {}
    for f in _features(text):
        counts[f] = counts.get(f, 0) + 1
    for f, c in counts.items():
        h = int.from_bytes(hashlib.sha1(f.encode()).digest()[:8], "big")
        v[h % N_BUCKETS] += (1.0 if (h >> 63) & 1 else -1.0) * (1.0 + math.log(c))
    return v


def _project(V: np.ndarray, dim: int, seed: int) -> np.ndarray:
    P = np.random.default_rng(seed).normal(size=(V.shape[1], dim)) / math.sqrt(dim)
    X = V @ P
    X /= np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    return X


class HashedEmbedder:
    def __init__(self, dim: int = 16, projection_seed: int = 12345, **_):
        self.dim, self.seed = int(dim), int(projection_seed)

    def embed(self, texts: list) -> np.ndarray:
        V = np.stack([hashed_vector(t) for t in texts])
        V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        return _project(V, self.dim, self.seed)


class OpenAICompatEmbedder:
    def __init__(self, base_url: str, model: str, api_key_env: str, dim: int = 16, projection_seed: int = 12345,
                 price_in_per_1m: float = 0.0, budget=None, cache=None, log=None, batch: int = 64, **_):
        self.base_url, self.model, self.api_key_env = base_url.rstrip("/"), model, api_key_env
        self.dim, self.seed, self.price = int(dim), int(projection_seed), float(price_in_per_1m)
        self.budget, self.cache, self.log, self.batch = budget, cache, log, batch

    def _fetch(self, texts):
        from .providers import ProviderError, _post_json
        key = os.environ.get(self.api_key_env or "")
        if not key:
            raise ProviderError(f"embeddings: environment variable {self.api_key_env!r} is not set")
        d = _post_json(self.base_url + "/embeddings", {"model": self.model, "input": texts},
                       {"authorization": f"Bearer {key}"})
        vecs = [r["embedding"] for r in sorted(d["data"], key=lambda r: r.get("index", 0))]
        tok = int((d.get("usage") or {}).get("total_tokens") or sum(estimate_tokens(len(t)) for t in texts))
        return vecs, tok

    def embed(self, texts: list) -> np.ndarray:
        out = [None] * len(texts)
        ident = dict(provider="embed:openai_compatible", base_url=self.base_url, model=self.model)
        todo = []
        for i, t in enumerate(texts):
            k = cache_key(ident, None, [{"role": "user", "content": t}], {})
            hit = self.cache.get(k) if self.cache else None
            if hit is not None:
                out[i] = hit["vec"]
            else:
                todo.append((i, k, t))
        for j in range(0, len(todo), self.batch):
            chunk = todo[j:j + self.batch]
            est = self.price * sum(estimate_tokens(len(t)) for _, _, t in chunk) / 1e6
            if self.budget:
                self.budget.check(est, "embeddings")
            vecs, tok = self._fetch([t for _, _, t in chunk])
            usd = self.price * tok / 1e6
            if self.budget:
                self.budget.charge(usd)
            if self.log:
                self.log.write(model=self.model, tag="embed", cache_hit=False, input_tokens=tok, usd=usd)
            for (i, k, _), v in zip(chunk, vecs):
                out[i] = v
                if self.cache:
                    self.cache.put(k, dict(vec=v, text="", input_tokens=0, output_tokens=0, usd=0.0))
        V = np.array(out, dtype=float)
        V /= np.maximum(np.linalg.norm(V, axis=1, keepdims=True), 1e-12)
        return _project(V, self.dim, self.seed)


def make_embedder(emb_cfg: dict, dry_run: bool, budget=None, cache=None, log=None):
    prov = emb_cfg.get("provider", "hashed")
    if dry_run or prov == "hashed":
        return HashedEmbedder(**emb_cfg)
    if prov == "openai_compatible":
        return OpenAICompatEmbedder(budget=budget, cache=cache, log=log,
                                    **{k: v for k, v in emb_cfg.items() if k != "provider"})
    raise ValueError(f"unknown embeddings provider {prov!r}")
