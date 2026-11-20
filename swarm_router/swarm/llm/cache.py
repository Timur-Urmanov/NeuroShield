"""On-disk response cache (sqlite). Key = sha256 of everything that determines the response:
provider identity (provider, model id, endpoint, provider params), system prompt, messages and
all sampling params (max_tokens, sample_id, ...). A hit is free for the budget, but the entry keeps
the original token usage so rewards use the *nominal* cost and do not depend on cache state."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from typing import Optional


def cache_key(identity: dict, system: Optional[str], messages: list, params: dict) -> str:
    blob = json.dumps(dict(identity=identity, system=system or "", messages=messages, params=params),
                      sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class ResponseCache:
    def __init__(self, path: Optional[str]):
        """path: sqlite file, or None for an in-memory cache."""
        self.path = path
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self.db = sqlite3.connect(path or ":memory:")
        self.db.execute("PRAGMA journal_mode=WAL") if path else None
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT, created REAL)")
        self.db.commit()
        self.hits = self.misses = 0

    @classmethod
    def in_dir(cls, cache_dir: str, name: str = "responses.sqlite"):
        return cls(os.path.join(cache_dir, name))

    def get(self, key: str) -> Optional[dict]:
        row = self.db.execute("SELECT value FROM cache WHERE key=?", (key,)).fetchone()
        if row is None:
            self.misses += 1
            return None
        self.hits += 1
        return json.loads(row[0])

    def put(self, key: str, value: dict):
        self.db.execute("INSERT OR REPLACE INTO cache VALUES (?,?,?)",
                        (key, json.dumps(value, ensure_ascii=False), time.time()))
        self.db.commit()

    def __len__(self):
        return self.db.execute("SELECT COUNT(*) FROM cache").fetchone()[0]

    def close(self):
        self.db.close()
