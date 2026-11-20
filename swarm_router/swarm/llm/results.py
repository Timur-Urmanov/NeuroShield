"""Result files + aggregation for the LLM experiments (same conventions as experiments/common.py:
experiment name, git commit, argv, timestamp, full config, seeds)."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

from .config import ROOT


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def save(name: str, payload: dict, out_dir: str, argv=None) -> str:
    os.makedirs(out_dir, exist_ok=True)
    payload = dict(experiment=name, git_commit=git_commit(), argv=argv or sys.argv,
                   created=time.strftime("%Y-%m-%dT%H:%M:%S"), **payload)
    path = os.path.join(out_dir, f"{name}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False, default=_default)
    print("saved", os.path.relpath(path, ROOT) if path.startswith(ROOT) else path)
    return path


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def mean_sd(rows: list, keys) -> dict:
    out = {}
    for k in keys:
        v = [r[k] for r in rows if r.get(k) is not None]
        if v:
            out[k] = float(np.mean(v))
            out[k + "_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return out


def config_dict(cfg) -> dict:
    return dict(path=cfg.path, models=[m.__dict__ for m in cfg.models], run=cfg.run,
                embeddings=cfg.embeddings, tasks=cfg.tasks)
