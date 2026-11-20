"""Shared experiment harness: parallel (variant x seed) runs, paired statistics, result files
that always carry the full config, seeds and git commit."""
from __future__ import annotations

import json, os, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from swarm import EnvConfig, SwarmConfig, run_swarm  # noqa: E402
from swarm.baselines import run_router, summary as router_summary  # noqa: E402

RESULTS = os.path.join(ROOT, "results")
METRICS = ("pre", "post", "cover_end", "q_end", "team_size")


def git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return None


def _job(spec):
    kind, env_cfg, payload, seed = spec
    if kind == "swarm":
        return run_swarm(env_cfg, payload, seed).summary()
    out = run_router(env_cfg, seed, **payload)
    return router_summary(out, env_cfg.T)


def run_variants(variants: dict, seeds, workers=None):
    """variants: name -> ("swarm", EnvConfig, SwarmConfig) | ("router", EnvConfig, kwargs)."""
    specs = [(v[0], v[1], v[2], sd) for v in variants.values() for sd in seeds]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as ex:
        res = list(ex.map(_job, specs, chunksize=1))
    out, i = {}, 0
    for name, v in variants.items():
        rows = res[i:i + len(seeds)]; i += len(seeds)
        out[name] = dict(kind=v[0],
                         env=_cfgdict(v[1]),
                         cfg=_cfgdict(v[2]) if v[0] == "swarm" else v[2],
                         per_seed=rows, **agg(rows))
    print(f"[{len(specs)} runs in {time.time() - t0:.0f}s]", flush=True)
    return out


def _cfgdict(c):
    d = asdict(c)
    if d.get("role_fractions") is not None:
        d["role_fractions"] = list(d["role_fractions"])
    return d


def agg(rows):
    out = {}
    for k in METRICS:
        v = [r[k] for r in rows if r.get(k) is not None]
        if v:
            out[k] = float(np.mean(v)); out[k + "_sd"] = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    return out


def paired(res, a, b, metric="post"):
    """Paired difference a - b over seeds (common random numbers). Effect iff |mean| > 2 sd_diff."""
    x = np.array([r[metric] for r in res[a]["per_seed"]]); y = np.array([r[metric] for r in res[b]["per_seed"]])
    d = x - y
    m, s = float(d.mean()), float(d.std(ddof=1))
    return dict(a=a, b=b, metric=metric, diff=m, sd=s, effect=bool(abs(m) > 2 * s))


def table(res, cols=("post", "cover_end", "q_end", "team_size")):
    w = max(len(n) for n in res) + 2
    lines = [f"| {'variant':<{w}} | " + " | ".join(cols) + " |", "|" + "-" * (w + 2) + "|" + "---|" * len(cols)]
    for n, r in res.items():
        cells = []
        for c in cols:
            cells.append(f"{r[c]:.3f} ± {r[c + '_sd']:.3f}" if c in r else "—")
        lines.append(f"| {n:<{w}} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def save(name, payload, argv=None):
    os.makedirs(RESULTS, exist_ok=True)
    payload = dict(experiment=name, git_commit=git_commit(), argv=argv or sys.argv,
                   created=time.strftime("%Y-%m-%dT%H:%M:%S"), **payload)
    path = os.path.join(RESULTS, f"{name}.json")
    json.dump(payload, open(path, "w"), indent=1, ensure_ascii=False)
    print("saved", os.path.relpath(path, ROOT))
    return path


def base_args(ap):
    ap.add_argument("--T", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--workers", type=int, default=None)
    return ap
