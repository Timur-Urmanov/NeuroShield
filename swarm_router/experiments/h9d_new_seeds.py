"""
NOTE: pinned to recruit_rule="independent", w_prior="w0" (defaults before H9d).
H9d: conditional recruitment + neutral prior vs mean_succ base on fresh seeds 5..19 (SE criterion)."""
import argparse

import numpy as np

from common import base_args, paired, run_variants, save, table
from swarm import EnvConfig, SwarmConfig

ap = base_args(argparse.ArgumentParser()); ap.set_defaults(seed0=5, seeds=15); a = ap.parse_args()
seeds = list(range(a.seed0, a.seed0 + a.seeds))
V = {"base": SwarmConfig(recruit_rule="independent", w_prior="w0", w_rule="mean_succ"),
     "cond, mean": SwarmConfig(w_rule="mean_succ", recruit_rule="conditional", w_prior="mean")}
cells = {f"role_by_type={r}, busy={b}": EnvConfig("roles", a.T, role_by_type=r, busy=b) for r in (False, True) for b in (0, 8)}
cells["homogeneous"] = EnvConfig("homogeneous", a.T)
out = {}
for name, env in cells.items():
    res = run_variants({k: ("swarm", env, v) for k, v in V.items()}, seeds, a.workers)
    c = paired(res, "cond, mean", "base"); c["se"] = c["sd"] / np.sqrt(len(seeds))
    c["effect_se"] = bool(abs(c["diff"]) > 2 * c["se"]); c["no_loss_se"] = bool(c["diff"] > -2 * c["se"])
    print(f"\n## {name}\n" + table(res, ("post", "cover_end")))
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})
    out[name] = dict(results=res, comparison=c)
save("h9d_new_seeds", dict(T=a.T, seeds=seeds, cells=out))
