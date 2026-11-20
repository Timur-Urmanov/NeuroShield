"""
NOTE: pinned to w_rule="hebb_r" (the default when this experiment was run; default is mean_succ since H8c).
H5b: reward-only Hebb vs Hebb + 0.3 on fresh seeds 5..19."""
import argparse

import numpy as np

from common import base_args, paired, run_variants, save, table
from swarm import EnvConfig, SwarmConfig

ap = base_args(argparse.ArgumentParser()); ap.set_defaults(seed0=5, seeds=15); a = ap.parse_args()
seeds = list(range(a.seed0, a.seed0 + a.seeds))
env = EnvConfig("roles", a.T)
res = run_variants({"bias=0": ("swarm", env, SwarmConfig(w_rule="hebb_r", hebb_bias=0.0)),
                    "bias=0.3": ("swarm", env, SwarmConfig(w_rule="hebb_r", hebb_bias=0.3))}, seeds, a.workers)
print(table(res, ("post", "cover_end")))
c = paired(res, "bias=0", "bias=0.3"); c["se"] = c["sd"] / np.sqrt(len(seeds)); c["effect_se"] = bool(abs(c["diff"]) > 2 * c["se"])
print(c)
save("h5b_hebb_bias_seeds", dict(T=a.T, seeds=seeds, results=res, comparison=c))
