"""
NOTE: pinned to w_rule="hebb_r" (the default when this experiment was run; default is mean_succ since H8c).
H8c: does w_rule=mean_succ keep parity in the homogeneous env? (notes/hypotheses.md)"""
import argparse

from common import base_args, paired, run_variants, save, table
from swarm import EnvConfig, SwarmConfig

ap = base_args(argparse.ArgumentParser()); a = ap.parse_args()
seeds = list(range(a.seed0, a.seed0 + a.seeds))
env = EnvConfig("homogeneous", a.T)
V = {"hebb_r": ("swarm", env, SwarmConfig(w_rule="hebb_r")),
     "mean_succ": ("swarm", env, SwarmConfig(w_rule="mean_succ")),
     "mean_r": ("swarm", env, SwarmConfig(w_rule="mean_r")),
     "no W": ("swarm", env, SwarmConfig(w_rule="hebb_r", plastic_W=False)),
     "solo": ("swarm", env, SwarmConfig(w_rule="hebb_r", coupling=False, plastic_W=False)),
     "linucb": ("router", env, dict(model="linucb"))}
res = run_variants(V, seeds, a.workers)
print(table(res, ("pre", "post", "team_size")))
cmp = [paired(res, "mean_succ", "hebb_r"), paired(res, "mean_succ", "linucb"), paired(res, "hebb_r", "linucb"),
       paired(res, "mean_succ", "no W"), paired(res, "mean_r", "hebb_r")]
for c in cmp:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})
save("h8c_homogeneous", dict(T=a.T, seeds=seeds, results=res, comparisons=cmp))
