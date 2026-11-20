"""
NOTE: pinned to recruit_rule="independent", w_prior="w0" (defaults before H9d).

NOTE: pinned to w_rule="hebb_r" (the default when this experiment was run; default is mean_succ since H8c).
H1b: roles env with type-dependent roles and/or busy primitives (notes/hypotheses.md)."""
import argparse

from common import base_args, paired, run_variants, save, table
from swarm import EnvConfig, SwarmConfig

ap = base_args(argparse.ArgumentParser()); ap.add_argument("--busy", type=int, default=8); a = ap.parse_args()
seeds = list(range(a.seed0, a.seed0 + a.seeds))
out = {}
for rbt in (False, True):
    for busy in (0, a.busy):
        env = EnvConfig("roles", a.T, role_by_type=rbt, busy=busy)
        V = {
            "swarm W_reward": ("swarm", env, SwarmConfig(recruit_rule="independent", w_prior="w0", w_rule="hebb_r")),
            "swarm sim-only (no W)": ("swarm", env, SwarmConfig(recruit_rule="independent", w_prior="w0", w_rule="hebb_r", plastic_W=False)),
            "linucb + random3": ("router", env, dict(model="linucb", team="random")),
            "linucb + agent_ucb": ("router", env, dict(model="linucb", team="agent_ucb", c=0.3)),
            **{f"linucb + pair_ucb c={c}": ("router", env, dict(model="linucb", team="pair_ucb", c=c))
               for c in (0.03, 0.1, 0.3)},
            **{f"linucb + cluster_pair_ucb c={c}": ("router", env, dict(model="linucb", team="cluster_pair_ucb", c=c))
               for c in (0.03, 0.1, 0.3)},
            "linucb + oracle team": ("router", env, dict(model="linucb", team="oracle")),
        }
        cell = f"role_by_type={rbt}, busy={busy}"
        print("\n##", cell)
        res = run_variants(V, seeds, a.workers)
        print(table(res, ("post", "cover_end", "q_end")))
        bandits = [k for k in res if "_ucb" in k]
        best = max(bandits, key=lambda k: res[k]["post"])
        cmp = [paired(res, "swarm W_reward", best), paired(res, "swarm W_reward", "swarm sim-only (no W)")]
        for c in cmp:
            print(c)
        out[cell] = dict(results=res, best_bandit=best, comparisons=cmp)
save("h1b_context_capacity", dict(T=a.T, seeds=seeds, cells=out))
