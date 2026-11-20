"""Post-hoc diagnostic after H7 (not pre-registered as a hypothesis test): how much of the gap to the
pair bandit is due to the leader-centred structure itself? recruit="oracle" keeps the swarm's leader
choice and model choice but fills the two missing roles with free primitives (hidden roles)."""
import argparse, json, os

from common import RESULTS, base_args, paired, run_variants, save, table
from swarm import EnvConfig, SwarmConfig

ap = base_args(argparse.ArgumentParser()); a = ap.parse_args()
seeds = list(range(a.seed0, a.seed0 + a.seeds))
h1b = json.load(open(os.path.join(RESULTS, "h1b_context_capacity.json")))["cells"]
out = {}
for rbt in (False, True):
    for busy in (0, 8):
        cell = f"role_by_type={rbt}, busy={busy}"
        env = EnvConfig("roles", a.T, role_by_type=rbt, busy=busy)
        res = run_variants({"swarm learned": ("swarm", env, SwarmConfig()),
                            "swarm oracle-recruit": ("swarm", env, SwarmConfig(recruit="oracle")),
                            "linucb + oracle team": ("router", env, dict(model="linucb", team="oracle"))}, seeds, a.workers)
        best = h1b[cell]["best_bandit"]; res[best] = h1b[cell]["results"][best]
        cmp = [paired(res, "swarm oracle-recruit", best), paired(res, "swarm oracle-recruit", "linucb + oracle team"),
               paired(res, "swarm oracle-recruit", "swarm learned")]
        print(f"\n## {cell}\n" + table(res, ("post", "cover_end", "q_end")))
        for c in cmp:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})
        out[cell] = dict(results=res, comparisons=cmp)
save("h7_posthoc_leader_ceiling", dict(T=a.T, seeds=seeds, cells=out))
