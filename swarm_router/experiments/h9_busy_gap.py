"""H9: conditional recruitment x neutral prior, base = mean_succ (notes/hypotheses.md)."""
import argparse, json, os

from common import RESULTS, base_args, paired, run_variants, save, table
from swarm import EnvConfig, SwarmConfig

ap = base_args(argparse.ArgumentParser()); a = ap.parse_args()
seeds = list(range(a.seed0, a.seed0 + a.seeds))
h1b = json.load(open(os.path.join(RESULTS, "h1b_context_capacity.json")))["cells"]
h8 = json.load(open(os.path.join(RESULTS, "h8_w_rule.json")))["cells"]
V = {"base (indep, w0)": SwarmConfig(w_rule="mean_succ"),
     "cond, w0": SwarmConfig(w_rule="mean_succ", recruit_rule="conditional"),
     "indep, mean prior": SwarmConfig(w_rule="mean_succ", w_prior="mean"),
     "cond, mean prior": SwarmConfig(w_rule="mean_succ", recruit_rule="conditional", w_prior="mean")}
NEW = [k for k in V if not k.startswith("base")]
out = {}
for rbt in (False, True):
    for busy in (0, 8):
        cell = f"role_by_type={rbt}, busy={busy}"
        env = EnvConfig("roles", a.T, role_by_type=rbt, busy=busy)
        res = run_variants({k: ("swarm", env, v) for k, v in V.items()}, seeds, a.workers)
        same = res["base (indep, w0)"]["per_seed"] == h8[cell]["results"]["mean_succ"]["per_seed"]
        for k, v in h1b[cell]["results"].items():
            if "_ucb" in k:
                res[k] = v
        best = max([k for k in res if "_ucb" in k], key=lambda k: res[k]["post"])
        cmp = [paired(res, k, "base (indep, w0)") for k in NEW] + [paired(res, k, best) for k in NEW]
        print(f"\n## {cell}   (base identical to H8 mean_succ: {same})\n"
              + table({k: res[k] for k in (*V, best)}, ("post", "cover_end", "q_end", "team_size")))
        for c in cmp:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})
        out[cell] = dict(results=res, best_bandit=best, comparisons=cmp, base_matches_h8=same)
env = EnvConfig("homogeneous", a.T)
res = run_variants({**{k: ("swarm", env, v) for k, v in V.items()},
                    "linucb": ("router", env, dict(model="linucb"))}, seeds, a.workers)
cmp = [paired(res, k, "base (indep, w0)") for k in NEW] + [paired(res, k, "linucb") for k in NEW]
print("\n## homogeneous\n" + table(res, ("post", "team_size")))
for c in cmp:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})
out["homogeneous"] = dict(results=res, comparisons=cmp)
save("h9_busy_gap", dict(T=a.T, seeds=seeds, cells=out))
