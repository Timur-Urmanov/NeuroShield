"""H8: W learning rule, 2x2 estimator {hebb, pair mean} x signal {r, success} (notes/hypotheses.md)."""
import argparse, json, os

import numpy as np

from common import RESULTS, base_args, paired, run_variants, save, table
from swarm import EnvConfig, SwarmConfig

ap = base_args(argparse.ArgumentParser()); a = ap.parse_args()
seeds = list(range(a.seed0, a.seed0 + a.seeds))
h1b = json.load(open(os.path.join(RESULTS, "h1b_context_capacity.json")))["cells"]
RULES = ("hebb_r", "hebb_succ", "mean_r", "mean_succ")
out, attr = {}, {"estimator": [], "signal": []}
for rbt in (False, True):
    for busy in (0, 8):
        cell = f"role_by_type={rbt}, busy={busy}"
        env = EnvConfig("roles", a.T, role_by_type=rbt, busy=busy)
        res = run_variants({r: ("swarm", env, SwarmConfig(w_rule=r)) for r in RULES}, seeds, a.workers)
        for k, v in h1b[cell]["results"].items():
            if "_ucb" in k:
                res[k] = v
        best = max([k for k in res if "_ucb" in k], key=lambda k: res[k]["post"])
        pair_best = max([k for k in res if "linucb + pair_ucb" in k], key=lambda k: res[k]["post"])
        cmp = [paired(res, "mean_succ", "hebb_r"), paired(res, "mean_succ", best), paired(res, "mean_succ", pair_best),
               paired(res, "mean_r", "hebb_r"), paired(res, "hebb_succ", "hebb_r"), paired(res, "mean_succ", "mean_r")]
        attr["estimator"].append(np.mean([res["mean_r"]["post"] - res["hebb_r"]["post"],
                                          res["mean_succ"]["post"] - res["hebb_succ"]["post"]]))
        attr["signal"].append(np.mean([res["hebb_succ"]["post"] - res["hebb_r"]["post"],
                                       res["mean_succ"]["post"] - res["mean_r"]["post"]]))
        print(f"\n## {cell}\n" + table({k: res[k] for k in (*RULES, best)}, ("post", "cover_end", "q_end", "team_size")))
        for c in cmp:
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})
        out[cell] = dict(results=res, best_bandit=best, comparisons=cmp)
attr = {k: [float(x) for x in v] for k, v in attr.items()}
print("\nattribution (per cell):", {k: [round(x, 3) for x in v] for k, v in attr.items()})
save("h8_w_rule", dict(T=a.T, seeds=seeds, cells=out, attribution=attr))
