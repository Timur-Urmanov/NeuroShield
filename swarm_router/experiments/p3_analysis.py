"""P3-local analysis (notes/hypotheses.md): paired comparisons over seeds (project rule) and over items,
pipeline vs others on the same first-N tasks, success by kind, model shares, per-item oracle ceiling."""
import json, os, sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "results", "p3_local.json")))
V = d["variants"]; best = "single:" + d["best_single"]; pipe = [k for k in V if k.startswith("pipeline:")][0]
seeds = [r["seed"] for r in V["swarm"]["per_seed"]]
lam, norm = d["config"]["run"]["lam"], d["config"]["run"]["cost_norm_usd"]


def per_seed(k, n=None, key=1):
    return np.array([np.mean([it[key] for it in r["items"][:n]]) for r in V[k]["per_seed"]])


def paired(a, b, n=None, key=1):
    x = per_seed(a, n, key) - per_seed(b, n, key)
    sd = x.std(ddof=1)
    # item-level: pair the i-th task of the same seed (same item), pool over seeds
    di = np.concatenate([[ia[key] - ib[key] for ia, ib in zip(ra["items"][:n], rb["items"][:n])]
                         for ra, rb in zip(V[a]["per_seed"], V[b]["per_seed"])])
    se = di.std(ddof=1) / np.sqrt(len(di))
    return dict(a=a, b=b, n_per_seed=n, diff=round(float(x.mean()), 4), sd=round(float(sd), 4),
                effect_2sd=bool(abs(x.mean()) > 2 * sd), item_diff=round(float(di.mean()), 4),
                item_se=round(float(se), 4), effect_item=bool(abs(di.mean()) > 2 * se))


out = {"paired": [paired("swarm", "linucb"), paired("swarm", best), paired("linucb", best),
                  paired("swarm", "swarm_solo")]}
NP = 100
out["pipeline_subset"] = [paired("swarm", pipe, NP), paired(best, pipe, NP), paired("swarm", best, NP),
                          paired("swarm", pipe, NP, key=2), paired(best, pipe, NP, key=2)]


def usd_first(k, n):           # nominal cost per task on the first n tasks (reward = succ - lam*usd/norm)
    return float(np.mean([np.mean([(it[2] - it[1]) * norm / lam for it in r["items"][:n]]) for r in V[k]["per_seed"]]))


out["cost_first100"] = {k: usd_first(k, NP) for k in ("swarm", "linucb", best, pipe)}
out["late_window"] = [paired("swarm", best, None) | {"note": "all 300"}]
x = []
for ra, rb in zip(V["swarm"]["per_seed"], V[best]["per_seed"]):
    x.append(np.mean([i[1] for i in ra["items"][200:]]) - np.mean([i[1] for i in rb["items"][200:]]))
out["late_window"].append(dict(window="tasks 200-299", diff=round(float(np.mean(x)), 4), sd=round(float(np.std(x, ddof=1)), 4)))
out["success_by_kind"] = {k: {kk: round(float(np.mean([r["success_by_kind"][kk] for r in V[k]["per_seed"]])), 3)
                              for kk in ("math", "code", "text")} for k in V}
out["model_share"] = {k: {m: round(float(np.mean([r["model_share"][m] for r in V[k]["per_seed"]])), 3)
                          for m in V[k]["per_seed"][0]["model_share"]} for k in ("swarm", "swarm_solo", "linucb")}
# per-item oracle: best single model per item ex post (reward), averaged over the stream of each seed
singles = [k for k in V if k.startswith("single:")]
orc = []
for i_sd in range(len(seeds)):
    rows = [V[k]["per_seed"][i_sd]["items"] for k in singles]
    orc.append(np.mean([max(r[t][1] for r in rows) for t in range(len(rows[0]))]))
out["oracle_per_item"] = dict(reward=round(float(np.mean(orc)), 4), sd=round(float(np.std(orc, ddof=1)), 4))
print(json.dumps(out, indent=1))
json.dump(out, open(os.path.join(ROOT, "results", "p3_local_analysis.json"), "w"), indent=1)
