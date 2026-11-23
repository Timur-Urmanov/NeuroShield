"""H15d: confirm on fresh seeds 5..19 that the full swarm at N=640 beats every tuned bandit at every N in
'roles by type + busy 48' (notes/hypotheses.md). Bandit constants c are those chosen in H15 on dev seeds."""
import json, os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from common import RESULTS, ROOT, save
from h15_capacity import BUSY, job
from swarm import EnvConfig, SwarmConfig

CELL = "roles by type + busy 48"
SEEDS = list(range(5, 20))

if __name__ == "__main__":
    chosen = json.load(open(os.path.join(RESULTS, "h15_capacity.json")))["chosen_c"]
    env = EnvConfig("roles", 8000, role_by_type=True, busy=BUSY)
    V = {"swarm full@640": ("swarm", SwarmConfig(N=640, inherit="team", inherit_frac=0.02, recruit_agg="min",
                                                 sim_weight="evidence"))}
    for N in (160, 640, 2560):
        V[f"agent_ucb@{N}"] = ("router", dict(N=N, model="linucb", team="agent_ucb", c=chosen[f"{CELL}|{N}|agent_ucb"]["c"]))
        V[f"pair_mean@{N}"] = ("router", dict(N=N, model="linucb", team="pair_mean"))
        if N <= 640:
            V[f"pair_ucb@{N}"] = ("router", dict(N=N, model="linucb", team="pair_ucb", c=chosen[f"{CELL}|{N}|pair_ucb"]["c"]))
    ck = os.path.join(ROOT, ".cache", "h15d.jsonl"); res = {}
    if os.path.exists(ck):
        for line in open(ck):
            r = json.loads(line); res[tuple(r["key"])] = r["post"]
    todo = [(v, sd) for v in V for sd in SEEDS if (v, sd) not in res]
    todo.sort(key=lambda k: -int(k[0].split("@")[1]) ** (2 if "pair" in k[0] else 1))
    print(f"{len(res)} from checkpoint, {len(todo)} to do", flush=True)
    with ProcessPoolExecutor(max_workers=2) as ex, open(ck, "a") as f:
        futs = {ex.submit(job, (V[v][0], env, V[v][1], sd)): (v, sd) for v, sd in todo}
        for fu in as_completed(futs):
            k = futs[fu]; res[k] = fu.result()["post"]
            f.write(json.dumps(dict(key=list(k), post=res[k])) + "\n"); f.flush()
    s = np.array([res[("swarm full@640", sd)] for sd in SEEDS]); out = {}
    for v in V:
        if v.startswith("swarm"):
            continue
        d = s - np.array([res[(v, sd)] for sd in SEEDS]); se = d.std(ddof=1) / np.sqrt(len(d))
        out[v] = dict(bandit=float(np.mean([res[(v, sd)] for sd in SEEDS])), diff=float(d.mean()), sd=float(d.std(ddof=1)),
                      se=float(se), effect_se=bool(d.mean() > 2 * se), effect_2sd=bool(d.mean() > 2 * d.std(ddof=1)),
                      positive_seeds=int((d > 0).sum()))
        print(v, {k: (round(x, 4) if isinstance(x, float) else x) for k, x in out[v].items()}, flush=True)
    print("swarm full@640:", round(float(s.mean()), 4), "CONFIRMED" if all(o["effect_se"] for o in out.values()) else "NOT CONFIRMED")
    save("h15d_confirm", dict(seeds=SEEDS, swarm_full_640=s.tolist(), comparisons=out,
                              confirmed=bool(all(o["effect_se"] for o in out.values()))))
