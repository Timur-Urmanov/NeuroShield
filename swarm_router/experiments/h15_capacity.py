"""H15: high capacity pressure (busy=48): does scale help, and does the full swarm beat bandits at any N?
(notes/hypotheses.md). Checkpointed."""
import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from common import ROOT, agg, base_args, save
from h13_team_selection import job as swarm_job
from swarm import EnvConfig, SwarmConfig
from swarm.baselines import run_router, summary as rsummary

BUSY = 48
CELLS = {"roles common + busy 48": dict(kind="roles", busy=BUSY),
         "roles by type + busy 48": dict(kind="roles", role_by_type=True, busy=BUSY)}


def job(spec):
    kind, env_cfg, payload, seed = spec
    if kind == "swarm":
        return swarm_job((env_cfg, payload, seed))
    t0 = time.time()
    res = rsummary(run_router(env_cfg, seed, **payload), env_cfg.T)
    res["ms_per_step"] = 1000 * (time.time() - t0) / env_cfg.T
    return res


def main():
    ap = base_args(argparse.ArgumentParser()); ap.add_argument("--Ns", default="160,640,2560")
    a = ap.parse_args()
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    specs, keys = [], []
    for cname, ck in CELLS.items():
        for N in [int(x) for x in a.Ns.split(",")]:
            env = EnvConfig(T=a.T, **ck)
            V = {"swarm orig": ("swarm", SwarmConfig(N=N)),
                 "swarm full": ("swarm", SwarmConfig(N=N, inherit="team", inherit_frac=0.02, recruit_agg="min",
                                                     sim_weight="evidence")),
                 "pair_mean": ("router", dict(N=N, model="linucb", team="pair_mean")),
                 "agent_ucb": ("router", dict(N=N, model="linucb", team="agent_ucb", c=0.3)),
                 "oracle team": ("router", dict(N=N, model="linucb", team="oracle"))}
            if N <= 640:
                V["pair_ucb c=0.03"] = ("router", dict(N=N, model="linucb", team="pair_ucb", c=0.03))
            for v, (kind, payload) in V.items():
                for sd in seeds:
                    specs.append((kind, env, payload, sd)); keys.append((cname, N, v, sd))
    ck = os.path.join(ROOT, ".cache", f"h15_T{a.T}.jsonl")
    res = {}
    if os.path.exists(ck):
        for line in open(ck):
            rec = json.loads(line); res[tuple(rec["key"])] = rec["res"]
    todo = [i for i in range(len(specs)) if tuple(keys[i]) not in res]
    todo.sort(key=lambda i: -(keys[i][1] ** 2 if specs[i][0] == "router" and "pair" in keys[i][2] else keys[i][1]))
    print(f"{len(res)} runs from checkpoint, {len(todo)} to do", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers or os.cpu_count()) as ex, open(ck, "a") as f:
        futs = {ex.submit(job, specs[i]): i for i in todo}
        for fu in as_completed(futs):
            i = futs[fu]; res[tuple(keys[i])] = fu.result()
            f.write(json.dumps(dict(key=list(keys[i]), res=res[tuple(keys[i])])) + "\n"); f.flush()
            print(f"  done {list(keys[i])} ({time.time() - t0:.0f}s)", flush=True)
    table = {}
    for (c, N, v, sd) in keys:
        table.setdefault(c, {}).setdefault(str(N), {}).setdefault(v, []).append(res[(c, N, v, sd)])
    for c in table:
        print(f"\n## {c}")
        for N in table[c]:
            for v in table[c][N]:
                rows = table[c][N][v]
                e = dict(per_seed=rows, **agg(rows))
                if "pop" in rows[0]:
                    e["pop"] = {k: float(np.mean([r["pop"][k] for r in rows])) for k in rows[0]["pop"]}
                table[c][N][v] = e
            print(f"N={N:>5}: " + "  ".join(f"{v}={e['post']:.3f}±{e['post_sd']:.3f}" for v, e in table[c][N].items()))
    save("h15_capacity", dict(T=a.T, seeds=seeds, busy=BUSY, table=table))


if __name__ == "__main__":
    main()
