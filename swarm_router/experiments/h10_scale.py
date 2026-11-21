"""H10: swarm vs centralised baselines as the number of primitives grows; structure of the learned links.
(notes/hypotheses.md)   python3 experiments/h10_scale.py [--Ns 40,160,640,2560] [--long]"""
import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace

import numpy as np

from common import agg, base_args, save
from swarm import EnvConfig, SwarmConfig, run_swarm
from swarm.baselines import run_router, summary as rsummary
from swarm.envs import make_env

CELLS = {"roles common": dict(kind="roles"), "roles by type": dict(kind="roles", role_by_type=True),
         "by type + busy 8": dict(kind="roles", role_by_type=True, busy=8), "homogeneous": dict(kind="homogeneous")}


def gini(x):
    x = np.sort(np.asarray(x, float))
    if x.sum() == 0:
        return 0.0
    n = len(x); return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))


def link_metrics(s, env, teams, T):
    N = s.N; n = s.nW_pair; W = s.W
    p0 = s.team_succ_sum / s.team_n if s.team_n else 0.5
    iu = np.triu_indices(N, 1)
    tried = n[iu] > 0
    strong = (n[iu] >= 3) & (W[iu] > p0 + 0.1)
    out = dict(tried_pair_frac=float(tried.mean()), mean_degree=float((n > 0).sum(1).mean()),
               strong_links=int(strong.sum()), strong_per_primitive=float(2 * strong.sum() / N), p0=float(p0))
    if hasattr(env, "role_k"):
        a, b = iu[0][strong], iu[1][strong]
        out["complementarity"] = (float(np.mean([np.mean(env.role_k[k][a] != env.role_k[k][b]) for k in range(env.K)]))
                                  if strong.any() else None)
    # connected components of the strong-link graph (union-find)
    par = list(range(N))
    def f(i):
        while par[i] != i:
            par[i] = par[par[i]]; i = par[i]
        return i
    for a, b in zip(iu[0][strong], iu[1][strong]):
        ra, rb = f(a), f(b)
        if ra != rb:
            par[ra] = rb
    comp = np.bincount([f(i) for i in range(N)], minlength=N); comp = comp[comp > 1]
    out.update(n_components=int(len(comp)), largest_component=int(comp.max()) if len(comp) else 0)
    last = teams[-2000:]
    load = np.bincount(np.concatenate([np.array(t) for t in last]), minlength=N)
    out.update(active_frac=float((load > 0).mean()), load_gini=gini(load),
               distinct_teams=len({tuple(sorted(t)) for t in last}), distinct_leaders=len({t[0] for t in last}))
    return out


def job(spec):
    kind, env_cfg, payload, seed = spec
    t0 = time.time()
    if kind == "swarm":
        tr = run_swarm(env_cfg, payload, seed, keep=True)
        res = tr.summary()
        env = make_env(env_cfg, seed, payload.N)
        if env_cfg.kind != "homogeneous":
            res["links"] = link_metrics(tr.swarm, env, tr.teams, env_cfg.T)
    else:
        res = rsummary(run_router(env_cfg, seed, **payload), env_cfg.T)
    res["ms_per_step"] = 1000 * (time.time() - t0) / env_cfg.T
    return res


def main():
    ap = base_args(argparse.ArgumentParser()); ap.add_argument("--Ns", default="40,160,640,2560")
    ap.add_argument("--long", action="store_true", help="T=32000 at N in {640, 2560}, by type + busy only")
    a = ap.parse_args()
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    Ns = [int(x) for x in a.Ns.split(",")]
    specs, keys = [], []
    for cname, ck in CELLS.items():
        if a.long and cname != "by type + busy 8":
            continue
        for N in ([640, 2560] if a.long else Ns):
            T = 32000 if a.long else a.T
            env = EnvConfig(T=T, **ck)
            V = {"swarm": ("swarm", env, SwarmConfig(N=N))}
            if ck["kind"] == "homogeneous":
                V["linucb"] = ("router", env, dict(N=N, model="linucb"))
            else:
                V["pair_mean"] = ("router", env, dict(N=N, model="linucb", team="pair_mean"))
                if not a.long:
                    V["pair_ucb c=0.03"] = ("router", env, dict(N=N, model="linucb", team="pair_ucb", c=0.03))
                    V["agent_ucb"] = ("router", env, dict(N=N, model="linucb", team="agent_ucb", c=0.3))
                    V["oracle team"] = ("router", env, dict(N=N, model="linucb", team="oracle"))
            for vname, v in V.items():
                for sd in seeds:
                    specs.append((v[0], v[1], v[2], sd)); keys.append((cname, N, vname, sd))
    # checkpoint: every finished run is appended to a jsonl; a restart skips the runs already there
    ck = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache",
                      f"h10_jobs{'_long' if a.long else ''}_T{32000 if a.long else a.T}.jsonl")
    os.makedirs(os.path.dirname(ck), exist_ok=True)
    res = {}
    if os.path.exists(ck):
        for line in open(ck):
            rec = json.loads(line); res[tuple(rec["key"])] = rec["res"]
    todo = [i for i in range(len(specs)) if (keys[i][0], keys[i][1], keys[i][2], keys[i][3]) not in res]
    # heavy (large-N centralised) jobs first so the pool stays busy
    todo.sort(key=lambda i: -(keys[i][1] ** 2 if specs[i][0] == "router" else keys[i][1]))
    print(f"{len(res)} runs from checkpoint, {len(todo)} to do", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers or os.cpu_count()) as ex, open(ck, "a") as fck:
        futs = {ex.submit(job, specs[i]): i for i in todo}
        for fu in as_completed(futs):
            i = futs[fu]; r = fu.result(); k = list(keys[i])
            res[tuple(k)] = r
            fck.write(json.dumps(dict(key=k, res=r)) + "\n"); fck.flush()
            print(f"  done {k} ({time.time() - t0:.0f}s)", flush=True)
    res = {k: res[k] for k in [tuple(x) for x in keys]}
    print(f"[{len(specs)} runs, {len(todo)} new in {time.time() - t0:.0f}s]")
    out = {}
    for (c, N, v, sd), r in res.items():
        out.setdefault(c, {}).setdefault(str(N), {}).setdefault(v, []).append((sd, r))
    table = {}
    for c, byN in out.items():
        for N, byV in byN.items():
            for v, rows in byV.items():
                rows = [r for _, r in sorted(rows, key=lambda x: x[0])]
                e = dict(per_seed=rows, **agg(rows), ms_per_step=float(np.mean([r["ms_per_step"] for r in rows])))
                if "links" in rows[0]:
                    e["links"] = {k: (float(np.mean([r["links"][k] for r in rows if r["links"][k] is not None]))
                                      if any(r["links"][k] is not None for r in rows) else None) for k in rows[0]["links"]}
                table.setdefault(c, {}).setdefault(N, {})[v] = e
    for c, byN in table.items():
        print(f"\n## {c}")
        for N, byV in sorted(byN.items(), key=lambda x: int(x[0])):
            line = "  ".join(f"{v}={e['post']:.3f}±{e['post_sd']:.3f}" for v, e in byV.items())
            print(f"N={N:>5}: {line}   [swarm {byV['swarm']['ms_per_step']:.2f} ms/step]")
    save("h10_scale_long" if a.long else "h10_scale", dict(T=32000 if a.long else a.T, seeds=seeds, table=table))


if __name__ == "__main__":
    main()
