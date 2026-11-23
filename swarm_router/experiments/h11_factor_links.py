"""H11: computed links (primitive vectors) vs the pair matrix, across N (notes/hypotheses.md).
  --stage dev                       lr sweep on dev seeds 100-104, N=160
  --stage test --lr k=v,...         chosen lr per variant on seeds 0-4, N in 40..2560 (checkpointed)"""
import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from common import ROOT, agg, base_args, save
from swarm import EnvConfig, SwarmConfig, run_swarm
from swarm.baselines import run_router, summary as rsummary
from swarm.envs import make_env

CELLS = {"roles common": dict(kind="roles"), "roles by type": dict(kind="roles", role_by_type=True),
         "by type + busy 8": dict(kind="roles", role_by_type=True, busy=8), "homogeneous": dict(kind="homogeneous")}
LRS = (0.02, 0.05, 0.1, 0.2)
SW = {"factor": "factor", "factor_ctx": "factor_ctx"}
CE = {"mf": "mf", "mf_ctx": "mf_ctx"}


def gini(x):
    x = np.sort(np.asarray(x, float)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum())) if x.sum() else 0.0


def structure(s, env, teams):
    """Complementarity of the N/2 strongest predicted links (chance = 2/3 with equal role shares)."""
    N = s.N; iu = np.triu_indices(N, 1); K = max(1, N // 2)
    p0 = s.team_succ_sum / s.team_n if s.team_n else 0.5
    rng = np.random.default_rng(0)
    comp = []
    for k in range(env.K):
        if s.fl is not None:
            xk = env.X[env.ks == k].mean(0); xk /= np.linalg.norm(xk)
            w = s.fl.full(xk, p0)[iu]
        else:
            w = np.where(s.nW_pair[iu] > 0, s.W[iu], p0)
        w = w + 1e-9 * rng.random(len(w))                      # random tie-break among untried pairs
        top = np.argpartition(-w, K)[:K]
        comp.append(float(np.mean(env.role_k[k][iu[0][top]] != env.role_k[k][iu[1][top]])))
    last = teams[-2000:]
    load = np.bincount(np.concatenate([np.array(t) for t in last]), minlength=N)
    return dict(top_link_complementarity=float(np.mean(comp)), active_frac=float((load > 0).mean()),
                load_gini=gini(load), distinct_leaders=len({t[0] for t in last}),
                distinct_teams=len({tuple(sorted(t)) for t in last}))


def job(spec):
    kind, env_cfg, payload, seed = spec
    t0 = time.time()
    if kind == "swarm":
        tr = run_swarm(env_cfg, payload, seed, keep=True)
        res = tr.summary()
        if env_cfg.kind != "homogeneous":
            res["links"] = structure(tr.swarm, make_env(env_cfg, seed, payload.N), tr.teams)
    else:
        res = rsummary(run_router(env_cfg, seed, **payload), env_cfg.T)
    res["ms_per_step"] = 1000 * (time.time() - t0) / env_cfg.T
    return res


def run(specs, keys, ck, workers):
    res = {}
    if os.path.exists(ck):
        for line in open(ck):
            rec = json.loads(line); res[tuple(rec["key"])] = rec["res"]
    todo = [i for i in range(len(specs)) if tuple(keys[i]) not in res]
    todo.sort(key=lambda i: -(keys[i][1] ** 2 if specs[i][0] == "router" else keys[i][1]))
    print(f"{len(res)} runs from checkpoint, {len(todo)} to do", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as ex, open(ck, "a") as f:
        futs = {ex.submit(job, specs[i]): i for i in todo}
        for fu in as_completed(futs):
            i = futs[fu]; res[tuple(keys[i])] = fu.result()
            f.write(json.dumps(dict(key=list(keys[i]), res=res[tuple(keys[i])])) + "\n"); f.flush()
            print(f"  done {list(keys[i])} ({time.time() - t0:.0f}s)", flush=True)
    return res


def collect(res):
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
                    e["links"] = {k: float(np.mean([r["links"][k] for r in rows])) for k in rows[0]["links"]}
                table.setdefault(c, {}).setdefault(N, {})[v] = e
    return table


def main():
    ap = base_args(argparse.ArgumentParser()); ap.add_argument("--stage", choices=["dev", "test"], required=True)
    ap.add_argument("--lr", default=""); ap.add_argument("--Ns", default="40,160,640,2560")
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, ".cache"), exist_ok=True)
    specs, keys = [], []
    if a.stage == "dev":
        seeds = list(range(100, 105)); N = 160
        for cname in ("roles common", "by type + busy 8"):
            env = EnvConfig(T=a.T, **CELLS[cname])
            for lr in LRS:
                for v, rule in SW.items():
                    for sd in seeds:
                        specs.append(("swarm", env, SwarmConfig(N=N, w_rule=rule, factor_lr=lr), sd)); keys.append((cname, N, f"{v} lr={lr}", sd))
                for v, tm in CE.items():
                    for sd in seeds:
                        specs.append(("router", env, dict(N=N, model="linucb", team=tm, lr=lr), sd)); keys.append((cname, N, f"{v} lr={lr}", sd))
        table = collect(run(specs, keys, os.path.join(ROOT, ".cache", f"h11_dev_T{a.T}.jsonl"), a.workers))
        best = {}
        for v in (*SW, *CE):
            score = {lr: np.mean([table[c][str(N)][f"{v} lr={lr}"]["post"] for c in table]) for lr in LRS}
            best[v] = max(score, key=score.get)
            print(v, {lr: round(float(s), 3) for lr, s in score.items()}, "->", best[v])
        save("h11_dev", dict(T=a.T, seeds=seeds, N=N, table=table, best_lr=best))
        print("--lr " + ",".join(f"{k}={v}" for k, v in best.items()))
        return
    lr = {k: float(v) for k, v in (x.split("=") for x in a.lr.split(","))}
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    for cname, ck_ in CELLS.items():
        for N in [int(x) for x in a.Ns.split(",")]:
            env = EnvConfig(T=a.T, **ck_)
            V = {"matrix": ("swarm", env, SwarmConfig(N=N)),
                 "factor": ("swarm", env, SwarmConfig(N=N, w_rule="factor", factor_lr=lr["factor"])),
                 "factor_ctx": ("swarm", env, SwarmConfig(N=N, w_rule="factor_ctx", factor_lr=lr["factor_ctx"]))}
            # centralised analogues only up to N=640: at N=2560 one run costs ~40 min of CPU (amendment
            # recorded in notes/hypotheses.md before any test result was looked at)
            if ck_["kind"] != "homogeneous" and N <= 640:
                V["mf"] = ("router", env, dict(N=N, model="linucb", team="mf", lr=lr["mf"]))
                V["mf_ctx"] = ("router", env, dict(N=N, model="linucb", team="mf_ctx", lr=lr["mf_ctx"]))
            for v, s in V.items():
                for sd in seeds:
                    specs.append((s[0], s[1], s[2], sd)); keys.append((cname, N, v, sd))
    res = run(specs, keys, os.path.join(ROOT, ".cache", f"h11_test_T{a.T}.jsonl"), a.workers)
    table = collect({k: res[k] for k in [tuple(x) for x in keys]})
    for c, byN in table.items():
        print(f"\n## {c}")
        for N, byV in sorted(byN.items(), key=lambda x: int(x[0])):
            print(f"N={N:>5}: " + "  ".join(f"{v}={e['post']:.3f}±{e['post_sd']:.3f}" for v, e in byV.items()))
    save("h11_factor_links", dict(T=a.T, seeds=seeds, lr=lr, table=table))


if __name__ == "__main__":
    main()
