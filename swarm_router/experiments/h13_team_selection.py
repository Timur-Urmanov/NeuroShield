"""H13: team-level selection (notes/hypotheses.md).
  --stage dev                 rho sweep on dev seeds 100-104, N=640
  --stage test --rho X        seeds 0-4, N in 40..2560 (checkpointed); `none`/`full` come from H12 results"""
import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from common import ROOT, agg, base_args, save
from swarm import EnvConfig, SwarmConfig, run_swarm

CELLS = {"roles common": dict(kind="roles"), "roles by type": dict(kind="roles", role_by_type=True),
         "by type + busy 8": dict(kind="roles", role_by_type=True, busy=8), "homogeneous": dict(kind="homogeneous")}


def job(spec):
    env_cfg, cfg, seed = spec
    t0 = time.time()
    tr = run_swarm(env_cfg, cfg, seed, keep=True)
    res = tr.summary(); s, env = tr.swarm, tr.env
    last = tr.teams[-2000:]
    load = np.bincount(np.concatenate([np.array(t) for t in last]), minlength=s.N)
    tg = s.tag_g
    inst = [len(t) == 3 and tg[t[0]] >= 0 and tg[t[0]] == tg[t[1]] == tg[t[2]]
            and len({int(s.tag_slot[i]) for i in t}) == 3 for t in last]
    pop = dict(active_frac=float((load > 0).mean()), births=int(s.n_births), genotypes=int(len(set(tg[tg >= 0].tolist()))),
               in_genotype_frac=float((tg >= 0).mean()),
               largest_genotype_frac=float(np.bincount(tg[tg >= 0]).max() / s.N) if (tg >= 0).any() else 0.0,
               genotype_team_frac=float(np.mean(inst)), distinct_leaders=len({t[0] for t in last}))
    if env_cfg.kind != "homogeneous":
        shares = np.array([np.bincount(env.role_k[k], minlength=3) / s.N for k in range(env.K)])
        pop.update(min_role_share=float(shares.min(1).mean()))
        # share of the population sitting in genotypes whose three slots cover all roles (averaged over types)
        full = []
        for g in set(tg[tg >= 0].tolist()):
            m = np.where(tg == g)[0]
            ok = np.mean([len({int(env.role_k[k][m[s.tag_slot[m] == sl][0]]) for sl in range(3)
                               if (s.tag_slot[m] == sl).any()}) == 3 for k in range(env.K)])
            full.append(ok * len(m) / s.N)
        pop["pop_in_full_cover_genotypes"] = float(np.sum(full)) if full else 0.0
    res["pop"] = pop
    res["ms_per_step"] = 1000 * (time.time() - t0) / env_cfg.T
    return res


def run(specs, keys, ck, workers):
    res = {}
    if os.path.exists(ck):
        for line in open(ck):
            rec = json.loads(line); res[tuple(rec["key"])] = rec["res"]
    todo = sorted([i for i in range(len(specs)) if tuple(keys[i]) not in res], key=lambda i: -keys[i][1])
    print(f"{len(res)} runs from checkpoint, {len(todo)} to do", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers or os.cpu_count()) as ex, open(ck, "a") as f:
        futs = {ex.submit(job, specs[i]): i for i in todo}
        for fu in as_completed(futs):
            i = futs[fu]; res[tuple(keys[i])] = fu.result()
            f.write(json.dumps(dict(key=list(keys[i]), res=res[tuple(keys[i])])) + "\n"); f.flush()
    print(f"[{len(todo)} new runs in {time.time() - t0:.0f}s]", flush=True)
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
                table.setdefault(c, {}).setdefault(N, {})[v] = dict(
                    per_seed=rows, **agg(rows), ms_per_step=float(np.mean([r["ms_per_step"] for r in rows])),
                    pop={k: float(np.mean([r["pop"][k] for r in rows])) for k in rows[0]["pop"]})
    return table


def main():
    ap = base_args(argparse.ArgumentParser()); ap.add_argument("--stage", choices=["dev", "test"], required=True)
    ap.add_argument("--rho", type=float); ap.add_argument("--Ns", default="40,160,640,2560")
    a = ap.parse_args()
    os.makedirs(os.path.join(ROOT, ".cache"), exist_ok=True)
    specs, keys = [], []
    if a.stage == "dev":
        seeds = list(range(100, 105)); N = 640; RHO = (0.02, 0.05, 0.1)
        for cname in ("roles common", "by type + busy 8"):
            env = EnvConfig(T=a.T, **CELLS[cname])
            V = {"none": SwarmConfig(N=N)}
            V.update({f"team rho={r}": SwarmConfig(N=N, inherit="team", inherit_frac=r) for r in RHO})
            for v, cfg in V.items():
                for sd in seeds:
                    specs.append((env, cfg, sd)); keys.append((cname, N, v, sd))
        table = collect(run(specs, keys, os.path.join(ROOT, ".cache", f"h13_dev_T{a.T}.jsonl"), a.workers))
        for c in table:
            print(c, {v: (round(e["post"], 3), round(e["cover_end"], 2), round(e["pop"]["genotype_team_frac"], 2),
                          round(e["pop"]["pop_in_full_cover_genotypes"], 2)) for v, e in table[c][str(N)].items()})
        score = {r: float(np.mean([table[c][str(N)][f"team rho={r}"]["post"] for c in table])) for r in RHO}
        best = max(score, key=score.get)
        print("mean post by rho:", {k: round(v, 3) for k, v in score.items()}, "-> --rho", best)
        save("h13_dev", dict(T=a.T, seeds=seeds, N=N, table=table, best_rho=best))
        return
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    for cname, ck_ in CELLS.items():
        for N in [int(x) for x in a.Ns.split(",")]:
            env = EnvConfig(T=a.T, **ck_)
            for v in ("team", "team_nopool"):
                for sd in seeds:
                    specs.append((env, SwarmConfig(N=N, inherit=v, inherit_frac=a.rho), sd)); keys.append((cname, N, v, sd))
    table = collect(run(specs, keys, os.path.join(ROOT, ".cache", f"h13_test_T{a.T}.jsonl"), a.workers))
    for c, byN in table.items():
        print(f"\n## {c}")
        for N, byV in sorted(byN.items(), key=lambda x: int(x[0])):
            print(f"N={N:>5}: " + "  ".join(f"{v}={e['post']:.3f}±{e['post_sd']:.3f}" for v, e in byV.items()))
    save("h13_team_selection", dict(T=a.T, seeds=seeds, rho=a.rho, table=table))


if __name__ == "__main__":
    main()
