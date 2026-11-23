"""H16 addendum: context-aware (per-cluster) bandits. H15e re-checks H15d; the same families are added to the
H16b shift cells and to the H16c evolved-population control. See notes/hypotheses.md. Checkpointed.
Needs .cache/h15d.jsonl and .cache/h16_T8000.jsonl (swarm runs and the original bandit grid)."""
import json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from common import ROOT, save
from h16_shift import AGENT_C, BUSY, DEV, FULL, N_SWARM, job as base_job, modes, paired, stats
from swarm import EnvConfig, SwarmConfig, run_swarm
from swarm.baselines import run_router

T, TEST, CONF = 8000, list(range(5)), list(range(5, 20))


def job(spec):
    if spec[0] != "evolved_ctx":
        return base_job(spec)
    _, env_cfg, _, seed = spec
    tr = run_swarm(env_cfg, SwarmConfig(N=N_SWARM, **FULL), seed, keep=True)
    rk = tr.env.role_k.copy(); out = {"swarm full": stats(tr.R, tr.coverage, T)}
    V = {"cluster_pair_mean (evolved)": dict(team="cluster_pair_mean", role_k=rk),
         "cluster_pair_mean (original)": dict(team="cluster_pair_mean")}
    V.update({f"cluster_agent_ucb c={c} (evolved)": dict(team="cluster_agent_ucb", c=c, role_k=rk) for c in AGENT_C})
    for v, kw in V.items():
        o = run_router(env_cfg, seed, N=N_SWARM, model="linucb", **kw); out[v] = stats(o["R"], o["cover"], T)
    return out


def load(path):
    return {tuple(r["key"]): r.get("res", r.get("post")) for r in map(json.loads, open(path))}


if __name__ == "__main__":
    specs, keys = [], []
    env0 = EnvConfig("roles", T, role_by_type=True, busy=BUSY)
    for N in (160, 640, 2560):                                   # H15e
        fam = {f"cluster_agent_ucb|{N}|g=1|{c}": dict(team="cluster_agent_ucb", c=c) for c in AGENT_C}
        if N <= 640:
            fam[f"cluster_pair_mean|{N}|g=1|-"] = dict(team="cluster_pair_mean")
        for v, kw in fam.items():
            for sd in [*DEV, *CONF]:
                specs.append(("router", env0, dict(N=N, model="linucb", **kw), sd)); keys.append(("no shift", v, sd))
    for shift in (1.0, 0.3):                                     # H16b
        env = EnvConfig("roles", T, role_by_type=True, busy=BUSY, role_shift=shift)
        for N in (160, 640):
            for mname, mkw in modes(T).items():
                fam = {f"cluster_pair_mean|{N}|{mname}|-": dict(team="cluster_pair_mean")}
                fam.update({f"cluster_agent_ucb|{N}|{mname}|{c}": dict(team="cluster_agent_ucb", c=c) for c in AGENT_C})
                for v, kw in fam.items():
                    for sd in [*DEV, *TEST]:
                        specs.append(("router", env, dict(N=N, model="linucb", **kw, **mkw), sd))
                        keys.append((f"shift {shift}", v, sd))
    for sd in [*DEV, *TEST]:                                     # H16c
        specs.append(("evolved_ctx", env0, None, sd)); keys.append(("evolved", "ctx", sd))
    ck = os.path.join(ROOT, ".cache", "h16b.jsonl"); res = load(ck) if os.path.exists(ck) else {}
    todo = [i for i in range(len(specs)) if tuple(keys[i]) not in res]
    todo.sort(key=lambda i: -(3 if specs[i][0] != "router" else 2 if "pair" in keys[i][1] and "|640|" in keys[i][1] else 1))
    print(f"{len(res)} from checkpoint, {len(todo)} to do", flush=True); t0 = time.time()
    with ProcessPoolExecutor(max_workers=4) as ex, open(ck, "a") as f:
        futs = {ex.submit(job, specs[i]): i for i in todo}
        for n, fu in enumerate(as_completed(futs)):
            i = futs[fu]; res[tuple(keys[i])] = fu.result()
            f.write(json.dumps(dict(key=list(keys[i]), res=res[tuple(keys[i])])) + "\n"); f.flush()
            print(f"  {n + 1}/{len(todo)} {list(keys[i])} ({time.time() - t0:.0f}s)", flush=True)
    out = {}
    # ---- H15e
    h15d = load(os.path.join(ROOT, ".cache", "h15d.jsonl"))
    s = np.array([h15d[("swarm full@640", sd)] for sd in CONF]); e = {}
    names = sorted({k[1] for k in keys if k[0] == "no shift"})
    dev = {v: float(np.mean([res[("no shift", v, sd)]["post"] for sd in DEV])) for v in names}
    print("\n## H15e (no shift, seeds 5-19), swarm full@640 =", round(float(s.mean()), 4))
    for fam, Ns in (("cluster_pair_mean", (160, 640)), ("cluster_agent_ucb", (160, 640, 2560))):
        for N in Ns:
            best = max([v for v in names if v.startswith(f"{fam}|{N}|")], key=dev.get)
            b = np.array([res[("no shift", best, sd)]["post"] for sd in CONF]); d = s - b; se = d.std(ddof=1) / np.sqrt(len(d))
            e[f"{fam}@{N} [{best}]"] = dict(bandit=float(b.mean()), diff=float(d.mean()), sd=float(d.std(ddof=1)), se=float(se),
                                           effect_se=bool(d.mean() > 2 * se), effect_2sd=bool(d.mean() > 2 * d.std(ddof=1)),
                                           positive_seeds=int((d > 0).sum()))
            print(f"{fam}@{N} [{best}]", {k: (round(x, 4) if isinstance(x, float) else x) for k, x in e[f"{fam}@{N} [{best}]"].items()})
    out["H15e"] = dict(swarm_full_640=s.tolist(), comparisons=e, dev_post=dev, h15d_stands=bool(all(x["effect_se"] for x in e.values())))
    print("H15d stands:", out["H15e"]["h15d_stands"])
    # ---- H16b with the union of grids
    h16 = load(os.path.join(ROOT, ".cache", f"h16_T{T}.jsonl"))
    for shift in (1.0, 0.3):
        cell = f"shift {shift}"; allr = {k: v for k, v in h16.items() if k[0] == cell}
        allr.update({k: v for k, v in res.items() if k[0] == cell})
        col = lambda v, m, S=TEST: [allr[(cell, v, sd)][m] for sd in S]
        bandits = sorted({k[1] for k in allr if "|" in k[1]})
        dev = {v: float(np.mean(col(v, "post", DEV))) for v in bandits}
        best = max(bandits, key=dev.get); c = {}
        for fam in ("cluster_pair_mean", "cluster_agent_ucb"):
            for N in (160, 640):
                b = max([v for v in bandits if v.startswith(f"{fam}|{N}|")], key=dev.get)
                c[f"full - {fam}@{N} [{b}]"] = dict(bandit_post=float(np.mean(col(b, "post"))), bandit_rec=float(np.mean(col(b, "rec"))),
                                                    **{m: paired(col("swarm full", m), col(b, m)) for m in ("rec", "post")})
        c[f"H16b full - best bandit of all [{best}]"] = dict(bandit_post=float(np.mean(col(best, "post"))),
                                                             **{m: paired(col("swarm full", m), col(best, m)) for m in ("rec", "post")})
        out[cell] = dict(comparisons=c, dev_post=dev, best_bandit=best)
        print(f"\n## {cell} (context-aware bandits added)")
        for k, v in c.items():
            print(f"  {k}: bandit post={v['bandit_post']:.3f}; full-bandit post {v['post']['diff']:+.3f}±{v['post']['sd']:.3f} effect={v['post']['effect']}; rec {v['rec']['diff']:+.3f}±{v['rec']['sd']:.3f}")
    # ---- H16c with context-aware bandits
    ev = lambda v, S=TEST: [res[("evolved", "ctx", sd)][v]["post"] for sd in S]
    names = [v for v in res[("evolved", "ctx", 0)] if v != "swarm full"]
    dev = {v: float(np.mean(ev(v, DEV))) for v in names}
    ag = max([v for v in names if v.startswith("cluster_agent_ucb")], key=dev.get)
    c = {v: dict(bandit=float(np.mean(ev(v))), **paired(ev("swarm full"), ev(v)))
         for v in ("cluster_pair_mean (evolved)", "cluster_pair_mean (original)", ag)}
    c["cluster_pair_mean evolved - original"] = paired(ev("cluster_pair_mean (evolved)"), ev("cluster_pair_mean (original)"))
    out["evolved"] = dict(swarm_full=float(np.mean(ev("swarm full"))), comparisons=c, dev_post=dev)
    print("\n## H16c evolved population, context-aware; swarm full =", round(out["evolved"]["swarm_full"], 4))
    for k, v in c.items():
        print(f"  {k}: {v.get('bandit', float('nan')):.3f}  diff {v['diff']:+.3f}±{v['sd']:.3f} effect={v['effect']}")
    save("h16b_ctx", dict(T=T, test_seeds=TEST, confirm_seeds=CONF, dev_seeds=DEV, results=out))
