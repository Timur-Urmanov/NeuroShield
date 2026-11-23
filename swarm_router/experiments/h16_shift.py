"""H16: adaptation after a role shift at T/2 (a, b) and the population-editing control for H15d (c).
See notes/hypotheses.md. Checkpointed in .cache/h16_T{T}.jsonl; safe to restart."""
import argparse, json, os, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from common import ROOT, base_args, save
from swarm import EnvConfig, SwarmConfig, run_swarm
from swarm.baselines import run_router

BUSY, N_SWARM = 48, 640
AGENT_C, PAIR_C, DEV = (0.01, 0.03, 0.1, 0.3), (0.01, 0.03, 0.1), list(range(100, 105))
FULL = dict(inherit="team", inherit_frac=0.02, recruit_agg="min", sim_weight="evidence")


def modes(T):
    return {"g=1": {}, "g=0.999": dict(team_gamma=0.999), "reset": dict(reset_at=T // 2)}


def stats(R, cov, T):
    return dict(pre=float(R[T // 2 - 1000:T // 2].mean()), rec=float(R[T // 2:T // 2 + 1000].mean()),
                post=float(R[-1000:].mean()), cover_end=float(np.nanmean(cov[-2000:])))


def job(spec):
    kind, env_cfg, payload, seed = spec
    T = env_cfg.T
    if kind == "swarm":
        tr = run_swarm(env_cfg, payload, seed)
        return stats(tr.R, tr.coverage, T)
    if kind == "router":
        o = run_router(env_cfg, seed, **payload)
        return stats(o["R"], o["cover"], T)
    # kind == "evolved": the swarm's final population handed to fresh centralised policies (H16c)
    tr = run_swarm(env_cfg, SwarmConfig(N=N_SWARM, **FULL), seed, keep=True)
    rk = tr.env.role_k.copy()
    out = {"swarm full": stats(tr.R, tr.coverage, T)}
    V = {"random (original)": dict(team="random"), "random (evolved)": dict(team="random", role_k=rk),
         "oracle team (evolved)": dict(team="oracle", role_k=rk), "pair_mean (evolved)": dict(team="pair_mean", role_k=rk)}
    V.update({f"agent_ucb c={c} (evolved)": dict(team="agent_ucb", c=c, role_k=rk) for c in AGENT_C})
    V.update({f"pair_ucb c={c} (evolved)": dict(team="pair_ucb", c=c, role_k=rk) for c in PAIR_C})
    for v, kw in V.items():
        o = run_router(env_cfg, seed, N=N_SWARM, model="linucb", **kw)
        out[v] = stats(o["R"], o["cover"], T)
    return out


def paired(a, b):
    d = np.array(a) - np.array(b)
    return dict(diff=float(d.mean()), sd=float(d.std(ddof=1)), effect=bool(abs(d.mean()) > 2 * d.std(ddof=1)))


def main():
    a = base_args(argparse.ArgumentParser()).parse_args()
    T, seeds = a.T, list(range(a.seed0, a.seed0 + a.seeds))
    specs, keys = [], []
    for shift in (1.0, 0.3):
        env = EnvConfig("roles", T, role_by_type=True, busy=BUSY, role_shift=shift)
        for v, (kind, payload) in {"swarm orig": ("swarm", SwarmConfig(N=N_SWARM)),
                                   "swarm full": ("swarm", SwarmConfig(N=N_SWARM, **FULL)),
                                   "oracle team": ("router", dict(N=N_SWARM, model="linucb", team="oracle")),
                                   "random": ("router", dict(N=N_SWARM, model="linucb", team="random"))}.items():
            for sd in seeds:
                specs.append((kind, env, payload, sd)); keys.append((f"shift {shift}", v, sd))
        for N in (160, 640):
            for mname, mkw in modes(T).items():
                fam = {f"pair_mean|{N}|{mname}|-": dict(team="pair_mean")}
                fam.update({f"agent_ucb|{N}|{mname}|{c}": dict(team="agent_ucb", c=c) for c in AGENT_C})
                if N == 160 or shift == 1.0:          # pair_ucb at N=640 is the slowest baseline: main cell only
                    fam.update({f"pair_ucb|{N}|{mname}|{c}": dict(team="pair_ucb", c=c) for c in PAIR_C})
                for v, kw in fam.items():
                    for sd in [*DEV, *seeds]:
                        specs.append(("router", env, dict(N=N, model="linucb", **kw, **mkw), sd))
                        keys.append((f"shift {shift}", v, sd))
    env0 = EnvConfig("roles", T, role_by_type=True, busy=BUSY)
    for sd in [*DEV, *seeds]:
        specs.append(("evolved", env0, None, sd)); keys.append(("evolved", "all", sd))

    ck = os.path.join(ROOT, ".cache", f"h16_T{T}.jsonl"); res = {}
    if os.path.exists(ck):
        for line in open(ck):
            r = json.loads(line); res[tuple(r["key"])] = r["res"]
    todo = [i for i in range(len(specs)) if tuple(keys[i]) not in res]
    cost = lambda i: (3 if specs[i][0] == "evolved" else 2 if "|640|" in keys[i][1] and "pair" in keys[i][1] else 1)
    todo.sort(key=lambda i: -cost(i))
    print(f"{len(res)} from checkpoint, {len(todo)} to do", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers or os.cpu_count()) as ex, open(ck, "a") as f:
        futs = {ex.submit(job, specs[i]): i for i in todo}
        for n, fu in enumerate(as_completed(futs)):
            i = futs[fu]; res[tuple(keys[i])] = fu.result()
            f.write(json.dumps(dict(key=list(keys[i]), res=res[tuple(keys[i])])) + "\n"); f.flush()
            print(f"  {n + 1}/{len(todo)} {list(keys[i])} ({time.time() - t0:.0f}s)", flush=True)

    out = {}
    for shift in (1.0, 0.3):
        cell = f"shift {shift}"; col = lambda v, m, S=seeds: [res[(cell, v, sd)][m] for sd in S]
        tab = {v: {m: dict(mean=float(np.mean(col(v, m))), sd=float(np.std(col(v, m), ddof=1)), per_seed=col(v, m))
                   for m in ("pre", "rec", "post", "cover_end")} for v in ("swarm orig", "swarm full", "oracle team", "random")}
        bandits = sorted({k[1] for k in keys if k[0] == cell and "|" in k[1]})
        dev = {v: float(np.mean(col(v, "post", DEV))) for v in bandits}
        chosen = {}
        for fam in ("agent_ucb", "pair_ucb", "pair_mean"):
            for N in (160, 640):
                c_ = [v for v in bandits if v.startswith(f"{fam}|{N}|")]
                if c_:
                    best = max(c_, key=dev.get); chosen[f"{fam}@{N}"] = best
                    tab[f"{fam}@{N} [{best}]"] = {m: dict(mean=float(np.mean(col(best, m))), sd=float(np.std(col(best, m), ddof=1)),
                                                         per_seed=col(best, m)) for m in ("pre", "rec", "post", "cover_end")}
        best_all = max(chosen.values(), key=dev.get)
        cmp_ = {"H16a full - orig": {m: paired(col("swarm full", m), col("swarm orig", m)) for m in ("rec", "post")},
                f"H16b full - best bandit [{best_all}]": {m: paired(col("swarm full", m), col(best_all, m)) for m in ("rec", "post")}}
        for name, v in chosen.items():
            cmp_[f"full - {name}"] = {m: paired(col("swarm full", m), col(v, m)) for m in ("rec", "post")}
        out[cell] = dict(table=tab, dev_post=dev, chosen=chosen, best_bandit=best_all, comparisons=cmp_)
        print(f"\n## {cell}")
        for v, e in tab.items():
            print(f"{v:<42} pre={e['pre']['mean']:.3f} rec={e['rec']['mean']:.3f} post={e['post']['mean']:.3f}±{e['post']['sd']:.3f} cover={e['cover_end']['mean']:.2f}")
        for k, v in cmp_.items():
            print(f"  {k}: post {v['post']['diff']:+.3f}±{v['post']['sd']:.3f} effect={v['post']['effect']}; rec {v['rec']['diff']:+.3f}±{v['rec']['sd']:.3f}")
    ev = lambda v, m, S=seeds: [res[("evolved", "all", sd)][v][m] for sd in S]
    names = list(res[("evolved", "all", seeds[0])])
    dev = {v: float(np.mean(ev(v, "post", DEV))) for v in names}
    pick = {fam: max([v for v in names if v.startswith(fam)], key=dev.get) for fam in ("agent_ucb", "pair_ucb")}
    show = ["swarm full", "random (original)", "random (evolved)", "oracle team (evolved)", "pair_mean (evolved)", *pick.values()]
    tab = {v: dict(post=float(np.mean(ev(v, "post"))), sd=float(np.std(ev(v, "post"), ddof=1)), cover_end=float(np.mean(ev(v, "cover_end"))),
                   per_seed=ev(v, "post")) for v in show}
    best = max(["pair_mean (evolved)", *pick.values()], key=dev.get)
    cmp_ = {f"H16c full - best bandit on evolved population [{best}]": paired(ev("swarm full", "post"), ev(best, "post"))}
    for v in ("pair_mean (evolved)", *pick.values()):
        cmp_[f"full - {v}"] = paired(ev("swarm full", "post"), ev(v, "post"))
    cmp_["random evolved - random original"] = paired(ev("random (evolved)", "post"), ev("random (original)", "post"))
    out["evolved population (no shift)"] = dict(table=tab, dev_post=dev, best_bandit=best, comparisons=cmp_)
    print("\n## evolved population (no shift)")
    for v, e in tab.items():
        print(f"{v:<36} post={e['post']:.3f}±{e['sd']:.3f} cover={e['cover_end']:.2f}")
    for k, v in cmp_.items():
        print(f"  {k}: {v['diff']:+.3f}±{v['sd']:.3f} effect={v['effect']}")
    save("h16_shift", dict(T=T, seeds=seeds, dev_seeds=DEV, busy=BUSY, N_swarm=N_SWARM, results=out))


if __name__ == "__main__":
    main()
