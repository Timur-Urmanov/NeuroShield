"""H35 stage C: does the timing of replacements matter? Interval K and packet size n; branches of a mature colony at 400k
(part 1) and the same regimes from the start (part 2). Criteria in notes/hypotheses.md. Written before the runs finished."""
import os

import numpy as np

from common import ROOT, save
from h25_analyze import st

B = os.path.join(ROOT, ".cache", "h35c"); SEEDS = range(150, 156); Q = 0.95 ** 3
P1 = {"stat": ("C0", "T100x3", "T100x9", "T367x33", "T27x9", "T300x99", "T50x33"), "mild": ("T100x9", "T27x9", "T300x99"), "hard": ("T100x9", "T27x9", "T300x99")}
RATE = {"NI": 0.33, "C0": 0.0, "T100x3": 0.03, "T100x9": 0.09, "T367x33": 0.0899, "T27x9": 0.3333, "T300x99": 0.33, "T50x33": 0.66}
WIN = {"stat": {"main": (120, 160), "0.4-1.6M": (40, 160)}, "mild": {"main": (80, 160), "0.8-1.2M": (80, 120), "1.2-1.6M": (120, 160)}}; WIN["hard"] = WIN["mild"]


def line(name, d):
    return f"{name}: {d['mean']:+.4f} sd {d['sd']:.4f} se {d['se']:.4f} t {d['t']:.1f} wins {d['wins']}/{d['n']} t>=2.5={abs(d['t']) >= 2.5} 2sd={d['effect_2sd']}"


def desc(F, a, b):
    G = lambda k: np.array([f[k] for f in F]); n = len(F)
    L = dict(competence=float((Q - G("q_pop"))[:, a:b].mean()), busy=float((G("q_pop") - G("q_avail"))[:, a:b].mean()), routing=float((G("q_avail") - G("q_team"))[:, a:b].mean()), model=float((G("er_best") - G("er"))[:, a:b].mean()))
    g = [f["genes"][-1] for f in F]; tg = [f["targets"] for f in F]
    cnt = np.array([[[(x == t[k, j]).all(1).sum() for j in range(3)] for k in range(3)] for x, t in zip(g, tg)])
    effc = [np.exp(-(p * np.log(p)).sum()) for p in (np.unique(x, axis=0, return_counts=True)[1] / len(x) for x in g)]; ev = G("ev")[:, a:b].sum(1)
    L.update(niches_ge16=float((cnt >= 16).sum((1, 2)).mean()), perfect_copies=float(cnt.sum((1, 2)).mean()), largest_niche=float(cnt.max((1, 2)).mean()), eff_configs=float(np.mean(effc)),
             births_k=float(ev[:, 0].mean() / 1000), zero_use_share=float((ev[:, 2] / np.maximum(ev[:, 0], 1)).mean()), uses_per_evicted=float((ev[:, 3] / np.maximum(ev[:, 0], 1)).mean()),
             perfect_share_births=float((ev[:, 1] / np.maximum(ev[:, 0], 1)).mean()), state_k=float(np.mean(G("state")) / 1000), min_free=float(G("min_free")[:, a:b].mean()))
    return L


if __name__ == "__main__":
    out = {}
    print("PART 1: branches of a mature colony at 400k (paired with NI up to the fork)")
    for env, kinds in P1.items():
        D = os.path.join(B, env); seeds = [s for s in SEEDS if all(os.path.exists(os.path.join(D, f"{t}_{s}.npz")) for t in ("NI",) + kinds)]
        F = {t: [np.load(os.path.join(D, f"{t}_{s}.npz")) for s in seeds] for t in ("NI",) + kinds}; n = len(seeds)
        R = {t: np.array([f["R"] for f in F[t]]) for t in F}
        for t in kinds:
            assert np.allclose(R[t][:, :40], R["NI"][:, :40]), (env, t)
        o = out["part1_" + env] = dict(seeds=seeds, curve_100k={t: R[t].reshape(n, 16, 10).mean(2).mean(0).tolist() for t in F}); print(f"\n=== {env}: seeds {seeds}")
        for t in F:
            print(f"  {t:<8} rate {RATE[t]:.2f}/task*100 | " + " ".join(f"{x:.3f}" for x in o["curve_100k"][t]))
        o["levels"] = {t: {w: st(R[t][:, a:b].mean(1)) for w, (a, b) in WIN[env].items()} for t in F}; T = o["tests"] = {}
        for t in kinds:
            for w, (a, b) in WIN[env].items():
                T[f"{t}-NI {w}"] = st(R[t][:, a:b].mean(1) - R["NI"][:, a:b].mean(1)); print("  " + line(f"{t} - NI on {w}", T[f"{t}-NI {w}"]))
        a, b = WIN[env]["main"]; m = lambda t: R[t][:, a:b].mean(1)
        if env == "stat":
            for x, y in (("T27x9", "T100x9"), ("T367x33", "T100x9"), ("T367x33", "C0"), ("T100x9", "C0"), ("T100x3", "C0"), ("T27x9", "T300x99"), ("T50x33", "T300x99")):
                T[f"{x}-{y}"] = st(m(x) - m(y)); print("  " + line(f"{x} - {y} on main", T[f"{x}-{y}"]))
        else:
            for x, y in (("T27x9", "T100x9"), ("T27x9", "T300x99")):
                T[f"{x}-{y}"] = st(m(x) - m(y)); print("  " + line(f"{x} - {y} on main", T[f"{x}-{y}"]))
        print("  per-seed main: " + " | ".join(f"{t} " + " ".join(f"{x:.3f}" for x in m(t)) for t in F))
        o["desc"] = {t: desc(F[t], a, b) for t in F}
        for t in F:
            print(f"  {t:<8} " + " ".join(f"{k} {v:.4g}" for k, v in o["desc"][t].items()))
        if env != "stat":
            o["refilled"] = {}
            for t in F:
                r = []
                for f in F[t]:
                    tg, t0 = f["targets"], f["targets0"]; ch = (tg != t0).any(-1); g = f["genes"][-1]
                    r.append(int(sum((g == tg[k, j]).all(1).sum() >= 16 for k, j in np.argwhere(ch))))
                o["refilled"][t] = r; print(f"  {t:<8} changed niches with >= 16 copies at the end, per seed {r} (mean {np.mean(r):.2f} of 3)")
    print("\nPART 2: the same regimes from the start (stationary; paired with NI by seed and task stream only)")
    D0 = os.path.join(B, "stat"); arms = ("27x9", "300x99", "50x33", "200x33")
    seeds = [s for s in SEEDS if os.path.exists(os.path.join(D0, f"NI_{s}.npz")) and all(os.path.exists(os.path.join(B, f"start_{a}", f"NI_{s}.npz")) for a in arms)]
    F = {"NI": [np.load(os.path.join(D0, f"NI_{s}.npz")) for s in seeds]}; F.update({a: [np.load(os.path.join(B, f"start_{a}", f"NI_{s}.npz")) for s in seeds] for a in arms})
    R = {t: np.array([f["R"] for f in F[t]]) for t in F}; n = len(seeds); o = out["part2"] = dict(seeds=seeds, curve_100k={t: R[t].reshape(n, 16, 10).mean(2).mean(0).tolist() for t in F})
    for t in F:
        print(f"  {t:<8} " + " ".join(f"{x:.3f}" for x in o["curve_100k"][t]))
    W2 = {"0-0.4M": (0, 40), "S3": (120, 160), "whole": (0, 160)}; T = o["tests"] = {}
    o["levels"] = {t: {w: st(R[t][:, a:b].mean(1)) for w, (a, b) in W2.items()} for t in F}
    for t in arms:
        for w, (a, b) in W2.items():
            T[f"{t}-NI {w}"] = st(R[t][:, a:b].mean(1) - R["NI"][:, a:b].mean(1)); print("  " + line(f"{t} - NI on {w}", T[f"{t}-NI {w}"]))
    print("  per-seed S3: " + " | ".join(f"{t} " + " ".join(f"{x:.3f}" for x in R[t][:, 120:].mean(1)) for t in F))
    o["desc"] = {t: desc(F[t], 120, 160) for t in F}
    for t in F:
        print(f"  {t:<8} " + " ".join(f"{k} {v:.4g}" for k, v in o["desc"][t].items()))
        f400 = [next((float(f["t"][i]) for i in range(len(f["t"])) if (np.array([[(f["genes"][i] == f["targets"][k, j]).all(1).sum() for j in range(3)] for k in range(3)]) >= 16).all()), np.nan) for f in F[t]]
        o["desc"][t]["t_all_niches_ge16"] = f400; print(f"           time until all 9 niches have >= 16 copies, per seed: {f400}")
    p1 = out["part1_stat"]["tests"]; v = out["verdict"] = {}
    for a_, k_ in (("27x9", "T27x9"), ("300x99", "T300x99")):
        d1, d2 = p1[f"{k_}-NI main"], T[f"{a_}-NI S3"]; same = np.sign(d1["mean"]) == np.sign(d2["mean"])
        v[a_] = dict(part1=d1["mean"], t1=d1["t"], part2=d2["mean"], t2=d2["t"], reproducible=bool(abs(d1["t"]) >= 2.5 and abs(d2["t"]) >= 2.5 and same))
    print("\ncondition 3 (equal rate, different timing; effect in both parts with the same sign):", {k: (round(x["part1"], 4), round(x["t1"], 1), round(x["part2"], 4), round(x["t2"], 1), x["reproducible"]) for k, x in v.items()})
    save("h35c_timing", out)
