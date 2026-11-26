"""H29 analysis: NI against the baselines with search (R random search, PN pattern + novelty) on the long horizon
(criteria in notes/hypotheses.md). Written before the runs finished. Original environment: NI, R from .cache/h29,
PN, A, C from H25 (.cache/h25, same seeds and task stream); busy=24 and genes_L=7: everything in .cache/h28_*."""
import os

import numpy as np

from common import ROOT, save
from h25_analyze import st

SEG = {"S1": (40, 80), "S2": (80, 120), "S3": (120, 160)}
ENV = {"orig": (range(20, 30), {"NI": "h29", "R": "h29", "PN": "h25", "A": "h25", "C": "h25"}),
       "b24": (range(50, 55), {t: "h28_b24" for t in ("NI", "R", "PN", "A", "C")}),
       "l7": (range(50, 55), {t: "h28_l7" for t in ("NI", "R", "PN", "A", "C")})}


def sg(y, s):
    return y[:, SEG[s][0]:SEG[s][1]].mean(1)


def line(name, d):
    return f"{name}: {d['mean']:+.4f} sd {d['sd']:.4f} se {d['se']:.4f} t {d['t']:.1f} wins {d['wins']}/{d['n']} t>=2.5={abs(d['t']) >= 2.5} 2sd={d['effect_2sd']}"


if __name__ == "__main__":
    out = {}; ok = []
    chk = [os.path.join(ROOT, ".cache", d, "A_20.npz") for d in ("h29", "h25")]
    if all(os.path.exists(f) for f in chk):                          # pairing check: the H26 runner reproduces the H25 run exactly
        a, b = (np.load(f)["R"] for f in chk); out["pairing_A20_max_abs_diff"] = float(np.abs(a - b).max()); print("pairing check A seed 20, max |diff|:", out["pairing_A20_max_abs_diff"])
    for env, (rng, src) in ENV.items():
        P = lambda t, s: os.path.join(ROOT, ".cache", src[t], f"{t}_{s}.npz")
        seeds = [s for s in rng if all(os.path.exists(P(t, s)) for t in src)]
        F = {t: [np.load(P(t, s)) for s in seeds] for t in src}; R = {t: np.array([f["R"] for f in F[t]]) for t in src}; n = len(seeds)
        assert all(np.array_equal(F["NI"][i]["targets"], F[t][i]["targets"]) and np.array_equal(F["NI"][i]["genes0"], F[t][i]["genes0"]) for t in src for i in range(n))
        o = out[env] = dict(seeds=seeds, wall_min={t: float(np.mean([f["wall"] for f in F[t]]) / 60) for t in src})
        print(f"\n=== {env}: seeds {seeds}, wall min " + " ".join(f"{t} {o['wall_min'][t]:.0f}" for t in src))
        o["curve_100k"] = {t: R[t].reshape(n, 16, 10).mean(2).mean(0).tolist() for t in src}
        for t in src:
            print(f"  {t:<3}", " ".join(f"{x:.3f}" for x in o["curve_100k"][t]))
        o["levels"] = {}
        for t in src:
            lv = o["levels"][t] = dict(**{s: st(sg(R[t], s)) for s in SEG}, d32=st(sg(R[t], "S3") - sg(R[t], "S2")), total=st(R[t].mean(1)))
            print(f"  {t:<3} " + " ".join(f"{s} {lv[s]['mean']:.3f}±{lv[s]['sd']:.3f}" for s in SEG) + f" | whole run {lv['total']['mean']:.3f} | " + line("d32", lv["d32"]))
        T = o["tests"] = {}
        for b in ("PN", "R", "A", "C"):
            T[f"NI-{b}"] = st(sg(R["NI"], "S3") - sg(R[b], "S3")); print("  " + line(f"NI - {b} on S3", T[f"NI-{b}"]))
            T[f"NI-{b} whole run"] = st(R["NI"].mean(1) - R[b].mean(1)); print("  " + line(f"NI - {b} whole run", T[f"NI-{b} whole run"]))
        for b in ("PN", "R"):
            T[f"{b}-A"] = st(sg(R[b], "S3") - sg(R["A"], "S3")); print("  " + line(f"{b} - A on S3", T[f"{b}-A"]))
            cross = [next((i for i in range(16) if R["NI"][j].reshape(16, 10).mean(1)[i] > R[b][j].reshape(16, 10).mean(1)[i]), None) for j in range(n)]
            T[f"first 100k block NI > {b}"] = cross; print(f"  first 100k block where NI > {b}, per seed: {cross}")
        for b in ("PN", "R"):
            p = bool(T[f"NI-{b}"]["mean"] > 0 and T[f"NI-{b}"]["t"] >= 2.5); o[f"NI>{b}"] = p; ok.append(p)
        print(f"  H29a (NI > PN) {o['NI>PN']}  H29b (NI > R) {o['NI>R']}")
        print("  per-seed S3: " + " | ".join(f"{t} " + " ".join(f"{x:.3f}" for x in sg(R[t], "S3")) for t in ("NI", "PN", "R")))
        o["composition"] = {}
        for t in ("NI", "PN", "R"):
            rows = []
            for f in F[t]:
                tg = f["targets"]; r = []
                for i in (19, 39, 59, 79):
                    g = f["genes"][i]; cnt = np.array([[(g == tg[k, j]).all(1).sum() for j in range(tg.shape[1])] for k in range(tg.shape[0])])
                    r.append((float((cnt > 0).sum()), float(cnt.max()), float(np.median(cnt))))
                rows.append(r)
            m = np.mean(rows, 0); o["composition"][t] = m.tolist()
            print(f"  {t:<3} niches filled " + " ".join(f"{x:.1f}" for x in m[:, 0]) + " | largest " + " ".join(f"{x:.0f}" for x in m[:, 1]) + " | median " + " ".join(f"{x:.0f}" for x in m[:, 2]))
    out["all_six"] = bool(all(ok)); print("\nall six comparisons pass:", out["all_six"])
    save("h29_baselines", out)
