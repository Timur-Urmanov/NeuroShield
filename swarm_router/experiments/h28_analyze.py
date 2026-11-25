"""H28 analysis: A, C (contrib) and NI in four other environments (criteria in notes/hypotheses.md).
Written before the runs finished."""
import os

import numpy as np

from common import ROOT, save
from h25_analyze import st

CFG = ("b24", "b96", "n320", "l7")
TAGS = ("A", "C", "NI")
SEG = {"S1": (40, 80), "S2": (80, 120), "S3": (120, 160)}
DELTA, Q = 0.015, 0.95 ** 3


def sg(y, s):
    return y[:, SEG[s][0]:SEG[s][1]].mean(1)


def line(name, d):
    return f"{name}: {d['mean']:+.4f} sd {d['sd']:.4f} se {d['se']:.4f} t {d['t']:.1f} wins {d['wins']}/{d['n']} t>=2.5={abs(d['t']) >= 2.5} 2sd={d['effect_2sd']}"


if __name__ == "__main__":
    out = {}
    for cfg in CFG:
        d = os.path.join(ROOT, ".cache", f"h28_{cfg}")
        seeds = [s for s in range(50, 55) if all(os.path.exists(os.path.join(d, f"{t}_{s}.npz")) for t in TAGS)]
        F = {t: [np.load(os.path.join(d, f"{t}_{s}.npz")) for s in seeds] for t in TAGS}
        G = lambda t, k: np.array([f[k] for f in F[t]])
        R = {t: G(t, "R") for t in TAGS}; n = len(seeds)
        o = out[cfg] = dict(seeds=seeds, wall_min={t: float(np.mean(G(t, "wall")) / 60) for t in TAGS})
        print(f"\n=== {cfg}: seeds {seeds}, wall min " + " ".join(f"{t} {o['wall_min'][t]:.0f}" for t in TAGS))
        o["curve_100k"] = {t: R[t].reshape(n, 16, 10).mean(2).mean(0).tolist() for t in TAGS}
        for t in TAGS:
            print(f"  {t:<3}", " ".join(f"{x:.3f}" for x in o["curve_100k"][t]))
        o["levels"] = {}
        for t in TAGS:
            lv = o["levels"][t] = dict(**{s: st(sg(R[t], s)) for s in SEG}, d32=st(sg(R[t], "S3") - sg(R[t], "S2")), d31=st(sg(R[t], "S3") - sg(R[t], "S1")))
            print(f"  {t:<3} " + " ".join(f"{s} {lv[s]['mean']:.3f}±{lv[s]['sd']:.3f}" for s in SEG) + " | " + line("d32", lv["d32"]) + f" | d31 {lv['d31']['mean']:+.4f} (t {lv['d31']['t']:.1f})")
        T = o["tests"] = {"NI-A": st(sg(R["NI"], "S3") - sg(R["A"], "S3")), "NI-C": st(sg(R["NI"], "S3") - sg(R["C"], "S3")), "C-A": st(sg(R["C"], "S3") - sg(R["A"], "S3"))}
        for k in T:
            print("  " + line(k + " on S3", T[k]))
        ni = o["levels"]["NI"]
        o["H28a"] = bool(T["NI-A"]["mean"] > 0 and T["NI-A"]["t"] >= 2.5); o["H28b"] = bool(T["NI-C"]["mean"] > 0 and T["NI-C"]["t"] >= 2.5)
        o["H28c"] = bool(ni["d32"]["mean"] >= -2 * ni["d32"]["se"] and ni["d31"]["mean"] >= -DELTA)
        o["C_declines"] = bool(o["levels"]["C"]["d32"]["mean"] < -2 * o["levels"]["C"]["d32"]["se"])
        print(f"  H28a {o['H28a']}  H28b {o['H28b']}  H28c {o['H28c']}  C declines {o['C_declines']}")
        o["loss"] = {}
        for t in TAGS:
            L = dict(q_team=G(t, "q_team"), competence=Q - G(t, "q_pop"), busy=G(t, "q_pop") - G(t, "q_avail"), routing=G(t, "q_avail") - G(t, "q_team"),
                     model=G(t, "er_best") - G(t, "er"))
            o["loss"][t] = {k: {s: float(sg(y, s).mean()) for s in SEG} for k, y in L.items()}
            print(f"  {t:<3} " + " | ".join(f"{k} " + " ".join(f"{o['loss'][t][k][s]:.3f}" for s in SEG) for k in L))
        o["composition"] = {}
        for t in TAGS:
            rows = []
            for f in F[t]:
                tg = f["targets"]; r = []
                for i in (19, 39, 59, 79):
                    g = f["genes"][i]; cnt = np.array([[(g == tg[k, j]).all(1).sum() for j in range(tg.shape[1])] for k in range(tg.shape[0])])
                    r.append((float((cnt > 0).sum()), float(cnt.max()), float(np.median(cnt))))
                rows.append(r)
            m = np.mean(rows, 0); o["composition"][t] = m.tolist()
            print(f"  {t:<3} niches filled " + " ".join(f"{x:.1f}" for x in m[:, 0]) + " | largest " + " ".join(f"{x:.0f}" for x in m[:, 1]) + " | median " + " ".join(f"{x:.0f}" for x in m[:, 2]))
        print("  per-seed S3: " + " | ".join(f"{t} " + " ".join(f"{x:.3f}" for x in sg(R[t], "S3")) for t in TAGS))
    print("\nsummary: " + " ; ".join(f"{c}: a={out[c]['H28a']} b={out[c]['H28b']} c={out[c]['H28c']}" for c in CFG))
    save("h28_portability", out)
