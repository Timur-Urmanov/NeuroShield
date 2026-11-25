"""H27 analysis: switching C to the noinherit eviction rule at 400k/800k, and the replication of the freeze control
(criteria in notes/hypotheses.md). Written before the runs finished."""
import os

import numpy as np

from common import ROOT, save
from h25_analyze import st

D27, D26 = os.path.join(ROOT, ".cache", "h27"), os.path.join(ROOT, ".cache", "h26")
TAGS = ("C", "NI", "F400", "F800", "S400", "S800")
SEG = {"S1": (40, 80), "S2": (80, 120), "S3": (120, 160)}
Q = 0.95 ** 3


def sg(y, s):
    return y[:, SEG[s][0]:SEG[s][1]].mean(1)


def line(name, d):
    return f"{name}: {d['mean']:+.4f} sd {d['sd']:.4f} se {d['se']:.4f} t {d['t']:.1f} wins {d['wins']}/{d['n']} t>=2.5={abs(d['t']) >= 2.5} 2sd={d['effect_2sd']}"


def load(d, tags, seeds):
    seeds = [s for s in seeds if all(os.path.exists(os.path.join(d, f"{t}_{s}.npz")) for t in tags)]
    return seeds, {t: [np.load(os.path.join(d, f"{t}_{s}.npz")) for s in seeds] for t in tags}


if __name__ == "__main__":
    seeds, F = load(D27, TAGS, range(40, 48)); n = len(seeds)
    G = lambda t, k: np.array([f[k] for f in F[t]])
    R = {t: G(t, "R") for t in TAGS}
    out = dict(seeds=seeds, wall_min={t: float(np.mean(G(t, "wall")) / 60) for t in TAGS}); print("seeds", seeds, "wall min", out["wall_min"])
    print("\nreward by 100k block:"); out["curve_100k"] = {}
    for t in TAGS:
        c = R[t].reshape(n, 16, 10).mean(2).mean(0); out["curve_100k"][t] = c.tolist(); print(f"  {t:<5}", " ".join(f"{x:.3f}" for x in c))
    out["levels"] = {}; print("\nlevels and d32 = S3 - S2")
    for t in TAGS:
        o = out["levels"][t] = dict(**{s: st(sg(R[t], s)) for s in SEG}, d32=st(sg(R[t], "S3") - sg(R[t], "S2")))
        print(f"  {t:<5} " + " ".join(f"{s} {o[s]['mean']:.3f}±{o[s]['sd']:.3f}" for s in SEG) + " | " + line("d32", o["d32"]))
    dd = lambda a, b, s="S3": st(sg(R[a], s) - sg(R[b], s))
    d32 = lambda t: sg(R[t], "S3") - sg(R[t], "S2")
    out["tests"] = T = {}
    print("\nH27a: S - C on S3 (both points needed), S has no decline")
    for p in ("400", "800"):
        T[f"S{p}-C"] = dd(f"S{p}", "C"); print("  " + line(f"S{p} - C", T[f"S{p}-C"]))
        T[f"d32(S{p})-d32(C)"] = st(d32(f"S{p}") - d32("C")); print("  " + line(f"d32(S{p}) - d32(C)", T[f"d32(S{p})-d32(C)"]))
    print("H27b: S - F at the same point on S3; S - NI on S3 and on the last 100k")
    for p in ("400", "800"):
        T[f"S{p}-F{p}"] = dd(f"S{p}", f"F{p}"); print("  " + line(f"S{p} - F{p}", T[f"S{p}-F{p}"]))
        T[f"S{p}-NI"] = dd(f"S{p}", "NI"); print("  " + line(f"S{p} - NI", T[f"S{p}-NI"]))
        T[f"S{p}-NI last100k"] = st(R[f"S{p}"][:, 150:].mean(1) - R["NI"][:, 150:].mean(1)); print("  " + line(f"S{p} - NI last 100k", T[f"S{p}-NI last100k"]))
        b = int(p) // 10; pre = R["C"][:, b - 10:b].mean(1)                      # recovery after the switch, in 100k blocks
        rec = [float((R[f"S{p}"][:, b + 10 * i:b + 10 * i + 10].mean(1) - pre).mean()) for i in range((160 - b) // 10)]
        T[f"S{p} recovery curve"] = rec; print(f"  S{p} minus level before switch, by 100k: " + " ".join(f"{x:+.3f}" for x in rec))
    T["NI-C"] = dd("NI", "C"); print("  " + line("NI - C", T["NI-C"]))
    print("H27d: freeze replication on these seeds")
    for p in ("400", "800"):
        T[f"F{p}-C"] = dd(f"F{p}", "C"); print("  " + line(f"F{p} - C", T[f"F{p}-C"]))
        T[f"d32(F{p})-d32(C)"] = st(d32(f"F{p}") - d32("C")); print("  " + line(f"d32(F{p}) - d32(C)", T[f"d32(F{p})-d32(C)"]))
    s26, F26 = load(D26, ("C", "F400", "F800", "NI"), range(30, 38))            # pre-registered pooled analysis, 16 seeds
    P = {t: np.concatenate([np.array([f["R"] for f in F26[t]]), R[t]]) for t in ("C", "F400", "F800", "NI")}
    pd32 = lambda t: P[t][:, 120:160].mean(1) - P[t][:, 80:120].mean(1)
    print(f"  pooled {len(s26)} + {n} seeds:"); out["pooled"] = PO = dict(seeds=s26 + seeds)
    PO["d32(C)"] = st(pd32("C")); print("  " + line("d32(C)", PO["d32(C)"]))
    for t in ("F400", "F800", "NI"):
        PO[f"{t}-C"] = st(P[t][:, 120:].mean(1) - P["C"][:, 120:].mean(1)); print("  " + line(f"{t} - C on S3", PO[f"{t}-C"]))
        PO[f"d32({t})-d32(C)"] = st(pd32(t) - pd32("C")); print("  " + line(f"d32({t}) - d32(C)", PO[f"d32({t})-d32(C)"]))
    print("\nH27c: loss components on S3 (competence = q_ideal - q_pop, busy = q_pop - q_avail, routing = q_avail - q_team)")
    L = {t: dict(competence=Q - G(t, "q_pop"), busy=G(t, "q_pop") - G(t, "q_avail"), routing=G(t, "q_avail") - G(t, "q_team"),
                 model=G(t, "er_best") - G(t, "er"), q_team=G(t, "q_team")) for t in TAGS}
    out["loss"] = {t: {k: {s: float(sg(y, s).mean()) for s in SEG} for k, y in L[t].items()} for t in TAGS}
    for t in TAGS:
        print(f"  {t:<5} " + " | ".join(f"{k} " + " ".join(f"{out['loss'][t][k][s]:.3f}" for s in SEG) for k in ("q_team", "competence", "busy", "routing", "model")))
    out["loss_minus_C_S3"] = {}
    for t in TAGS[1:]:
        o = out["loss_minus_C_S3"][t] = {k: st(sg(L[t][k], "S3") - sg(L["C"][k], "S3")) for k in L[t]}
        print(f"  {t} - C on S3: " + " | ".join(f"{k} {o[k]['mean']:+.4f} (se {o[k]['se']:.4f}, t {o[k]['t']:.1f})" for k in o))
    print("\nper-seed S3 reward:")
    for t in TAGS:
        print(f"  {t:<5} " + " ".join(f"{x:.3f}" for x in sg(R[t], "S3")))
    print("\ncomposition at 400k / 800k / 1.2M / 1.6M (means over seeds): niches with >= 16 perfect copies, largest niche")
    out["composition"] = {}
    for t in TAGS:
        rows = []
        for f in F[t]:
            tg = f["targets"]; r = []
            for i in (19, 39, 59, 79):
                g = f["genes"][i]; cnt = np.array([[(g == tg[k, j]).all(1).sum() for j in range(tg.shape[1])] for k in range(tg.shape[0])])
                r.append((float((cnt >= 16).sum()), float(cnt.max()), float(np.median(cnt))))
            rows.append(r)
        m = np.mean(rows, 0); out["composition"][t] = m.tolist()
        print(f"  {t:<5} ge16 " + " ".join(f"{x:.1f}" for x in m[:, 0]) + " | largest " + " ".join(f"{x:.0f}" for x in m[:, 1]) + " | median " + " ".join(f"{x:.0f}" for x in m[:, 2]))
    save("h27_switch", out)
