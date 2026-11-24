"""H26 analysis (stages 1-2): freeze comparison and the account of quality losses (criteria in notes/hypotheses.md)."""
import os, sys

import numpy as np

from common import ROOT, save
from h25_analyze import st, comp

OUT = os.path.join(ROOT, ".cache", "h26")
TAGS = [t for t in ("C", "NI", "F400", "F600", "F800", "A", "D", "FN600") if any(f.startswith(t + "_") for f in os.listdir(OUT))]
SEG = {"S1": (40, 80), "S2": (80, 120), "S3": (120, 160)}                  # in 10k blocks
DELTA = 0.015


def sg(y, s):
    return y[:, SEG[s][0]:SEG[s][1]].mean(1)


def line(name, d):
    return f"{name}: {d['mean']:+.4f} sd {d['sd']:.4f} se {d['se']:.4f} t {d['t']:.1f} wins {d['wins']}/{d['n']} SE={d['effect_se']} 2sd={d['effect_2sd']}"


if __name__ == "__main__":
    seeds = [sd for sd in range(30, 38) if all(os.path.exists(os.path.join(OUT, f"{t}_{sd}.npz")) for t in TAGS)]
    F = {t: [np.load(os.path.join(OUT, f"{t}_{sd}.npz")) for sd in seeds] for t in TAGS}
    G = lambda t, k: np.array([f[k] for f in F[t]])
    out = dict(seeds=seeds, tags=TAGS, cost={t: float(np.mean([f["wall"] for f in F[t]]) / 60) for t in TAGS}); print("seeds", seeds, "wall min", out["cost"])
    R = {t: G(t, "R") for t in TAGS}
    print("\nreward by 100k block:"); out["curve_100k"] = {}
    for t in TAGS:
        c = R[t].reshape(len(seeds), 16, 10).mean(2).mean(0); out["curve_100k"][t] = c.tolist(); print(f"  {t:<5}", " ".join(f"{x:.3f}" for x in c))
    out["stage1"] = {}
    print("\nstage 1 (reward): levels, d32 = S3 - S2, difference to C on S3, level kept since the 100k before freezing")
    for t in TAGS:
        o = dict(levels={s: st(sg(R[t], s)) for s in SEG}, d32=st(sg(R[t], "S3") - sg(R[t], "S2")), d21=st(sg(R[t], "S2") - sg(R[t], "S1")))
        print(f"  {t:<5} " + " ".join(f"{s} {o['levels'][s]['mean']:.3f}±{o['levels'][s]['sd']:.3f}" for s in SEG) + " | " + line("d32", o["d32"]))
        if t != "C":
            o["minus_C_S3"] = st(sg(R[t], "S3") - sg(R["C"], "S3")); o["minus_C_last100k"] = st(R[t][:, 150:].mean(1) - R["C"][:, 150:].mean(1))
            print("        " + line("- C on S3", o["minus_C_S3"])); print("        " + line("- C on last 100k", o["minus_C_last100k"]))
        if t.startswith("F"):
            b = int("".join(ch for ch in t if ch.isdigit())) // 10; pre = R["C"][:, b - 10:b].mean(1)
            o["kept"] = st(sg(R[t], "S3") - pre); o["kept_ok"] = bool(o["kept"]["mean"] - 2 * o["kept"]["se"] > -DELTA)
            o["post_freeze_100k"] = st(R[t][:, b:b + 10].mean(1) - pre)
            print("        " + line("S3 - level before freeze", o["kept"]) + f" kept={o['kept_ok']}"); print("        " + line("first 100k after freeze - before", o["post_freeze_100k"]))
        out["stage1"][t] = o
    out["stage2"] = {}
    print("\nstage 2: quality account (q_ideal = %.3f): competence = q_ideal - q_pop, busy = q_pop - q_avail, routing = q_avail - q_team" % (0.95 ** 3))
    comp_ = {}
    for t in TAGS:
        qp, qa, qt = G(t, "q_pop"), G(t, "q_avail"), G(t, "q_team")
        comp_[t] = dict(competence=0.95 ** 3 - qp, busy=qp - qa, routing=qa - qt, q_team=qt, model=G(t, "er_best") - G(t, "er"), reward=R[t],
                        min_free=G(t, "min_free"), cap_obs=G(t, "cap_obs"), good_obs=G(t, "good_obs"),
                        **{f"cat_{n}": G(t, "cat_n")[:, :, i] / G(t, "cat_n").sum(2) for i, n in enumerate(("absent", "busy", "routing", "full"))},
                        **{f"succ_{n}": G(t, "cat_s")[:, :, i] / np.maximum(G(t, "cat_n")[:, :, i], 1) for i, n in enumerate(("absent", "busy", "routing", "full"))},
                        **{f"role_{n}": G(t, "role")[:, :, i] for i, n in enumerate(("absent", "busy", "notpicked", "perfect"))})
        o = out["stage2"][t] = {k: dict(**{s: st(sg(y, s)) for s in SEG}, d31=st(sg(y, "S3") - sg(y, "S1")), curve_100k=y.reshape(len(seeds), 16, 10).mean(2).mean(0).tolist())
                                for k, y in comp_[t].items()}
        print(f"  {t}")
        for k in ("q_team", "competence", "busy", "routing", "model", "min_free", "cap_obs", "good_obs", "cat_absent", "cat_busy", "cat_routing", "cat_full",
                  "role_absent", "role_busy", "role_notpicked", "role_perfect", "succ_absent", "succ_busy", "succ_routing", "succ_full"):
            print(f"    {k:<15} " + " ".join(f"{s} {o[k][s]['mean']:.4f}" for s in SEG) + " | " + line("S3-S1", o[k]["d31"]))
        dq = -o["q_team"]["d31"]["mean"]
        o["shares_of_q_drop"] = {k: float(o[k]["d31"]["mean"] / dq) if dq else None for k in ("competence", "busy", "routing")}
        print("    shares of the S1->S3 drop in q_team:", {k: round(v, 2) for k, v in o["shares_of_q_drop"].items()}, "(drop %.4f)" % dq)
    print("\nH26e: frozen minus C on S3, by component")
    out["H26e"] = {}
    for t in TAGS:
        if t == "C":
            continue
        o = out["H26e"][t] = {k: st(sg(comp_[t][k], "S3") - sg(comp_["C"][k], "S3")) for k in ("q_team", "competence", "busy", "routing", "model", "reward")}
        print(f"  {t}: " + " | ".join(f"{k} {o[k]['mean']:+.4f} (se {o[k]['se']:.4f}, t {o[k]['t']:.1f})" for k in o))
    print("\nobservable proxy vs hidden deficit (within-seed correlation over 10k blocks from 200k, mean over seeds):")
    out["proxy"] = {}
    for t in TAGS:
        if t == "A":
            continue
        cs = {k: float(np.mean([np.corrcoef(comp_[t]["cap_obs"][i, 20:], comp_[t][k][i, 20:])[0, 1] for i in range(len(seeds))])) for k in ("busy", "min_free", "q_team", "reward")}
        out["proxy"][t] = cs; print(f"  {t}: corr(cap_obs, .) =", {k: round(v, 2) for k, v in cs.items()})
    print("\ncomposition (means over seeds at 400k / 600k / 800k / 1.2M / 1.6M):")
    out["composition"] = {}
    for t in TAGS:
        if t == "A":
            continue
        M = [comp(f) for f in F[t]]; L = 5; o = out["composition"][t] = {}
        cop = []
        for f in F[t]:
            g, tg = f["genes"], f["targets"]; mt = (g[:, None, :, None, :] == tg[None, :, None, :, :]).sum(-1)            # snap x K x N x roles
            cop.append(np.sort((mt == L).sum(2).reshape(len(g), -1), 1))                                                  # copies per niche, sorted
        cop = np.array(cop, float); p = cop / np.maximum(cop.sum(2, keepdims=True), 1)
        M2 = dict(eff_cfg=np.array([m["eff_cfg"] for m in M]), eff_lin=np.array([m["eff_lin"] for m in M]), perfect_share=np.array([m["perfect_share"] for m in M]),
                  niches=np.array([m["niches"] for m in M]), eff_niches=1 / np.maximum((p ** 2).sum(2), 1e-12), largest_niche=cop[:, :, -1],
                  median_niche=np.median(cop, 2), niches_ge16=(cop >= 16).sum(2), second_niche=cop[:, :, -2])
        for k, y in M2.items():
            o[k] = {str(tt): st(y[:, tt // 20000 - 1]) for tt in (400000, 600000, 800000, 1200000, 1600000)}
            print(f"  {t:<5}{k:<15} " + " ".join(f"{o[k][x]['mean']:.2f}" for x in o[k]))
    if "nosave" not in sys.argv:
        save("h26_freeze", out)
