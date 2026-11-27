"""H34 main test: colony (A = NI), evolutionary search (B = ES), hybrid (C = HC), hybrid with mixed generation (D = HD),
factor control (T = EC). Criteria in notes/hypotheses.md. Written before the runs finished."""
import os

import numpy as np

from common import ROOT, save
from h25_analyze import st

V = {"A": "NI", "B": "ES", "C": "HC", "D": "HD", "T": "EC"}
ENVS = ("stat", "mild", "hard")
WIN = {"stat": {"main": (120, 160), "whole": (0, 160)}, "mild": {"main": (80, 160), "whole": (0, 160), "0.8-1.2M": (80, 120), "1.2-1.6M": (120, 160)}}
WIN["hard"] = WIN["mild"]; Q, DELTA = 0.95 ** 3, 0.015


def line(name, d):
    return f"{name}: {d['mean']:+.4f} sd {d['sd']:.4f} se {d['se']:.4f} t {d['t']:.1f} wins {d['wins']}/{d['n']} t>=2.5={abs(d['t']) >= 2.5} 2sd={d['effect_2sd']}"


def pos(d):
    return bool(d["mean"] > 0 and d["t"] >= 2.5)


if __name__ == "__main__":
    out = {}; ver = {h: {} for h in "CD"}
    for env in ENVS:
        D = os.path.join(ROOT, ".cache", "h34", env)
        seeds = [s for s in range(130, 138) if all(os.path.exists(os.path.join(D, f"{t}_{s}.npz")) for t in V.values())]
        F = {v: [np.load(os.path.join(D, f"{t}_{s}.npz")) for s in seeds] for v, t in V.items()}; n = len(seeds)
        G = lambda v, k: np.array([f[k] for f in F[v]])
        assert all(np.array_equal(F["A"][i]["targets"], F[v][i]["targets"]) for v in V for i in range(n))
        R = {v: G(v, "R") for v in V}; o = out[env] = dict(seeds=seeds, wall_min={v: float(np.mean(G(v, "wall")) / 60) for v in V})
        print(f"\n=== {env}: seeds {seeds}, wall min " + " ".join(f"{v} {o['wall_min'][v]:.0f}" for v in V))
        o["curve_100k"] = {v: R[v].reshape(n, 16, 10).mean(2).mean(0).tolist() for v in V}
        for v in V:
            print(f"  {v} " + " ".join(f"{x:.3f}" for x in o["curve_100k"][v]))
        m = lambda v, w="main": R[v][:, WIN[env][w][0]:WIN[env][w][1]].mean(1)
        o["levels"] = {v: {w: st(m(v, w)) for w in WIN[env]} for v in V}
        for v in V:
            print(f"  {v} " + " | ".join(f"{w} {x['mean']:.4f}±{x['sd']:.4f}" for w, x in o["levels"][v].items()) + f" | cumulative {R[v].mean(1).mean() * 1.6e6:,.0f}")
        T = o["tests"] = {}
        for a, b in (("C", "A"), ("C", "B"), ("D", "A"), ("D", "B"), ("C", "T"), ("T", "B"), ("D", "C"), ("A", "B")):
            for w in WIN[env]:
                T[f"{a}-{b} {w}"] = st(m(a, w) - m(b, w)); print("  " + line(f"{a} - {b} on {w}", T[f"{a}-{b} {w}"]))
        best = "A" if m("A").mean() >= m("B").mean() else "B"; o["best_single"] = best
        for h in "CD":
            ba, bb = pos(T[f"{h}-A main"]), pos(T[f"{h}-B main"]); d = st(m(h) - m(best))
            ver[h][env] = dict(beats_A=ba, beats_B=bb, beats_both=ba and bb, best_single=best, vs_best=d["mean"], t=d["t"], not_worse=bool(d["mean"] >= -DELTA))
            print(f"  H34a {h}: beats A {ba}, beats B {bb}; vs best single ({best}) {d['mean']:+.4f} (t {d['t']:.1f}), not worse than delta {d['mean'] >= -DELTA}")
        print("  per-seed main: " + " | ".join(f"{v} " + " ".join(f"{x:.3f}" for x in m(v)) for v in V))
        a, b = WIN[env]["main"]; o["loss"] = {}; o["search"] = {}
        for v in V:
            L = dict(competence=Q - G(v, "q_pop"), busy=G(v, "q_pop") - G(v, "q_avail"), routing=G(v, "q_avail") - G(v, "q_team"), model=G(v, "er_best") - G(v, "er"))
            o["loss"][v] = {k: float(y[:, a:b].mean()) for k, y in L.items()}
            ev = G(v, "ev").sum(1); blk = int(F[v][0]["blk"]); age = G(v, "age_end")
            g = [f["genes"][-1] for f in F[v]]; tg = [f["targets"] for f in F[v]]
            cnt = np.array([[[(x == t[k, j]).all(1).sum() for j in range(3)] for k in range(3)] for x, t in zip(g, tg)])
            effc = [np.exp(-(p * np.log(p)).sum()) for p in (np.unique(x, axis=0, return_counts=True)[1] / len(x) for x in g)]
            o["search"][v] = dict(candidates_k=float(ev[:, 0].mean() / 1000), zero_use_share=float((ev[:, 2] / ev[:, 0]).mean()),
                                  settled_share=float(((G(v, "long_lived").sum(1) + (age >= 5000).sum(1) - (age > 1e8).sum(1)) / ev[:, 0]).mean()),
                                  perfect_share_of_candidates=float((ev[:, 1] / ev[:, 0]).mean()), subs_per_task=float((G(v, "subs").sum(1) / (160 * blk)).mean()),
                                  state_size_k=float(np.mean(G(v, "state")) / 1000), niches_ge16=float((cnt >= 16).sum((1, 2)).mean()), perfect_copies=float(cnt.sum((1, 2)).mean()),
                                  eff_configs=float(np.mean(effc)), wall_min=o["wall_min"][v])
            print(f"  {v} losses(main) " + " ".join(f"{k} {x:.4f}" for k, x in o["loss"][v].items()) + " | " + " ".join(f"{k} {x:.3g}" for k, x in o["search"][v].items()))
        if env != "stat":                                                 # the changed niches: found / kept / used (hidden; diagnostics)
            o["niches"] = {}; cls_all = {}
            for v in V:
                rows = []; cls = np.zeros(4)
                for f in F[v]:
                    tg, t0 = f["targets"], f["targets0"]; ch = np.argwhere((tg != t0).any(-1)); ts = f["t"]
                    for k, j in ch:
                        c = np.array([(gg == tg[k, j]).all(1).sum() for gg in f["genes"]]); fb = float(f["first_birth"][k, j]); nb = float(f["births_niche"][k, j])
                        t16 = next((float(ts[i]) for i in range(len(ts)) if ts[i] > 800000 and c[i] >= 16), np.nan)
                        nn = f["nich"][120:, k, j].sum(0); use = nn[3] / nn[2] if nn[2] > 0 else np.nan      # taken when a free perfect specialist exists (1.2-1.6M)
                        ub = f["nich"][80:, k, j]; ok = np.where(ub[:, 2] >= 50, ub[:, 3] / np.maximum(ub[:, 2], 1), 0) >= 0.5
                        tuse = next((800000 + 10000 * (i + 1) for i in range(len(ok)) if ok[i:i + 3].all() and len(ok[i:i + 3]) == 3), np.nan)
                        found = fb >= 0; kept = c[-1] >= 16; used = kept and use >= 0.5
                        cls[0 if not found else 1 if not kept else 2 if not used else 3] += 1
                        rows.append((found, (fb - 800000) if found else np.nan, nb, c[-1], kept, (t16 - 800000) if np.isfinite(t16) else np.nan, use, (tuse - fb) if found and np.isfinite(tuse) else np.nan))
                r = np.array(rows, float); per_seed_kept = r[:, 4].reshape(n, -1).sum(1); cls_all[v] = per_seed_kept
                o["niches"][v] = dict(not_found=float(cls[0] / n), found_lost=float(cls[1] / n), kept_unused=float(cls[2] / n), recovered=float(cls[3] / n),
                                      refilled_per_seed=per_seed_kept.tolist(), found=float(r[:, 0].sum() / n), t_found=float(np.nanmedian(r[:, 1])) if np.isfinite(r[:, 1]).any() else None,
                                      perfect_births=float(np.nanmean(r[:, 2])), t_refill=float(np.nanmedian(r[:, 5])) if np.isfinite(r[:, 5]).any() else None,
                                      use_when_free=float(np.nanmean(r[:, 6])) if np.isfinite(r[:, 6]).any() else None, t_birth_to_use=float(np.nanmedian(r[:, 7])) if np.isfinite(r[:, 7]).any() else None)
                x = o["niches"][v]
                print(f"  {v} changed niches per seed (of 3): not found {x['not_found']:.2f} | found, lost {x['found_lost']:.2f} | kept, unused {x['kept_unused']:.2f} | recovered {x['recovered']:.2f}"
                      f" || median time to first perfect birth {x['t_found']}, perfect births per niche {x['perfect_births']:.0f}, median time to >=16 copies {x['t_refill']}, taken when free {x['use_when_free']}, birth -> stable use {x['t_birth_to_use']}")
            for h in "CD":
                for b_ in "AB":
                    T[f"refilled {h}-{b_}"] = st(cls_all[h] - cls_all[b_]); print("  " + line(f"H34b refilled niches {h} - {b_}", T[f"refilled {h}-{b_}"]))
            T["refilled B-A"] = st(cls_all["B"] - cls_all["A"]); print("  " + line("refilled niches B - A", T["refilled B-A"]))
    out["verdict"] = ver
    for h in "CD":
        hard = ver[h]["hard"]["beats_both"]; others = all(ver[h][e]["not_worse"] for e in ("stat", "mild")); strict = all(ver[h][e]["beats_both"] for e in ENVS)
        out[f"complementarity_{h}"] = bool(hard and others); out[f"strict_{h}"] = bool(strict)
        print(f"\n{h}: beats both A and B after the hard shift: {hard}; not worse than the best single algorithm elsewhere: {others}; complementarity shown: {hard and others}; strict (all three): {strict}")
        print("   ", {e: (x["beats_A"], x["beats_B"], x["best_single"], round(x["vs_best"], 4), round(x["t"], 1)) for e, x in ver[h].items()})
    save("h34_hybrid", out)
