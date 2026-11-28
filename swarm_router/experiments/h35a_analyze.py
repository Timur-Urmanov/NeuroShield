"""H35 diagnostic stage. A: tau_route - recovery of the frozen colony's routing after controlled perturbations.
B: tau_pop - lifetimes, uses, establishment and turnover on the NI trajectory. Definitions in notes/hypotheses.md."""
import os

import numpy as np

from common import ROOT, save
from h25_analyze import st

D = os.path.join(ROOT, ".cache", "h35a"); SEEDS = [s for s in range(140, 146) if os.path.exists(os.path.join(D, f"S50_{s}.npz"))]
KINDS = ("R05", "R20", "R50", "G05", "G20", "G50", "S05", "S20", "S50"); Q = 0.95 ** 3; BLK = 7500; F0 = 800000 // BLK + 1


def series(f):
    """(time after the fork, dict of per-bin metrics): 250-task bins for 50k tasks, then 7500-task blocks."""
    fi = f["fine"]; n = np.maximum(fi[:, 0], 1); fine = dict(q_team=fi[:, 4] / n, q_avail=fi[:, 3] / n, q_pop=fi[:, 2] / n, model=(fi[:, 6] - fi[:, 5]) / n, succ=fi[:, 1] / n)
    co = dict(q_team=f["q_team"][F0 + 6:], q_avail=f["q_avail"][F0 + 6:], q_pop=f["q_pop"][F0 + 6:], model=(f["er_best"] - f["er"])[F0 + 6:], succ=f["succ"][F0 + 6:])
    t = np.concatenate([np.arange(200) * 250 + 125, (np.arange(F0 + 6, 160) + 0.5) * BLK - 800000])
    return t, {k: np.concatenate([fine[k], co[k]]) for k in fine}


def tau(t, d, frac):
    """first time from which the loss stays at or below frac * initial loss (initial = mean of the first 500 tasks); smoothed over 2000 tasks"""
    d0 = d[:2].mean()
    if d0 <= 0:
        return np.nan, d0
    sm = np.concatenate([np.convolve(d[:200], np.ones(8) / 8, mode="same"), d[200:]])
    ok = sm <= frac * d0
    for i in range(len(ok)):
        if ok[i:].all():
            return float(t[i]), d0
    return np.inf, d0


if __name__ == "__main__":
    out = dict(seeds=SEEDS); print("seeds", SEEDS)
    C = [series(np.load(os.path.join(D, f"C0_{s}.npz"))) for s in SEEDS]; t = C[0][0]
    print("\nstage A: loss of team quality against the frozen unperturbed control (mean over seeds); 'inf' = not within the 400k horizon")
    out["A"] = {}
    for k in KINDS:
        P = [series(np.load(os.path.join(D, f"{k}_{s}.npz")))[1] for s in SEEDS]
        d = np.array([C[i][1]["q_team"] - P[i]["q_team"] for i in range(len(SEEDS))])
        comp = {c: np.array([P[i][c] - C[i][1][c] for i in range(len(SEEDS))]) for c in ("q_pop", "q_avail", "model")}
        rout = np.array([(P[i]["q_avail"] - P[i]["q_team"]) - (C[i][1]["q_avail"] - C[i][1]["q_team"]) for i in range(len(SEEDS))])
        busy = np.array([(P[i]["q_pop"] - P[i]["q_avail"]) - (C[i][1]["q_pop"] - C[i][1]["q_avail"]) for i in range(len(SEEDS))])
        win = lambda y, a, b: y[:, (t >= a) & (t < b)].mean(1)
        t50 = [tau(t, x, 0.5)[0] for x in d]; t90 = [tau(t, x, 0.1)[0] for x in d]; dm = d.mean(0); m50, _ = tau(t, dm, 0.5); m90, d0 = tau(t, dm, 0.1)
        o = out["A"][k] = dict(initial=st(win(d, 0, 500)), at_2k=st(win(d, 1500, 2500)), at_10k=st(win(d, 8000, 12000)), at_50k=st(win(d, 40000, 50000)), at_200k=st(win(d, 150000, 250000)), last_100k=st(win(d, 300000, 400001)),
                               tau50_mean_curve=m50, tau90_mean_curve=m90, tau50_per_seed=t50, tau90_per_seed=t90,
                               init_competence=float(-win(comp["q_pop"], 0, 500).mean()), init_busy=float(win(busy, 0, 500).mean()), init_routing=float(win(rout, 0, 500).mean()), init_model=float(win(comp["model"], 0, 500).mean()),
                               last_competence=float(-win(comp["q_pop"], 300000, 400001).mean()), last_routing=float(win(rout, 300000, 400001).mean()))
        print(f"  {k}: initial {o['initial']['mean']:+.4f} (t {o['initial']['t']:.1f}) | 2k {o['at_2k']['mean']:+.4f} | 10k {o['at_10k']['mean']:+.4f} | 50k {o['at_50k']['mean']:+.4f} | 200k {o['at_200k']['mean']:+.4f} | last 100k {o['last_100k']['mean']:+.4f} (t {o['last_100k']['t']:.1f})"
              f" || tau50 {m50} tau90 {m90} (mean curve); per seed tau50 {t50} tau90 {t90}")
        print(f"       initial loss by component: competence {o['init_competence']:+.4f} busy {o['init_busy']:+.4f} routing {o['init_routing']:+.4f} model(reward) {o['init_model']:+.4f} | last 100k: competence {o['last_competence']:+.4f} routing {o['last_routing']:+.4f}")
    print("\nstage B: NI trajectory, 0.4-1.2M"); B = out["B"] = {}; rows = []
    for s in SEEDS:
        f = np.load(os.path.join(D, f"NI_{s}.npz")); e = f["evs"]; e = e[e[:, 4] == 0]; life, uses, fs, perf = e[:, 0], e[:, 1], e[:, 2], e[:, 3].astype(bool)
        g = f["genes"]; ep = f["epoch"]; ts = f["t"]; tg = f["targets"]; i0 = [i for i in range(len(ts)) if ts[i] >= 400000]
        surv, ovl, novl = {}, {}, {}
        def dist(x):
            u, c = np.unique(x, axis=0, return_counts=True); return {a.tobytes(): b / len(x) for a, b in zip(u, c)}
        dd = {i: dist(g[i]) for i in i0}; nn = {i: np.array([[(g[i] == tg[k, j]).all(1).sum() for j in range(3)] for k in range(3)]).ravel() / len(g[i]) for i in i0}
        for lag in range(1, 41):
            pr = [(i, i + lag) for i in i0 if i + lag in dd]
            if pr:
                surv[lag] = np.mean([(ep[a] == ep[b]).mean() for a, b in pr]); ovl[lag] = np.mean([sum(min(v, dd[b].get(k, 0)) for k, v in dd[a].items()) for a, b in pr])
                novl[lag] = np.mean([np.minimum(nn[a], nn[b]).sum() / max(nn[a].sum(), 1e-9) for a, b in pr])
        step = int(ts[1] - ts[0]); half = lambda c: next((lag * step for lag in sorted(c) if c[lag] <= 0.5), np.inf)
        stt = f["stat"][54:]; g16 = f["g16"]; g16 = g16[g16[:, 0] >= 400000]
        rows.append(dict(life_median=float(np.median(life)), life_mean=float(life.mean()), life_p90=float(np.percentile(life, 90)), life_p99=float(np.percentile(life, 99)),
                         uses_median=float(np.median(uses)), uses_mean=float(uses.mean()), zero_use=float((uses == 0).mean()), ever_success=float((fs >= 0).mean()),
                         first_success_age_median=float(np.median(fs[fs >= 0])), long_5000=float((life >= 5000).mean()), perfect_share=float(perf.mean()),
                         life_median_perfect=float(np.median(life[perf])), life_mean_used=float(life[uses > 0].mean()),
                         est_n=float(len(g16)), est_time_median=float(np.median(g16[:, 1] - g16[:, 0])) if len(g16) else np.nan,
                         surv_15k=float(surv[1]), surv_half=half(surv), genotype_overlap_15k=float(ovl[1]), genotype_overlap_half=half(ovl), genotype_overlap_300k=float(ovl[20]), niche_overlap_15k=float(novl[1]), niche_overlap_300k=float(novl[20]),
                         inherited_share=float(stt[:, 0].mean()), state_k=float(stt[:, 1].mean() / 1000), rows_dropped_per_gen=float(stt[:, 2].mean() / (BLK / 100)), rows_inherited_per_gen=float(stt[:, 3].mean() / (BLK / 100)),
                         pair_updates_per_gen=300.0, used_primitives=float(stt[:, 4].mean())))
    for k in rows[0]:
        v = np.array([r[k] for r in rows], float); B[k] = dict(mean=float(np.mean(v[np.isfinite(v)])) if np.isfinite(v).any() else None, per_seed=v.tolist()); print(f"  {k:<28} {B[k]['mean']}   per seed {np.round(v, 3).tolist()}")
    B["tau_turnover_nominal"] = 640 * 100 / 33; B["weighted_instance_life"] = None
    print(f"\n  nominal turnover N*K/n = {B['tau_turnover_nominal']:.0f} tasks")
    save("h35a_timescales", out)
