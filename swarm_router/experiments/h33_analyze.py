"""H33 main test: controllers of the replacement rate (H = global, A = global without the alarm, Z = local) and the
controls (NI constant, F600 freeze, Q400 constant reduced rate), all branched from one NI trajectory.
Criteria in notes/hypotheses.md. Written before the runs finished."""
import os

import numpy as np

from common import ROOT, save
from h25_analyze import st

TAGS = ("NI", "F600", "Q400", "H400", "A400", "Z400")
ENVS = ("stat", "hard", "mild")
WIN = {"stat": {"0.6-1.6M": (60, 160), "S3": (120, 160)}, "hard": {"0.8-1.2M": (80, 120), "1.2-1.6M": (120, 160), "0.8-1.6M": (80, 160)}}
WIN["mild"] = WIN["hard"]; MAIN = {"stat": "S3", "hard": "0.8-1.6M", "mild": "0.8-1.6M"}
Q, DELTA = 0.95 ** 3, 0.015


def line(name, d):
    return f"{name}: {d['mean']:+.4f} sd {d['sd']:.4f} se {d['se']:.4f} t {d['t']:.1f} wins {d['wins']}/{d['n']} t>=2.5={abs(d['t']) >= 2.5} 2sd={d['effect_2sd']}"


if __name__ == "__main__":
    out = {}; verdict = {c: {} for c in ("H400", "Z400")}
    for env in ENVS:
        D = os.path.join(ROOT, ".cache", "h33", env)
        seeds = [s for s in range(120, 126) if all(os.path.exists(os.path.join(D, f"{t}_{s}.npz")) for t in TAGS)]
        F = {t: [np.load(os.path.join(D, f"{t}_{s}.npz")) for s in seeds] for t in TAGS}; n = len(seeds)
        G = lambda t, k: np.array([f[k] for f in F[t]])
        R = {t: G(t, "R") for t in TAGS}
        for t in TAGS[1:]:
            fk = 60 if t == "F600" else 40; assert np.allclose(R[t][:, :fk], R["NI"][:, :fk]), (env, t)
        o = out[env] = dict(seeds=seeds); print(f"\n=== {env}: seeds {seeds}")
        o["curve_100k"] = {t: R[t].reshape(n, 16, 10).mean(2).mean(0).tolist() for t in TAGS}
        o["births_100k"] = {t: (G(t, "ev")[:, :, 0].reshape(n, 16, 10).sum(2) / 1000).mean(0).tolist() for t in TAGS}      # replacements per generation
        for t in TAGS:
            print(f"  {t:<5} R " + " ".join(f"{x:.3f}" for x in o["curve_100k"][t]))
        for t in TAGS:
            print(f"  {t:<5} replacements/generation " + " ".join(f"{x:4.1f}" for x in o["births_100k"][t]))
        o["levels"] = {t: {w: st(R[t][:, a:b].mean(1)) for w, (a, b) in WIN[env].items()} for t in TAGS}
        for t in TAGS:
            print(f"  {t:<5} " + " | ".join(f"{w} {v['mean']:.4f}±{v['sd']:.4f}" for w, v in o["levels"][t].items()))
        T = o["tests"] = {}
        for t in TAGS[1:]:
            for w, (a, b) in WIN[env].items():
                T[f"{t}-NI {w}"] = st(R[t][:, a:b].mean(1) - R["NI"][:, a:b].mean(1)); print("  " + line(f"{t} - NI on {w}", T[f"{t}-NI {w}"]))
        a, b = WIN[env][MAIN[env]]; m = lambda t: R[t][:, a:b].mean(1)
        for c in ("H400", "A400", "Z400"):
            for k in ("F600", "Q400"):
                T[f"{c}-{k}"] = st(m(c) - m(k)); print("  " + line(f"{c} - {k} on {MAIN[env]}", T[f"{c}-{k}"]))
        T["H400-A400"] = st(m("H400") - m("A400")); print("  " + line(f"H400 - A400 (value of the alarm) on {MAIN[env]}", T["H400-A400"]))
        best = max(("NI", "F600", "Q400"), key=lambda t: m(t).mean()); o["best_fixed"] = best
        for c in ("H400", "Z400"):
            d = st(m(c) - m(best)); T[f"{c}-best_fixed"] = d; beats_all = all(st(m(c) - m(k))["mean"] > 0 and st(m(c) - m(k))["t"] >= 2.5 for k in ("NI", "F600", "Q400"))
            verdict[c][env] = dict(best_fixed=best, diff=d["mean"], t=d["t"], not_worse=bool(d["mean"] >= -DELTA), beats_all=bool(beats_all))
            print("  " + line(f"{c} - best fixed control ({best}) on {MAIN[env]}", d) + f" | not worse than delta: {d['mean'] >= -DELTA} | beats every fixed control: {beats_all}")
        if env == "stat":
            gain = st(m("F600") - m("NI")); o["freeze_gain"] = gain
            for c in ("H400", "A400", "Z400"):
                o[f"share_of_freeze_gain_{c}"] = float((m(c) - m("NI")).mean() / gain["mean"]) if gain["mean"] else float("nan")
            print("  share of the freeze gain taken: " + " ".join(f"{c} {o[f'share_of_freeze_gain_{c}']:.2f}" for c in ("H400", "A400", "Z400")) + f" (freeze gain {gain['mean']:+.4f}, t {gain['t']:.1f})")
        print("  per-seed, main window: " + " | ".join(f"{t} " + " ".join(f"{x:.3f}" for x in m(t)) for t in TAGS))
        o["loss"] = {}
        for t in TAGS:
            L = dict(competence=Q - G(t, "q_pop"), busy=G(t, "q_pop") - G(t, "q_avail"), routing=G(t, "q_avail") - G(t, "q_team"), model=G(t, "er_best") - G(t, "er"))
            o["loss"][t] = {k: float(y[:, a:b].mean()) for k, y in L.items()}
            o["loss"][t]["births_total_k"] = float(G(t, "ev")[:, :, 0].sum(1).mean() / 1000); o["loss"][t]["alarm_generations"] = float(np.mean(G(t, "n_alarm")))
            print(f"  {t:<5} losses on {MAIN[env]}: " + " | ".join(f"{k} {v:.4g}" for k, v in o["loss"][t].items()))
        if env != "stat":                                                 # recovery of the changed niches (hidden; diagnostics)
            o["recovery"] = {}
            for t in TAGS:
                rows = []
                for f in F[t]:
                    tg, t0 = f["targets"], f["targets0"]; ch = (tg != t0).any(-1); ts = f["t"]
                    cnt = np.array([[[(g == tg[k, j]).all(1).sum() for j in range(3)] for k in range(3)] for g in f["genes"]])[:, ch]      # snapshots x changed
                    rec = [next((ts[i] for i in range(len(ts)) if ts[i] > 800000 and cnt[i, c] >= 16), np.nan) for c in range(cnt.shape[1])]
                    rows.append((int(np.isfinite(rec).sum()), float(np.nanmean(rec)) if np.isfinite(rec).any() else np.nan, int((cnt[-1] > 0).sum())))
                rows = np.array(rows, float); o["recovery"][t] = dict(niches_refilled=float(rows[:, 0].mean()), t_refill=float(np.nanmean(rows[:, 1])) if np.isfinite(rows[:, 1]).any() else None, niches_found=float(rows[:, 2].mean()))
                print(f"  {t:<5} changed niches refilled (>=16 copies) {rows[:, 0].mean():.2f} of 3, found {rows[:, 2].mean():.2f}, mean refill time {o['recovery'][t]['t_refill']}")
            lag = []
            for f in F["H400"]:                                           # first block after the shift with full-rate evolution
                bl = f["ev"][:, 0]; lag.append(next((10 * (i - 80) for i in range(80, 160) if bl[i] >= 0.9 * 3300), np.nan))
            o["resume_lag_k"] = lag; print("  H400: thousand tasks after the shift until replacements are back at >= 90 % of the full rate, per seed:", lag)
    out["verdict"] = verdict
    for c in ("H400", "Z400"):
        ok = all(v["not_worse"] for v in verdict[c].values()) and any(v["beats_all"] for v in verdict[c].values())
        out[f"self_regulation_works_{c}"] = bool(ok); print(f"\n{c}: not worse than the best fixed control by more than delta in every environment and better than every fixed control in at least one: {ok}")
        print("   ", {e: (v["best_fixed"], round(v["diff"], 4), round(v["t"], 1)) for e, v in verdict[c].items()})
    save("h33_selfregulation", out)
