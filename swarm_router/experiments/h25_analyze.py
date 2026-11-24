"""H25 analysis: plateau tests, scenarios, break-even and composition metrics (criteria in notes/hypotheses.md)."""
import os, sys

import numpy as np

from common import ROOT, save

OUT = os.path.join(ROOT, ".cache", "h25")
DELTA, BLK, SNAP = 0.015, 10_000, 20_000
SEG = {"S0": (200_000, 400_000), "S1": (400_000, 800_000), "S2": (800_000, 1_200_000), "S3": (1_200_000, 1_600_000)}
VARS = ("C", "A", "PN")


def st(d):
    d = np.asarray(d, float); n = len(d); sd = d.std(ddof=1) if n > 1 else float("nan"); se = sd / np.sqrt(n)
    return dict(mean=float(d.mean()), sd=float(sd), se=float(se), n=n, wins=int((d > 0).sum()),
                effect_se=bool(abs(d.mean()) > 2 * se), effect_2sd=bool(abs(d.mean()) > 2 * sd), t=float(d.mean() / se) if se > 0 else float("nan"))


def seg(y, name, step=BLK):                    # y: seeds x points; mean over a segment
    a, b = SEG[name]; return y[:, a // step:b // step].mean(1)


def slope(y, a, b, blk=50_000):                # per-seed OLS slope per 100k tasks over 50k blocks in (a, b]
    z = y[:, a // BLK:b // BLK].reshape(len(y), -1, blk // BLK).mean(2); x = (np.arange(z.shape[1]) + 0.5) * blk / 100_000
    return np.polyfit(x, z.T, 1)[0]


def verdict(y):                                # plateau / growth / degradation on S2 -> S3 (and S1 -> S2)
    d32, d21 = st(seg(y, "S3") - seg(y, "S2")), st(seg(y, "S2") - seg(y, "S1")); sl = st(slope(y, 800_000, 1_600_000))
    flat32 = abs(d32["mean"]) + 2 * d32["se"] < DELTA; flat21 = abs(d21["mean"]) + 2 * d21["se"] < DELTA
    if flat32 and not sl["effect_se"]:
        v = "plateau (from <=800k)" if flat21 else "plateau (S2-S3)"
    elif d32["effect_se"]:
        v = "still growing" if d32["mean"] > 0 else "degrading"
    else:
        v = "undetermined"
    return dict(verdict=v, d32=d32, d21=d21, slope_S2S3_per100k=sl, slopes_per100k={k: st(slope(y, *SEG[k])) for k in SEG},
                levels={k: st(seg(y, k)) for k in SEG})


def comp(f):                                   # composition metrics per snapshot
    g, lin, ep, tg = f["genes"], f["lin"], f["epoch"], f["targets"]; S, N, L = g.shape
    code = (g.astype(np.int64) * (4 ** np.arange(L))).sum(-1)
    m = dict(eff_cfg=[], n_cfg=[], eff_lin=[], n_lin=[], perfect_share=[], niches=[], best_match=[])
    for s in range(S):
        for key, ids in (("cfg", code[s]), ("lin", lin[s])):
            c = np.unique(ids, return_counts=True)[1] / N
            m["eff_" + key].append(1 / (c ** 2).sum()); m["n_" + key].append(len(c))
        mt = (g[s][None, :, None, :] == tg[:, None, :, :]).sum(-1)
        m["perfect_share"].append((mt == L).any((0, 2)).mean()); m["niches"].append(((mt == L).sum(1) > 0).sum()); m["best_match"].append(mt.max(1).mean())
    prev = np.concatenate([(f["genes0"].astype(np.int64) * (4 ** np.arange(L))).sum(-1)[None], code[:-1]]); pe = np.concatenate([np.zeros((1, N), int), ep[:-1]])
    m["net_change_20k"] = (code != prev).mean(1); m["slots_rewritten_20k"] = (ep != pe).mean(1); m["never_rewritten"] = (ep == 0).mean(1)
    k = 400_000 // SNAP                         # genotype mass retained over a 400k lag, total variation distance
    ret, tv = np.full(S, np.nan), np.full(S, np.nan)
    for s in range(k, S):
        ret[s] = np.isin(code[s], code[s - k]).mean()
        u = np.union1d(code[s], code[s - k]); tv[s] = 0.5 * np.abs((code[s][:, None] == u).mean(0) - (code[s - k][:, None] == u).mean(0)).sum()
    m["retained_400k"], m["tv_400k"] = ret, tv
    return {k_: np.asarray(v, float) for k_, v in m.items()}


if __name__ == "__main__":
    seeds = [sd for sd in range(20, 30) if all(os.path.exists(os.path.join(OUT, f"{v}_{sd}.npz")) for v in VARS)]
    F = {v: [np.load(os.path.join(OUT, f"{v}_{sd}.npz")) for sd in seeds] for v in VARS}
    R = {v: np.array([f["R"] for f in F[v]]) for v in VARS}; Q = {v: np.array([f["q"] for f in F[v]]) for v in VARS}
    out = dict(seeds=seeds, delta=DELTA, cost={v: dict(wall_min=float(np.mean([f["wall"] for f in F[v]]) / 60), state=float(np.mean([f["state"] for f in F[v]])),
                                                  births=float(np.mean([f["births"] for f in F[v]]))) for v in VARS})
    print("seeds", seeds, "| cost", out["cost"])
    out["curve_50k"] = {v: R[v].reshape(len(seeds), -1, 5).mean(2).mean(0).tolist() for v in VARS}
    out["curve_50k_sd"] = {v: R[v].reshape(len(seeds), -1, 5).mean(2).std(0, ddof=1).tolist() for v in VARS}
    print("\nreward by 100k block (mean over seeds):")
    for v in VARS:
        print(f"  {v:<3}", " ".join(f"{x:.3f}" for x in R[v].reshape(len(seeds), -1, 10).mean(2).mean(0)))
    out["horizons"] = {}
    for H in (200_000, 400_000, 800_000, 1_600_000):
        w = slice((H - 100_000) // BLK, H // BLK); lv = {v: R[v][:, w].mean(1) for v in VARS}
        out["horizons"][H] = dict(levels={v: st(lv[v]) for v in VARS}, C_minus_A=st(lv["C"] - lv["A"]), C_minus_PN=st(lv["C"] - lv["PN"]), PN_minus_A=st(lv["PN"] - lv["A"]),
                                  cum={f"C-{x}": st((R["C"] - R[x])[:, :H // BLK].sum(1) * BLK) for x in ("A", "PN")})
        o = out["horizons"][H]
        print(f"\nhorizon {H}: last 100k  " + "  ".join(f"{v} {o['levels'][v]['mean']:.3f}±{o['levels'][v]['sd']:.3f}" for v in VARS))
        for k in ("C_minus_A", "C_minus_PN", "PN_minus_A"):
            d = o[k]; print(f"   {k}: {d['mean']:+.3f} sd {d['sd']:.3f} se {d['se']:.3f} t {d['t']:.1f} wins {d['wins']}/{d['n']} SE={d['effect_se']} 2sd={d['effect_2sd']}")
        print("   cumulative reward difference:", {k: f"{d['mean']:+.0f} (se {d['se']:.0f}, wins {d['wins']}/{d['n']})" for k, d in o["cum"].items()})
    out["reward"] = {}
    print("\nplateau tests (reward):")
    for name, y in [("C", R["C"]), ("A", R["A"]), ("PN", R["PN"]), ("C-A", R["C"] - R["A"]), ("C-PN", R["C"] - R["PN"]), ("q C", Q["C"])]:
        o = out["reward"][name] = verdict(y)
        print(f"  {name:<5} {o['verdict']:<22} levels " + " ".join(f"{k} {o['levels'][k]['mean']:+.3f}(se {o['levels'][k]['se']:.3f})" for k in SEG)
              + f" | d21 {o['d21']['mean']:+.4f} se {o['d21']['se']:.4f} | d32 {o['d32']['mean']:+.4f} se {o['d32']['se']:.4f} t {o['d32']['t']:.1f} wins {o['d32']['wins']}"
              + f" | slope S2S3 {o['slope_S2S3_per100k']['mean']:+.5f} se {o['slope_S2S3_per100k']['se']:.5f}"
              + " | slopes/100k " + " ".join(f"{k} {o['slopes_per100k'][k]['mean']:+.4f}" for k in SEG))
    cs = R["C"].reshape(len(seeds), -1, 5).mean(2); ma = np.array([np.convolve(r, np.ones(3) / 3, "valid") for r in cs]); fin = seg(R["C"], "S3")
    tp = [int(next((b for b in range(ma.shape[1]) if np.all(np.abs(ma[i, b:] - fin[i]) < 2 * DELTA)), ma.shape[1])) * 50_000 for i in range(len(seeds))]
    out["plateau_time_C"] = dict(per_seed=tp, median=float(np.median(tp)), min=int(min(tp)), max=int(max(tp)))
    pk = cs.mean(0); out["peak_C"] = dict(block_end=int((pk.argmax() + 1) * 50_000), level=float(pk.max()), last_block=float(pk[-1]))
    print("\nplateau time of C (valid only if plateau established):", out["plateau_time_C"], "| peak 50k block:", out["peak_C"])
    out["breakeven"] = {}
    for x in ("A", "PN"):
        cum = np.cumsum(R["C"] - R[x], 1) * BLK; be = []
        for c in cum:
            neg = np.where(c < 0)[0]; be.append(None if len(neg) and neg[-1] == len(c) - 1 else int((neg[-1] + 2) * BLK if len(neg) else BLK))
        ok = [b for b in be if b is not None]; mc = cum.mean(0); negm = np.where(mc < 0)[0]
        out["breakeven"][x] = dict(per_seed=be, reached=len(ok), median_of_reached=float(np.median(ok)) if ok else None, deepest_loss=st(cum.min(1)),
                                   deepest_loss_time=float(np.median(cum.argmin(1) + 1) * BLK), final=st(cum[:, -1]),
                                   mean_curve_breakeven=None if len(negm) and negm[-1] == len(mc) - 1 else int((negm[-1] + 2) * BLK if len(negm) else BLK))
        o = out["breakeven"][x]
        print(f"break-even C vs {x}: reached {o['reached']}/{len(seeds)} seeds, median {o['median_of_reached']}, per seed {be}, mean-curve {o['mean_curve_breakeven']}, "
              f"deepest loss {o['deepest_loss']['mean']:.0f} at ~{o['deepest_loss_time']:.0f}, final {o['final']['mean']:+.0f} (se {o['final']['se']:.0f}, wins {o['final']['wins']})")
    out["composition"] = {}
    print("\ncomposition (segment means; d32 = S3 - S2):")
    for v in ("C", "PN", "A"):
        M = [comp(f) for f in F[v]]; out["composition"][v] = {}
        for k in M[0]:
            y = np.array([m[k] for m in M]); o = {s: st(np.nanmean(y[:, SEG[s][0] // SNAP:SEG[s][1] // SNAP], 1)) for s in SEG}
            o["first_100k"] = st(y[:, :5].mean(1)) if not np.isnan(y[:, :5]).any() else None
            o["d32"] = st(np.nanmean(y[:, SEG["S3"][0] // SNAP:SEG["S3"][1] // SNAP], 1) - np.nanmean(y[:, SEG["S2"][0] // SNAP:SEG["S2"][1] // SNAP], 1))
            o["end"] = st(y[:, -1]); o["curve_100k"] = np.nanmean(y.reshape(len(seeds), -1, 5), 2).mean(0).tolist(); out["composition"][v][k] = o
            if v != "A" or k in ("eff_cfg", "niches", "best_match", "perfect_share"):
                print(f"  {v:<3}{k:<20} " + " ".join(f"{s} {o[s]['mean']:.3f}" for s in SEG) + f" | end {o['end']['mean']:.3f}±{o['end']['sd']:.3f}"
                      f" | d32 {o['d32']['mean']:+.4f} se {o['d32']['se']:.4f} t {o['d32']['t']:.1f}")
    if "nosave" not in sys.argv:
        save("h25_long", out)
