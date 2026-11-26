"""H32 stages 2-3: the colony (NI) against the evolutionary search with trials (ES), ES with the colony's eviction (EC)
and the colony with trials (NT). Criteria in notes/hypotheses.md. Written before the runs finished."""
import os

import numpy as np

from common import ROOT, save
from h25_analyze import st

TAGS = ("NI", "ES", "EC", "NT")
ENV = {"orig": range(80, 88), "b24": range(80, 85), "l7": range(80, 85)}
SEG = {"S1": (40, 80), "S2": (80, 120), "S3": (120, 160)}
Q, DELTA = 0.95 ** 3, 0.015


def sg(y, s):
    return y[:, SEG[s][0]:SEG[s][1]].mean(1)


def line(name, d):
    return f"{name}: {d['mean']:+.4f} sd {d['sd']:.4f} se {d['se']:.4f} t {d['t']:.1f} wins {d['wins']}/{d['n']} t>=2.5={abs(d['t']) >= 2.5} 2sd={d['effect_2sd']}"


def niche_counts(f):
    tg = f["targets"]; return np.array([[[(g == tg[k, j]).all(1).sum() for j in range(tg.shape[1])] for k in range(tg.shape[0])] for g in f["genes"]])      # snapshots x K x roles


if __name__ == "__main__":
    out = {}; verdict = {}
    for env, rng in ENV.items():
        D = os.path.join(ROOT, ".cache", "h32", env)
        seeds = [s for s in rng if all(os.path.exists(os.path.join(D, f"{t}_{s}.npz")) for t in TAGS)]
        F = {t: [np.load(os.path.join(D, f"{t}_{s}.npz")) for s in seeds] for t in TAGS}; n = len(seeds)
        G = lambda t, k: np.array([f[k] for f in F[t]])
        assert all(np.array_equal(F["NI"][i]["targets"], F[t][i]["targets"]) and np.array_equal(F["NI"][i]["genes0"], F[t][i]["genes0"]) for t in TAGS for i in range(n))
        R = {t: G(t, "R") for t in TAGS}
        o = out[env] = dict(seeds=seeds, wall_min={t: float(np.mean(G(t, "wall")) / 60) for t in TAGS})
        print(f"\n=== {env}: seeds {seeds}, wall min " + " ".join(f"{t} {o['wall_min'][t]:.0f}" for t in TAGS))
        o["curve_100k"] = {t: R[t].reshape(n, 16, 10).mean(2).mean(0).tolist() for t in TAGS}
        o["cum_100k"] = {t: (np.cumsum(R[t].reshape(n, 16, 10).mean(2), 1) * 1e5).mean(0).tolist() for t in TAGS}
        for t in TAGS:
            print(f"  {t:<3}", " ".join(f"{x:.3f}" for x in o["curve_100k"][t]))
        o["levels"] = {}
        for t in TAGS:
            lv = o["levels"][t] = dict(**{s: st(sg(R[t], s)) for s in SEG}, d32=st(sg(R[t], "S3") - sg(R[t], "S2")), total=st(R[t].mean(1)), first400k=st(R[t][:, :40].mean(1)))
            print(f"  {t:<3} " + " ".join(f"{s} {lv[s]['mean']:.3f}±{lv[s]['sd']:.3f}" for s in SEG) + f" | whole run {lv['total']['mean']:.3f} | first 400k {lv['first400k']['mean']:.3f} | d32 {lv['d32']['mean']:+.4f} (t {lv['d32']['t']:.1f})")
        T = o["tests"] = {}
        for name, a, b in (("H32a NI - ES", "NI", "ES"), ("H32b EC - ES", "EC", "ES"), ("H32c NI - EC", "NI", "EC"), ("H32d NT - NI", "NT", "NI"), ("NT - ES", "NT", "ES")):
            T[name + " S3"] = st(sg(R[a], "S3") - sg(R[b], "S3")); print("  " + line(name + " on S3", T[name + " S3"]))
            T[name + " whole run"] = st(R[a].mean(1) - R[b].mean(1)); print("  " + line(name + " whole run", T[name + " whole run"]))
            T[name + " first 400k"] = st(R[a][:, :40].mean(1) - R[b][:, :40].mean(1)); print("  " + line(name + " first 400k", T[name + " first 400k"]))
        d = T["H32a NI - ES S3"]; verdict[env] = "NI > ES" if d["mean"] > 0 and d["t"] >= 2.5 else "ES > NI" if d["mean"] < 0 and d["t"] <= -2.5 else "indistinguishable (within delta)" if abs(d["mean"]) <= DELTA else "not established"
        print(f"  verdict H32a on S3: {verdict[env]}")
        print("  per-seed S3: " + " | ".join(f"{t} " + " ".join(f"{x:.3f}" for x in sg(R[t], "S3")) for t in TAGS))
        print("  stage 3 (means over seeds):"); o["stage3"] = {}
        for t in TAGS:
            ev = G(t, "ev"); tot = ev.sum(1); blk = int(F[t][0]["blk"]); late = ev[:, 120:].sum(1)
            cnt = np.array([niche_counts(f) for f in F[t]]); ts = F[t][0]["t"]                       # seeds x snapshots x K x roles
            first = np.array([[next((ts[i] for i in range(len(ts)) if c[i, k, j] > 0), np.nan) for k in range(c.shape[1]) for j in range(c.shape[2])] for c in cnt])
            full = np.array([[next((ts[i] for i in range(len(ts)) if c[i, k, j] >= 16), np.nan) for k in range(c.shape[1]) for j in range(c.shape[2])] for c in cnt])
            g = [f["genes"][-1] for f in F[t]]
            effc = [np.exp(-(p * np.log(p)).sum()) for p in (np.unique(x, axis=0, return_counts=True)[1] / len(x) for x in g)]
            s3 = dict(perf_births_per_100=float((tot[:, 1] / (160 * blk) * 100).mean()), perf_share_of_births=float((tot[:, 1] / tot[:, 0]).mean()),
                      perf_dead_long_share=float((tot[:, 5] / np.maximum(tot[:, 4], 1)).mean()), zero_use_share=float((tot[:, 2] / tot[:, 0]).mean()),
                      uses_of_evicted=float((tot[:, 3] / tot[:, 0]).mean()), uses_of_evicted_S3=float((late[:, 3] / late[:, 0]).mean()),
                      subs_per_task=float((G(t, "subs").sum(1) / (160 * blk)).mean()), skipped=float(np.mean(G(t, "n_skip"))),
                      niches_found=float(np.isfinite(first).sum(1).mean()), t_first_median=float(np.nanmedian(first)), t_all_found=float(np.mean([np.nanmax(x) if np.isfinite(x).all() else np.nan for x in first])),
                      niches_ge16_end=float((cnt[:, -1] >= 16).sum((1, 2)).mean()), t_ge16_median=float(np.nanmedian(full)), seeds_all_ge16=int(np.isfinite(full).all(1).sum()),
                      perfect_copies_end=float(cnt[:, -1].sum((1, 2)).mean()), largest_niche_end=float(cnt[:, -1].max((1, 2)).mean()), eff_configs_end=float(np.mean(effc)),
                      min_free_S3=float(G(t, "min_free")[:, 120:].mean()), competence_S3=float((Q - G(t, "q_pop"))[:, 120:].mean()),
                      busy_S3=float((G(t, "q_pop") - G(t, "q_avail"))[:, 120:].mean()), routing_S3=float((G(t, "q_avail") - G(t, "q_team"))[:, 120:].mean()),
                      model_S3=float((G(t, "er_best") - G(t, "er"))[:, 120:].mean()), state_size=float(np.mean(G(t, "state"))))
            o["stage3"][t] = s3
            print(f"    {t:<3} " + " | ".join(f"{k} {v:.3g}" for k, v in s3.items()))
        o["loss_tests"] = {}
        for name, a, b in (("NI - ES", "NI", "ES"), ("EC - ES", "EC", "ES"), ("NI - EC", "NI", "EC")):
            L = lambda t, k: {"competence": Q - G(t, "q_pop"), "busy": G(t, "q_pop") - G(t, "q_avail"), "routing": G(t, "q_avail") - G(t, "q_team"), "model": G(t, "er_best") - G(t, "er")}[k][:, 120:].mean(1)
            o["loss_tests"][name] = {k: st(L(a, k) - L(b, k)) for k in ("competence", "busy", "routing", "model")}
            print(f"  losses on S3, {name}: " + " | ".join(f"{k} {v['mean']:+.4f} (t {v['t']:.1f})" for k, v in o["loss_tests"][name].items()))
    out["verdict"] = verdict; print("\nH32a by environment:", verdict)
    save("h32_mechanisms", out)
