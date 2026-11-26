"""H30 analysis: where the routing loss of NI sits (freeze / true task type / both, branched at 800k).
Criteria in notes/hypotheses.md. Written before the runs finished."""
import os

import numpy as np

from common import ROOT, save
from h25_analyze import st

D = os.path.join(ROOT, ".cache", "h30")
TAGS = ("NI", "F800", "O800", "G800")
S3 = slice(120, 160)
Q = 0.95 ** 3


def line(name, d):
    return f"{name}: {d['mean']:+.4f} sd {d['sd']:.4f} se {d['se']:.4f} t {d['t']:.1f} wins {d['wins']}/{d['n']} t>=2.5={abs(d['t']) >= 2.5} 2sd={d['effect_2sd']}"


if __name__ == "__main__":
    seeds = [s for s in range(60, 68) if all(os.path.exists(os.path.join(D, f"{t}_{s}.npz")) for t in TAGS)]
    F = {t: [np.load(os.path.join(D, f"{t}_{s}.npz")) for s in seeds] for t in TAGS}; n = len(seeds)
    G = lambda t, k: np.array([f[k] for f in F[t]])
    assert all(np.allclose(G(t, "R")[:, :80], G("NI", "R")[:, :80]) for t in TAGS)
    M = {t: dict(reward=G(t, "R"), q_team=G(t, "q_team"), competence=Q - G(t, "q_pop"), busy=G(t, "q_pop") - G(t, "q_avail"),
                 routing=G(t, "q_avail") - G(t, "q_team"), model=G(t, "er_best") - G(t, "er")) for t in TAGS}
    out = dict(seeds=seeds); print("seeds", seeds)
    print("reward by 100k block:"); out["curve_100k"] = {}
    for t in TAGS:
        c = M[t]["reward"].reshape(n, 16, 10).mean(2).mean(0); out["curve_100k"][t] = c.tolist(); print(f"  {t:<5}", " ".join(f"{x:.3f}" for x in c))
    out["S3"] = {t: {k: st(y[:, S3].mean(1)) for k, y in M[t].items()} for t in TAGS}
    print("\nS3 levels (mean ± sd):")
    for t in TAGS:
        print(f"  {t:<5} " + " | ".join(f"{k} {v['mean']:.4f}±{v['sd']:.4f}" for k, v in out["S3"][t].items()))
    out["minus_NI"] = {}
    for t in TAGS[1:]:
        o = out["minus_NI"][t] = {k: st(M[t][k][:, S3].mean(1) - M["NI"][k][:, S3].mean(1)) for k in M[t]}
        print(f"  {t} - NI on S3:")
        for k in ("reward", "routing", "competence", "busy", "model", "q_team"):
            print("    " + line(k, o[k]))
    r0 = M["NI"]["routing"][:, S3].mean(1)
    out["share_removed"] = {t: float(-out["minus_NI"][t]["routing"]["mean"] / r0.mean()) for t in TAGS[1:]}
    print("\nshare of NI routing loss removed:", {t: round(v, 2) for t, v in out["share_removed"].items()})
    inter = (M["G800"]["routing"] - M["F800"]["routing"] - M["O800"]["routing"] + M["NI"]["routing"])[:, S3].mean(1)
    out["interaction_routing"] = st(inter); print(line("interaction F x O on routing loss", out["interaction_routing"]))
    inter = (M["G800"]["reward"] - M["F800"]["reward"] - M["O800"]["reward"] + M["NI"]["reward"])[:, S3].mean(1)
    out["interaction_reward"] = st(inter); print(line("interaction F x O on reward", out["interaction_reward"]))
    print("\nwithout intervention (S3): cluster purity; routing loss by cluster ok / wrong and by young member / none (share of tasks)")
    out["split"] = {}
    for t in TAGS:
        cf = G(t, "conf")[:, S3].sum(1); pur = cf.max(2).sum(1) / cf.sum((1, 2))
        sp = G(t, "split")[:, S3].sum(1); loss = sp[:, :, 0] / np.maximum(sp[:, :, 1], 1); share = sp[:, :, 1] / sp[:, :2, 1].sum(1, keepdims=True)
        o = out["split"][t] = dict(purity=float(pur.mean()), loss=loss.mean(0).tolist(), share=share.mean(0).tolist(),
                                   contrib=(loss * share).mean(0).tolist())
        print(f"  {t:<5} purity {o['purity']:.3f} | loss ok {o['loss'][0]:.3f} wrong {o['loss'][1]:.3f} young {o['loss'][2]:.3f} old {o['loss'][3]:.3f}"
              f" | share ok {o['share'][0]:.3f} wrong {o['share'][1]:.3f} young {o['share'][2]:.3f} old {o['share'][3]:.3f}"
              f" | contribution wrong {o['contrib'][1]:.4f} young {o['contrib'][2]:.4f}")
    print("\ncategories of tasks on S3 (absent | busy | routing | full team) and success in each:")
    for t in TAGS:
        cn = G(t, "cat_n")[:, S3].sum(1); cs = G(t, "cat_s")[:, S3].sum(1)
        print(f"  {t:<5} share " + " ".join(f"{x:.3f}" for x in (cn / cn.sum(1, keepdims=True)).mean(0)) + " | success " + " ".join(f"{x:.3f}" for x in (cs / np.maximum(cn, 1)).mean(0)))
    print("\nper-seed S3 reward: " + " | ".join(f"{t} " + " ".join(f"{x:.3f}" for x in M[t]["reward"][:, S3].mean(1)) for t in TAGS))
    save("h30_routing", out)
