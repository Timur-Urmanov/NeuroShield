"""H31 analysis: composition of the victims under the old (contrib) and the new (noinherit) eviction rule, computed
on the same colony states (criteria in notes/hypotheses.md). Written before the runs finished."""
import os

import numpy as np

from common import ROOT, save
from h25_analyze import st

D = os.path.join(ROOT, ".cache", "h31")
CAT = ("scarce", "abundant", "dominant", "imperfect")
W = slice(40, 160)                                                        # 0.4M - 1.6M


def line(name, d):
    return f"{name}: {d['mean']:+.4f} sd {d['sd']:.4f} se {d['se']:.4f} t {d['t']:.1f} wins {d['wins']}/{d['n']} t>=2.5={abs(d['t']) >= 2.5} 2sd={d['effect_2sd']}"


if __name__ == "__main__":
    seeds = [s for s in range(70, 75) if all(os.path.exists(os.path.join(D, f"{t}_{s}.npz")) for t in ("C", "NI"))]
    out = dict(seeds=seeds); print("seeds", seeds)
    for tag in ("C", "NI"):
        F = [np.load(os.path.join(D, f"{tag}_{s}.npz")) for s in seeds]
        n = np.array([f["vn"][W].sum() for f in F])[:, None]
        pop = np.array([f["pop"][W].sum(0) for f in F]) / n; prot = np.array([f["prot"][W].sum(0) for f in F]) / n
        vic = np.array([f["vict"][W].sum(0) for f in F]) / n[:, None]            # seeds x rule x cat, per generation
        use = np.array([f["vuse"][W].sum(0) for f in F]) / n; inh = np.array([f["vinh"][W].sum(0) for f in F]) / n
        o = out[tag] = dict(reward_S3=st(np.array([f["R"][120:].mean() for f in F])), pop=pop.mean(0).tolist(), protected_unused=prot.mean(0).tolist(),
                            victims_contrib=vic[:, 0].mean(0).tolist(), victims_noinherit=vic[:, 1].mean(0).tolist())
        print(f"\n=== states of {tag} (reward on S3 {o['reward_S3']['mean']:.3f}); per generation, window 0.4-1.6M; categories: " + " | ".join(CAT))
        print("  population            " + " ".join(f"{x:8.1f}" for x in pop.mean(0)))
        print("  protected unused      " + " ".join(f"{x:8.1f}" for x in prot.mean(0)) + f"   (share of population {prot.sum(1).mean() / pop.sum(1).mean():.3f})")
        print("  victims, contrib      " + " ".join(f"{x:8.2f}" for x in vic[:, 0].mean(0)))
        print("  victims, noinherit    " + " ".join(f"{x:8.2f}" for x in vic[:, 1].mean(0)))
        rate = vic / np.maximum(pop[:, None, :], 1e-9)                           # eviction hazard per member of the category, per generation
        print("  hazard, contrib       " + " ".join(f"{x:8.4f}" for x in rate[:, 0].mean(0)))
        print("  hazard, noinherit     " + " ".join(f"{x:8.4f}" for x in rate[:, 1].mean(0)))
        o["tests"] = {}
        for i, c in enumerate(CAT):
            o["tests"][f"victims {c}: contrib - noinherit"] = d = st(vic[:, 0, i] - vic[:, 1, i]); print("  " + line(f"victims per generation, {c}: contrib - noinherit", d))
        o["tests"]["own uses of victims: contrib - noinherit"] = d = st(use[:, 0] - use[:, 1]); print("  " + line("mean own uses of a victim: contrib - noinherit", d) + f"   (levels {use[:, 0].mean():.1f} vs {use[:, 1].mean():.1f})")
        o["tests"]["inherited credit of victims: contrib - noinherit"] = d = st(inh[:, 0] - inh[:, 1]); print("  " + line("mean inherited credit of a victim: contrib - noinherit", d))
        blk = lambda a: np.array([x.reshape(16, 10, *x.shape[1:]).sum(1) for x in a])      # per 100k
        vn = blk([f["vn"] for f in F]); pp = blk([f["pop"] for f in F]) / vn[..., None]; vv = blk([f["vict"] for f in F]) / vn[..., None, None]
        pr = blk([f["prot"] for f in F]) / vn[..., None]
        o["by_100k"] = dict(pop=pp.mean(0).tolist(), victims=vv.mean(0).tolist(), protected=pr.mean(0).tolist())
        print("  by 100k: scarce specialists in population    " + " ".join(f"{x:.0f}" for x in pp.mean(0)[:, 0]))
        print("  by 100k: scarce victims/generation, contrib  " + " ".join(f"{x:.2f}" for x in vv.mean(0)[:, 0, 0]))
        print("  by 100k: scarce victims/generation, noinher. " + " ".join(f"{x:.2f}" for x in vv.mean(0)[:, 1, 0]))
        print("  by 100k: protected unused (all categories)   " + " ".join(f"{x:.0f}" for x in pr.mean(0).sum(1)))
    save("h31_victims", out)
