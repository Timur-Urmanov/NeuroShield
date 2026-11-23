"""H18: forgetting in genotype records and in link statistics after a role shift (notes/hypotheses.md)."""
import json, os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import ROOT, save
from h16_shift import BUSY, DEV, FULL, N_SWARM, job, paired
from swarm import EnvConfig, SwarmConfig

T, TEST, DS = 8000, list(range(5)), (0.8, 0.9, 0.97)
CELLS = {"shift 1.0": 1.0, "shift 0.3": 0.3, "no shift": 0.0}
VARS = {"none": {}}
for d in DS:
    VARS.update({f"geno {d}": dict(forget_geno=d), f"links {d}": dict(forget_links=d), f"both {d}": dict(forget_geno=d, forget_links=d)})

if __name__ == "__main__":
    keys = [(c, v, sd) for c in CELLS for v in VARS for sd in ([*DEV, *TEST] if c == "shift 1.0" else TEST)]
    specs = [("swarm", EnvConfig("roles", T, role_by_type=True, busy=BUSY, role_shift=CELLS[c]),
              SwarmConfig(N=N_SWARM, **FULL, **VARS[v]), sd) for c, v, sd in keys]
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = dict(zip(keys, ex.map(job, specs, chunksize=1)))
    col = lambda c, v, m, S=TEST: [res[(c, v, sd)][m] for sd in S]
    dev = {v: float(np.mean(col("shift 1.0", v, "post", DEV))) for v in VARS if v != "none"}
    dev_rec = {v: float(np.mean(col("shift 1.0", v, "rec", DEV))) for v in VARS}
    best = max(dev, key=dev.get); d = best.split()[1]
    print("dev post (shift 1.0):", {k: round(x, 3) for k, x in dev.items()}, "none:", round(float(np.mean(col("shift 1.0", "none", "post", DEV))), 3))
    print("chosen:", best)
    bandit = {tuple(r["key"]): r["res"] for r in map(json.loads, open(os.path.join(ROOT, ".cache", "h16b.jsonl")))}
    bname = "cluster_pair_mean|640|reset|-"
    out = dict(dev_post=dev, dev_rec=dev_rec, chosen=best, cells={})
    for c in CELLS:
        tab = {v: {m: dict(mean=float(np.mean(col(c, v, m))), sd=float(np.std(col(c, v, m), ddof=1))) for m in ("pre", "rec", "post", "cover_end")}
               for v in VARS}
        cmp_ = {f"{v} - none": {m: paired(col(c, v, m), col(c, "none", m)) for m in ("rec", "post")}
                for v in (best, f"geno {d}", f"links {d}", f"both {d}")}
        if c == "shift 1.0":
            b = lambda m: [bandit[(c, bname, sd)][m] for sd in TEST]
            cmp_[f"{best} - bandit [{bname}]"] = {m: paired(col(c, best, m), b(m)) for m in ("rec", "post")}
            cmp_[f"none - bandit [{bname}]"] = {m: paired(col(c, "none", m), b(m)) for m in ("rec", "post")}
        out["cells"][c] = dict(table=tab, comparisons=cmp_)
        print(f"\n## {c}")
        for v, e in tab.items():
            print(f"{v:<11} pre={e['pre']['mean']:.3f} rec={e['rec']['mean']:.3f} post={e['post']['mean']:.3f}±{e['post']['sd']:.3f} cover={e['cover_end']['mean']:.2f}")
        for k, v in cmp_.items():
            print(f"  {k}: rec {v['rec']['diff']:+.3f}±{v['rec']['sd']:.3f} effect={v['rec']['effect']}; post {v['post']['diff']:+.3f}±{v['post']['sd']:.3f} effect={v['post']['effect']}")
    save("h18_forget", dict(T=T, test_seeds=TEST, dev_seeds=DEV, busy=BUSY, N=N_SWARM, **out))
