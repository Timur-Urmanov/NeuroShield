"""H24: which ingredient of the colony eviction rule matters (notes/hypotheses.md). No tuning: test seeds only.
Run with the same arguments as H22/H23; shares their checkpoint."""
import json, os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from common import ROOT, save
import h20_cycle as h
import h23_colony  # noqa: F401  (registers the H23 variants in h.GRID)

NEW = {f"C/{v}": [dict(team="evo_colony", frac=0.05, pm=0.1, beta=1.0, victim=v)] for v in ("rate", "nodrift", "noinherit")}
h.GRID.update(NEW)
REF = {"A fixed": 0, "I individual selection": 2, "PN pattern + novelty": 1, "C colony": 0, "Cq quotas only": 0}
TEST = list(range(15))

if __name__ == "__main__":
    keys = [(v, 0, sd) for v in NEW for sd in TEST]
    ck = os.path.join(ROOT, ".cache", f"h20{h.TAG}.jsonl"); res = {}
    for line in open(ck):
        r = json.loads(line); res[tuple(r["key"])] = r["res"]
    todo = [k for k in keys if k not in res]
    print(f"{len(keys) - len(todo)} from checkpoint, {len(todo)} to do", flush=True)
    with ProcessPoolExecutor(max_workers=4) as ex, open(ck, "a") as f:
        futs = {ex.submit(h.job, k): k for k in todo}
        for fu in as_completed(futs):
            k = futs[fu]; res[k] = fu.result()
            f.write(json.dumps(dict(key=list(k), res=res[k])) + "\n"); f.flush()
    idx = {**REF, **{v: 0 for v in NEW}}
    col = lambda v, m="post": np.array([res[(v, idx[v], sd)][m] for sd in TEST])
    out = dict(table={}, comparisons={})
    for v in idx:
        e = out["table"][v] = dict(post=float(col(v).mean()), sd=float(col(v).std(ddof=1)), mid=float(col(v, "mid").mean()), per_seed=col(v).tolist(),
                                   **{m: float(col(v, m).mean()) for m in ("q", "niches_with_perfect", "perfect_share", "best_match_mean")})
        print(f"{v:<26} post={e['post']:.3f}±{e['sd']:.3f} mid={e['mid']:.3f} q={e['q']:.3f} niches={e['niches_with_perfect']:.1f} "
              f"share={e['perfect_share']:.2f} best_m={e['best_match_mean']:.2f}")
    pairs = [(f"C - {v}", "C colony", v) for v in NEW] + [(f"{v} - {b.split()[0]}", v, b) for v in NEW for b in ("A fixed", "Cq quotas only", "PN pattern + novelty")]
    for name, a, b in pairs:
        d = col(a) - col(b); se = d.std(ddof=1) / np.sqrt(len(d))
        o = out["comparisons"][name] = dict(diff=float(d.mean()), sd=float(d.std(ddof=1)), se=float(se), wins=int((d > 0).sum()), n=len(d),
                                            effect_se=bool(abs(d.mean()) > 2 * se), effect_2sd=bool(abs(d.mean()) > 2 * d.std(ddof=1)))
        print(f"  {name}: {o['diff']:+.3f} sd {o['sd']:.3f} se {o['se']:.3f} wins {o['wins']}/15 SE={o['effect_se']} 2sd={o['effect_2sd']}")
    save("h24_victim" + h.TAG, dict(T=h.T, N=h.N, busy=h.BUSY, test_seeds=TEST, **out))
