"""H22 confirmation: chosen settings of A, R, I, P, IN, PN on seeds 5-14 (notes/hypotheses.md).
Run with the same arguments as the H22 run; shares its checkpoint."""
import json, os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from common import ROOT, save
import h20_cycle as h

CHOSEN = {"A fixed": 0, "R random search": 1, "I individual selection": 2, "P pattern selection": 2,
          "IN individual + novelty": 1, "PN pattern + novelty": 1}
NEW, OLD = list(range(5, 15)), list(range(5))

if __name__ == "__main__":
    keys = [(v, i, sd) for v, i in CHOSEN.items() for sd in NEW]
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
    col = lambda v, S, m="post": np.array([res[(v, CHOSEN[v], sd)][m] for sd in S])
    out = dict(params={v: h.GRID[v][i] for v, i in CHOSEN.items()}, table={}, comparisons={})
    for v in CHOSEN:
        e = out["table"][v] = dict(post=float(col(v, NEW).mean()), sd=float(col(v, NEW).std(ddof=1)), per_seed=col(v, NEW).tolist(),
                                   **{m: float(col(v, NEW, m).mean()) for m in ("q", "niches_with_perfect", "perfect_share", "best_match_mean")})
        print(f"{v:<26} post={e['post']:.3f}±{e['sd']:.3f} q={e['q']:.3f} niches={e['niches_with_perfect']:.1f} "
              f"share={e['perfect_share']:.2f} best_m={e['best_match_mean']:.2f}")
    PN, IN, P, I, A, R = "PN pattern + novelty", "IN individual + novelty", "P pattern selection", "I individual selection", "A fixed", "R random search"
    for name, a, b in [("H22a PN - P", PN, P), ("H22b PN - A", PN, A), ("H22b PN - R", PN, R), ("H22c PN - IN", PN, IN),
                       ("IN - I", IN, I), ("IN - A", IN, A), ("IN - R", IN, R)]:
        o = {}
        for tag, S in (("seeds5-14", NEW), ("seeds0-14", OLD + NEW)):
            d = col(a, S) - col(b, S); se = d.std(ddof=1) / np.sqrt(len(d))
            o[tag] = dict(diff=float(d.mean()), sd=float(d.std(ddof=1)), se=float(se), wins=int((d > 0).sum()), n=len(d),
                          effect_se=bool(abs(d.mean()) > 2 * se), effect_2sd=bool(abs(d.mean()) > 2 * d.std(ddof=1)))
        out["comparisons"][name] = o; x = o["seeds5-14"]; y = o["seeds0-14"]
        print(f"  {name}: 5-14: {x['diff']:+.3f} sd {x['sd']:.3f} se {x['se']:.3f} wins {x['wins']}/10 SE={x['effect_se']} 2sd={x['effect_2sd']}"
              f" | 0-14: {y['diff']:+.3f} sd {y['sd']:.3f} se {y['se']:.3f} wins {y['wins']}/15 SE={y['effect_se']} 2sd={y['effect_2sd']}")
    save("h22_confirm" + h.TAG, dict(T=h.T, N=h.N, busy=h.BUSY, seeds=NEW, **out))
