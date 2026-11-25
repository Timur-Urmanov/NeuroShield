"""H23: composition is rewritten by the colony's deficit, not by a primitive's own success (notes/hypotheses.md).
Run with the same arguments as H22; shares its checkpoint (controls A, R, I, P, IN, PN are reused)."""
import json, os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from common import ROOT, save
from h16_shift import DEV
import h20_cycle as h

NEW = {"C colony": [dict(team="evo_colony", frac=0.05, pm=0.1, beta=b, victim="contrib") for b in (1.0, 4.0)],
       "Cq quotas only": [dict(team="evo_colony_q", frac=0.05, pm=0.1, beta=1.0)],
       "Cv victims only": [dict(team="evo_colony_v", frac=0.05, pm=0.1, victim="contrib")]}
h.GRID.update(NEW)
CTRL = {"A fixed": 0, "R random search": 1, "I individual selection": 2, "P pattern selection": 2,
        "IN individual + novelty": 1, "PN pattern + novelty": 1}
TEST = list(range(15))

if __name__ == "__main__":
    keys = [(v, i, sd) for v in NEW for i in range(len(NEW[v])) for sd in [*DEV, *TEST]]
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
    chosen = dict(CTRL); dev = {}
    for v in NEW:
        dev[v] = [float(np.mean([res[(v, i, sd)]["post"] for sd in DEV])) for i in range(len(NEW[v]))]
        chosen[v] = int(np.argmax(dev[v]))
    col = lambda v, m="post": np.array([res[(v, chosen[v], sd)][m] for sd in TEST])
    out = dict(params={v: h.GRID[v][i] for v, i in chosen.items()}, dev_post=dev, table={}, comparisons={})
    for v in chosen:
        e = out["table"][v] = dict(post=float(col(v).mean()), sd=float(col(v).std(ddof=1)), per_seed=col(v).tolist(),
                                   **{m: float(col(v, m).mean()) for m in ("q", "niches_with_perfect", "perfect_share", "best_match_mean")})
        print(f"{v:<26} {str(h.GRID[v][chosen[v]]):<62} post={e['post']:.3f}±{e['sd']:.3f} q={e['q']:.3f} "
              f"niches={e['niches_with_perfect']:.1f} share={e['perfect_share']:.2f} best_m={e['best_match_mean']:.2f}")
    C = "C colony"
    for name, a, b in [("H23a C - A", C, "A fixed"), ("H23a C - R", C, "R random search"), ("H23b C - IN", C, "IN individual + novelty"),
                       ("H23b C - PN", C, "PN pattern + novelty"), ("H23c C - Cq", C, "Cq quotas only"), ("H23c C - Cv", C, "Cv victims only"),
                       ("C - I", C, "I individual selection"), ("Cq - I", "Cq quotas only", "I individual selection"),
                       ("Cv - I", "Cv victims only", "I individual selection"), ("Cq - A", "Cq quotas only", "A fixed"),
                       ("Cv - A", "Cv victims only", "A fixed")]:
        d = col(a) - col(b); se = d.std(ddof=1) / np.sqrt(len(d))
        o = out["comparisons"][name] = dict(diff=float(d.mean()), sd=float(d.std(ddof=1)), se=float(se), wins=int((d > 0).sum()), n=len(d),
                                            effect_se=bool(abs(d.mean()) > 2 * se), effect_2sd=bool(abs(d.mean()) > 2 * d.std(ddof=1)))
        print(f"  {name}: {o['diff']:+.3f} sd {o['sd']:.3f} se {o['se']:.3f} wins {o['wins']}/15 SE={o['effect_se']} 2sd={o['effect_2sd']}")
    save("h23_colony" + h.TAG, dict(T=h.T, N=h.N, busy=h.BUSY, test_seeds=TEST, dev_seeds=DEV, **out))
