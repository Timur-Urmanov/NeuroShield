"""H20: variation -> selection -> retention over patterns in an env with rewritable configurations
(notes/hypotheses.md). Hyperparameters of every variant are chosen on dev seeds."""
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import save
from h16_shift import DEV, paired
from swarm import EnvConfig, SwarmConfig, run_swarm
from swarm.baselines import run_router
from swarm.envs import make_env

T, N, BUSY, TEST = 20000, 640, 48, list(range(5))
FRAC, PM = (0.02, 0.05), (0.1, 0.3)
ENV_KW = dict(a.split("=") for a in sys.argv[1:] if "=" in a)          # e.g. genes_L=4 for follow-ups
TAG = "".join(f"_{k}{v}" for k, v in ENV_KW.items())
GRID = {"A fixed": [dict(team="evo_fixed")],
        "R random search": [dict(team="evo_random", frac=f) for f in FRAC],
        "I individual selection": [dict(team="evo_individual", frac=f, pm=p) for f in FRAC for p in PM],
        "P pattern selection": [dict(team="evo_pattern", frac=f, pm=p) for f in FRAC for p in PM],
        "P0 pattern copy, no mutation": [dict(team="evo_pattern", frac=f, pm=0.0) for f in FRAC],
        "S full swarm": [dict(swarm=True, frac=f, pm=p) for f in FRAC for p in PM]}


def job(spec):
    v, i, seed = spec
    kw = dict(GRID[v][i]); env_cfg = EnvConfig("config", T, busy=BUSY, **{k: int(x) for k, x in ENV_KW.items()})
    if kw.pop("swarm", False):
        tr = run_swarm(env_cfg, SwarmConfig(N=N, inherit="team", recruit_agg="min", sim_weight="evidence", inherit_frac=kw["frac"],
                                            inherit_mut=kw["pm"], inherit_mut_known=True), seed, keep=True)
        R, q, env = tr.R, tr.q, tr.env
    else:
        o = run_router(env_cfg, seed, N=N, model="linucb", **kw); R, q, env = o["R"], o["q"], o["env"]
    m = (env.genes[None, :, None, :] == env.targets[:, None, :, :]).sum(-1)
    return dict(post=float(R[-2000:].mean()), mid=float(R[T // 2 - 2000:T // 2].mean()), q=float(q[-2000:].mean()),
                niches_with_perfect=int(((m == env.L).sum(1) > 0).sum()), perfect_share=float((m == env.L).any((0, 2)).mean()),
                best_match_mean=float(m.max(1).mean()), oracle=float(env.oracle_value("post", env.q_full)))


if __name__ == "__main__":
    keys = [(v, i, sd) for v in GRID for i in range(len(GRID[v])) for sd in [*DEV, *TEST]]
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = dict(zip(keys, ex.map(job, keys, chunksize=1)))
    col = lambda v, i, m="post", S=TEST: np.array([res[(v, i, sd)][m] for sd in S])
    chosen, tab = {}, {}
    for v in GRID:
        dev = [float(col(v, i, "post", DEV).mean()) for i in range(len(GRID[v]))]
        i = int(np.argmax(dev)); chosen[v] = i
        tab[v] = dict(params=GRID[v][i], dev_post=dev, post=float(col(v, i).mean()), sd=float(col(v, i).std(ddof=1)),
                      **{m: float(col(v, i, m).mean()) for m in ("mid", "q", "niches_with_perfect", "perfect_share", "best_match_mean")},
                      per_seed=col(v, i).tolist())
        e = tab[v]
        print(f"{v:<30} {str(GRID[v][i]):<52} post={e['post']:.3f}±{e['sd']:.3f} mid={e['mid']:.3f} q={e['q']:.3f} "
              f"niches={e['niches_with_perfect']:.1f}/9 perfect_share={e['perfect_share']:.2f} best_m={e['best_match_mean']:.2f}")
    c = lambda v: col(v, chosen[v])
    P = "P pattern selection"
    cmp_ = {"H20a P - A": paired(c(P), c("A fixed")), "H20b P - I": paired(c(P), c("I individual selection")),
            "H20c P - R": paired(c(P), c("R random search")), "H20d P - P0": paired(c(P), c("P0 pattern copy, no mutation")),
            "H20e S - P": paired(c("S full swarm"), c(P)), "I - A": paired(c("I individual selection"), c("A fixed")),
            "I - R": paired(c("I individual selection"), c("R random search")), "S - A": paired(c("S full swarm"), c("A fixed"))}
    for k, v in cmp_.items():
        print(f"  {k}: {v['diff']:+.3f}±{v['sd']:.3f} effect={v['effect']}")
    orc = float(np.mean([res[("A fixed", 0, sd)]["oracle"] for sd in TEST])); print("oracle (perfect population):", round(orc, 3))
    save("h20_cycle" + TAG, dict(T=T, N=N, busy=BUSY, env_kw=ENV_KW, test_seeds=TEST, dev_seeds=DEV, table=tab, comparisons=cmp_, oracle=orc))
