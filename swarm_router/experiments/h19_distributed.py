"""H19: distributed setting - nodes, local information, paid remote calls (notes/hypotheses.md)."""
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import save
from h16_shift import BUSY, DEV, paired
from swarm import EnvConfig, SwarmConfig, run_swarm
from swarm.baselines import run_router

T, N, P, KAPPA, TEST, EPS = 8000, 640, 4, 0.05, list(range(5)), (0.0, 0.05, 0.1, 0.2, 0.4)
H14 = dict(recruit_agg="min", sim_weight="evidence", comm_aware=True)
SWARMS = {"swarm default": dict(comm_aware=True), "swarm (no inheritance)": H14, "swarm full (skill migration)": dict(inherit="team", **H14)}
ROUTERS = {"random local": dict(team="node_random"), "oracle": dict(team="node_oracle"), "shard bandits": dict(team="shard_sparse")}
TUNED = {f"{fam}|eps={e}": dict(team=f"{fam}_sparse", eps=e) for fam in ("node", "central") for e in EPS}


def job(spec):
    skew, v, seed = spec
    env_cfg = EnvConfig("roles", T, role_by_type=True, busy=BUSY, nodes=P, comm_cost=KAPPA, node_skew=skew)
    if v in SWARMS:
        tr = run_swarm(env_cfg, SwarmConfig(N=N, **SWARMS[v]), seed, keep=True); R, cov, env = tr.R, tr.coverage, tr.env
    else:
        o = run_router(env_cfg, seed, N=N, model="linucb", **{**ROUTERS, **TUNED}[v]); R, cov, env = o["R"], o["cover"], o["env"]
    w = slice(T - 1000, T)
    return dict(post=float(R[w].mean()), calls=float(env.remote_calls[w].mean()), cover=float(np.nanmean(cov[w])),
                consult=float(KAPPA * (env.home[w] != 0).mean()), early=float(R[:2000].mean()))


if __name__ == "__main__":
    keys = [(sk, v, sd) for sk in (0.0, 1.0) for v in [*SWARMS, *ROUTERS, *TUNED] for sd in ([*DEV, *TEST] if v in TUNED else TEST)]
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = dict(zip(keys, ex.map(job, keys, chunksize=1)))
    out = {}
    for sk in (0.0, 1.0):
        col = lambda v, m="post", S=TEST: np.array([res[(sk, v, sd)][m] for sd in S])
        dev = {v: float(col(v, "post", DEV).mean()) for v in TUNED}
        node = max([v for v in TUNED if v.startswith("node")], key=dev.get)
        # the centre is chosen on dev by reward after the consult cost (same constant for every eps)
        cen = max([v for v in TUNED if v.startswith("central")], key=dev.get)
        rows = {v: col(v) for v in [*SWARMS, *ROUTERS]}
        rows[f"node bandits [{node}]"] = col(node); rows[f"central, ideal [{cen}]"] = col(cen)
        rows[f"central, with consult cost [{cen}]"] = col(cen) - col(cen, "consult")
        src = {**{v: v for v in [*SWARMS, *ROUTERS]}, f"node bandits [{node}]": node, f"central, ideal [{cen}]": cen,
               f"central, with consult cost [{cen}]": cen}
        tab = {v: dict(post=float(r.mean()), sd=float(r.std(ddof=1)), calls=float(col(src[v], "calls").mean()),
                       cover=float(col(src[v], "cover").mean()), early=float(col(src[v], "early").mean()), per_seed=r.tolist())
               for v, r in rows.items()}
        dist = {v: r for v, r in rows.items() if not v.startswith(("central", "oracle", "swarm full"))}
        best_dist = max(dist, key=lambda v: dist[v].mean())
        sw = rows["swarm (no inheritance)"]
        cmp_ = {"H19a swarm (no inheritance) - shard bandits": paired(sw, rows["shard bandits"]),
                f"H19b swarm (no inheritance) - node bandits [{node}]": paired(sw, rows[f"node bandits [{node}]"]),
                f"H19c central with consult cost - best distributed [{best_dist}]": paired(rows[f"central, with consult cost [{cen}]"], rows[best_dist]),
                f"central ideal - best distributed [{best_dist}]": paired(rows[f"central, ideal [{cen}]"], rows[best_dist]),
                "H19d swarm full (skill migration) - swarm (no inheritance)": paired(rows["swarm full (skill migration)"], sw),
                f"swarm full (skill migration) - best distributed [{best_dist}]": paired(rows["swarm full (skill migration)"], rows[best_dist]),
                "swarm full (skill migration) - oracle": paired(rows["swarm full (skill migration)"], rows["oracle"])}
        out[f"skew {sk}"] = dict(table=tab, dev_post=dev, comparisons=cmp_)
        print(f"\n## skew {sk}   dev: " + " ".join(f"{k}={v:.3f}" for k, v in dev.items()))
        for v, e in tab.items():
            print(f"{v:<44} post={e['post']:.3f}±{e['sd']:.3f} calls={e['calls']:.2f} cover={e['cover']:.2f} first2000={e['early']:.3f}")
        for k, v in cmp_.items():
            print(f"  {k}: {v['diff']:+.3f}±{v['sd']:.3f} effect={v['effect']}")
    save("h19_distributed", dict(T=T, N=N, nodes=P, comm_cost=KAPPA, busy=BUSY, test_seeds=TEST, dev_seeds=DEV, results=out))
