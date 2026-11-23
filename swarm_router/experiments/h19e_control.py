"""H19e: control for H19d - wider eps grid; bandits on the population produced by the full swarm (skill migration)."""
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import save
from h16_shift import BUSY, DEV, paired
from h19_distributed import H14, KAPPA, N, P, T, TEST
from swarm import EnvConfig, SwarmConfig, run_swarm
from swarm.baselines import run_router

EPS = (0.4, 0.6, 0.8, 1.0)


def job(seed):
    env_cfg = EnvConfig("roles", T, role_by_type=True, busy=BUSY, nodes=P, comm_cost=KAPPA, node_skew=1.0)
    w = slice(T - 1000, T)
    st = lambda R, env, cov: dict(post=float(R[w].mean()), calls=float(env.remote_calls[w].mean()), cover=float(np.nanmean(cov[w])))
    tr = run_swarm(env_cfg, SwarmConfig(N=N, inherit="team", **H14), seed, keep=True)
    rk = tr.env.role_k.copy(); out = {"swarm full": st(tr.R, tr.env, tr.coverage)}
    V = {"shard (migrated)": dict(team="shard_sparse", role_k=rk), "oracle (migrated)": dict(team="node_oracle", role_k=rk),
         "random local (migrated)": dict(team="node_random", role_k=rk)}
    for e in EPS:
        V[f"node eps={e} (original)"] = dict(team="node_sparse", eps=e)
    for e in (0.0, 0.1, 0.4):
        V[f"node eps={e} (migrated)"] = dict(team="node_sparse", eps=e, role_k=rk)
    for v, kw in V.items():
        o = run_router(env_cfg, seed, N=N, model="linucb", **kw); out[v] = st(o["R"], o["env"], o["cover"])
    return out


if __name__ == "__main__":
    seeds = [*DEV, *TEST]
    with ProcessPoolExecutor(max_workers=4) as ex:
        res = dict(zip(seeds, ex.map(job, seeds, chunksize=1)))
    col = lambda v, m="post", S=TEST: np.array([res[sd][v][m] for sd in S])
    names = list(res[0]); dev = {v: float(col(v, "post", DEV).mean()) for v in names}
    orig = max([v for v in names if v.endswith("(original)")], key=dev.get)
    mig = max([v for v in names if v.startswith(("node", "shard")) and v.endswith("(migrated)")], key=dev.get)
    tab = {v: dict(post=float(col(v).mean()), sd=float(col(v).std(ddof=1)), calls=float(col(v, "calls").mean()), cover=float(col(v, "cover").mean()))
           for v in names}
    cmp_ = {f"swarm full - best node bandits on original population [{orig}]": paired(col("swarm full"), col(orig)),
            f"swarm full - best bandit on migrated population [{mig}]": paired(col("swarm full"), col(mig)),
            "swarm full - shard (migrated)": paired(col("swarm full"), col("shard (migrated)"))}
    print("dev:", {k: round(v, 3) for k, v in dev.items()})
    for v, e in tab.items():
        print(f"{v:<28} post={e['post']:.3f}±{e['sd']:.3f} calls={e['calls']:.2f} cover={e['cover']:.2f}")
    for k, v in cmp_.items():
        print(f"  {k}: {v['diff']:+.3f}±{v['sd']:.3f} effect={v['effect']}")
    save("h19e_control", dict(test_seeds=TEST, dev_seeds=DEV, table=tab, dev_post=dev, comparisons=cmp_))
