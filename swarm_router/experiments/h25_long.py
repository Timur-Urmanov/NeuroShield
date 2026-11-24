"""H25: long horizon for C (H23), A and PN (notes/hypotheses.md). One 1.6M-task run per variant and seed;
saves 10k-block rewards and population snapshots every 20k tasks to .cache/h25/."""
import os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from common import ROOT
from swarm import EnvConfig
from swarm.baselines import run_router

T, SHIFT, N, BUSY, SNAP, BLK = 1_600_000, 100_000, 640, 48, 20_000, 10_000
SEEDS = list(range(20, 30))
VAR = {"C": dict(team="evo_colony", frac=0.05, pm=0.1, beta=1.0), "A": dict(team="evo_fixed"),
       "PN": dict(team="evo_pattern", frac=0.05, pm=0.1, nov=2.0)}
OUT = os.path.join(ROOT, ".cache", "h25")


def job(v, seed):
    t0 = time.time()
    o = run_router(EnvConfig("config", T, busy=BUSY, genes_L=5, genes_pow=1.0, shift_at=SHIFT), seed, N=N, model="linucb",
                   trace_every=SNAP, **VAR[v])
    tp, env = o["policy"], o["env"]
    np.savez_compressed(os.path.join(OUT, f"{v}_{seed}.tmp.npz"), R=o["R"].reshape(-1, BLK).mean(1), q=o["q"].reshape(-1, BLK).mean(1),
                        t=np.array([s[0] for s in tp.trace]), genes=np.array([s[1] for s in tp.trace]),
                        lin=np.array([s[2] for s in tp.trace]), epoch=np.array([s[3] for s in tp.trace]),
                        targets=env.targets, genes0=np.random.default_rng(seed + 2900).integers(env.A, size=(N, env.L)),
                        wall=time.time() - t0, state=tp.state_size, births=tp.births)
    os.replace(os.path.join(OUT, f"{v}_{seed}.tmp.npz"), os.path.join(OUT, f"{v}_{seed}.npz"))
    return v, seed, time.time() - t0


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    todo = [(v, sd) for sd in SEEDS for v in VAR if not os.path.exists(os.path.join(OUT, f"{v}_{sd}.npz"))]
    print(f"{30 - len(todo)} done, {len(todo)} to do", flush=True)
    with ProcessPoolExecutor(max_workers=4) as ex:
        for fu in as_completed([ex.submit(job, *k) for k in todo]):
            v, sd, w = fu.result(); print(f"{v} seed {sd}: {w / 60:.1f} min", flush=True)
