"""H17: cost per step vs N for the full swarm and context-aware pair bandits (dense and sparse).
One fresh process per run (peak RSS is per process). See notes/hypotheses.md. Checkpointed."""
import json, os, resource, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np

from common import ROOT, save
from h16_shift import BUSY, FULL, paired

T, SEEDS, NS = 8000, list(range(5)), (160, 640, 2560, 5120)
VARIANTS = {"swarm full": NS, "bandit sparse": NS, "bandit dense": (160, 640, 2560)}


def one(variant, N, seed):
    from swarm import EnvConfig, SwarmConfig, run_swarm
    from swarm.baselines import run_router
    env = EnvConfig("roles", T, role_by_type=True, busy=BUSY)
    base = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    t0 = time.process_time()
    if variant == "swarm full":
        tr = run_swarm(env, SwarmConfig(N=N, **FULL), seed, keep=True)
        R, state = tr.R, int((np.triu(tr.swarm.nW_pair, 1) > 0).sum())
    else:
        o = run_router(env, seed, N=N, model="linucb",
                       team="cluster_pair_mean_sparse" if variant == "bandit sparse" else "cluster_pair_mean")
        R = o["R"]
        state = o["policy"].state_size if variant == "bandit sparse" else int(sum((np.triu(n, 1) > 0).sum() for n in o["policy"].n))
    return dict(post=float(R[-1000:].mean()), ms_per_step=1000 * (time.process_time() - t0) / T,
                peak_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, base_mb=base, state_pairs=state)


def spawn(key):
    v, N, sd = key
    out = subprocess.check_output([sys.executable, __file__, "--one", v, str(N), str(sd)], cwd=os.path.dirname(__file__),
                                  env=dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1"))
    return json.loads(out.decode().strip().split("\n")[-1])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--one":
        print(json.dumps(one(sys.argv[2], int(sys.argv[3]), int(sys.argv[4])))); sys.exit()
    ck = os.path.join(ROOT, ".cache", "h17.jsonl"); res = {}
    if os.path.exists(ck):
        for line in open(ck):
            r = json.loads(line); res[tuple(r["key"])] = r["res"]
    todo = sorted([(v, N, sd) for v, Ns in VARIANTS.items() for N in Ns for sd in SEEDS if (v, N, sd) not in res], key=lambda k: -k[1])
    print(f"{len(res)} from checkpoint, {len(todo)} to do", flush=True)
    with ThreadPoolExecutor(max_workers=2) as ex, open(ck, "a") as f:      # 2 at a time on 4 cores: CPU time is not contended
        futs = {ex.submit(spawn, k): k for k in todo}
        for fu in as_completed(futs):
            k = futs[fu]; res[k] = fu.result()
            f.write(json.dumps(dict(key=list(k), res=res[k])) + "\n"); f.flush(); print("  done", k, res[k], flush=True)
    tab, cmp_ = {}, {}
    col = lambda v, N, m: [res[(v, N, sd)][m] for sd in SEEDS]
    for v, Ns in VARIANTS.items():
        for N in Ns:
            tab[f"{v}|{N}"] = dict(post=float(np.mean(col(v, N, "post"))), post_sd=float(np.std(col(v, N, "post"), ddof=1)),
                                   ms_per_step=float(np.mean(col(v, N, "ms_per_step"))), ms_sd=float(np.std(col(v, N, "ms_per_step"), ddof=1)),
                                   peak_mb=float(np.mean(col(v, N, "peak_mb"))), state_pairs=float(np.mean(col(v, N, "state_pairs"))))
            e = tab[f"{v}|{N}"]
            print(f"{v:<14} N={N:<5} post={e['post']:.3f}±{e['post_sd']:.3f}  {e['ms_per_step']:.3f}±{e['ms_sd']:.3f} ms/step  peak={e['peak_mb']:.0f} MB  pairs={e['state_pairs']:.0f}")
    for N in NS:
        c = dict(time_ratio_swarm_over_sparse=tab[f"swarm full|{N}"]["ms_per_step"] / tab[f"bandit sparse|{N}"]["ms_per_step"],
                 reward_swarm_minus_sparse=paired(col("swarm full", N, "post"), col("bandit sparse", N, "post")))
        if N in VARIANTS["bandit dense"]:
            c["time_ratio_dense_over_swarm"] = tab[f"bandit dense|{N}"]["ms_per_step"] / tab[f"swarm full|{N}"]["ms_per_step"]
            c["reward_sparse_minus_dense"] = paired(col("bandit sparse", N, "post"), col("bandit dense", N, "post"))
        cmp_[str(N)] = c; print(N, json.dumps(c))
    save("h17_cost", dict(T=T, seeds=SEEDS, busy=BUSY, table=tab, comparisons=cmp_))
