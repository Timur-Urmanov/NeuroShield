"""Correction run after the independent review of H14 (notes/hypotheses.md, "H14 — ревью"):
 (1) agent-UCB with its exploration constant tuned on dev seeds 100-104 (per cell, N=2560), then run on seeds 0-4;
 (2) the full swarm without the mutation flag leak (inherit_mut_known=False), seeds 0-4, N=2560."""
import json, os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from common import RESULTS, save
from h13_team_selection import CELLS, job as swarm_job
from swarm import EnvConfig, SwarmConfig
from swarm.baselines import run_router, summary as rsummary

N, T, CS = 2560, 8000, (0.01, 0.03, 0.1, 0.3)
ROLE_CELLS = [c for c in CELLS if c != "homogeneous"]


def bandit(spec):
    c, cval, sd = spec
    return rsummary(run_router(EnvConfig(T=T, **CELLS[c]), sd, N=N, model="linucb", team="agent_ucb", c=cval), T)["post"]


def swarm(spec):
    c, sd = spec
    cfg = SwarmConfig(N=N, inherit="team", inherit_frac=0.02, recruit_agg="min", sim_weight="evidence")
    return swarm_job((EnvConfig(T=T, **CELLS[c]), cfg, sd))


if __name__ == "__main__":
    dev, test = list(range(100, 105)), list(range(5))
    with ProcessPoolExecutor(max_workers=2) as ex:
        dspecs = [(c, cv, sd) for c in ROLE_CELLS for cv in CS for sd in dev]
        dres = dict(zip(dspecs, ex.map(bandit, dspecs)))
        best = {c: max(CS, key=lambda cv: np.mean([dres[(c, cv, sd)] for sd in dev])) for c in ROLE_CELLS}
        devtab = {c: {str(cv): float(np.mean([dres[(c, cv, sd)] for sd in dev])) for cv in CS} for c in ROLE_CELLS}
        print("dev means:", devtab, "-> chosen c:", best, flush=True)
        tspecs = [(c, best[c], sd) for c in ROLE_CELLS for sd in test]
        tres = dict(zip(tspecs, ex.map(bandit, tspecs)))
        sspecs = [(c, sd) for c in ROLE_CELLS for sd in test]
        sres = dict(zip(sspecs, ex.map(swarm, sspecs)))
    h14 = json.load(open(os.path.join(RESULTS, "h14_recruit.json")))["table"]
    h12 = json.load(open(os.path.join(RESULTS, "h12_inherit.json")))["table"]
    out = {}
    for c in ROLE_CELLS:
        b = np.array([tres[(c, best[c], sd)] for sd in test]); s = np.array([sres[(c, sd)]["post"] for sd in test])
        leak = np.array([r["post"] for r in h14[c][str(N)]["team|min|evidence"]["per_seed"]])
        orig = np.array([r["post"] for r in h12[c][str(N)]["none"]["per_seed"]])
        f = lambda x: dict(diff=float(x.mean()), sd=float(x.std(ddof=1)), effect=bool(abs(x.mean()) > 2 * x.std(ddof=1)))
        out[c] = dict(agent_ucb_c=best[c], agent_ucb=b.tolist(), swarm_full_honest=s.tolist(),
                      swarm_full_honest_pop={k: float(np.mean([sres[(c, sd)]["pop"][k] for sd in test])) for k in sres[(c, 0)]["pop"]},
                      cover_end=float(np.mean([sres[(c, sd)]["cover_end"] for sd in test])),
                      honest_minus_bandit=f(s - b), honest_minus_orig=f(s - orig), honest_minus_leaky=f(s - leak),
                      leaky_minus_bandit=f(leak - b))
        print(c, f"agent_ucb(c={best[c]}) {b.mean():.3f} | swarm honest {s.mean():.3f} (leaky {leak.mean():.3f}, orig {orig.mean():.3f})",
              "| honest-bandit", out[c]["honest_minus_bandit"], "| honest-orig", out[c]["honest_minus_orig"], flush=True)
    save("h14_review_rerun", dict(T=T, N=N, dev_seeds=dev, seeds=test, dev_means=devtab, cells=out))
