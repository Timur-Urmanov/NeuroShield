"""Generate golden regression numbers from the ORIGINAL legacy/swarm.py (T=2000, seeds 0..4).

Run once: python3 tests/make_golden.py. The refactored package must reproduce these within ±0.01
(in practice bit-exactly, because the RNG call order is preserved).
"""
import json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "legacy"))
import swarm as S  # noqa: E402  (legacy module)

T, SEEDS = 2000, 5
CLEAN = dict(sleep=False, fatigue=False)
HOM = {
    "W+kuramoto": dict(),
    "W_only": dict(kuramoto=False),
    "kuramoto_only": dict(plastic_W=False),
    "neither": dict(kuramoto=False, plastic_W=False),
    "solo": dict(coupling=False, kuramoto=False, plastic_W=False),
}
ROLES = {
    "W+kuramoto": dict(),
    "W_only": dict(kuramoto=False),
    "kuramoto_only": dict(plastic_W=False),
    "neither": dict(kuramoto=False, plastic_W=False),
    "solo": dict(coupling=False, kuramoto=False, plastic_W=False),
    "W_reward": dict(kuramoto=False, hebb_bias=0.0),
    "W_reward+kuramoto": dict(hebb_bias=0.0),
    "W_reward_sim0": dict(kuramoto=False, hebb_bias=0.0, sim_w=0.0),
}
FULL = {  # legacy defaults: sleep + fatigue + hormones + everything
    "full": dict(),
    "no_hormones": dict(hormones=False),
    # without Kuramoto (removed from the package after H6); these are what the package must reproduce
    "full_nokur": dict(kuramoto=False),
    "no_hormones_nokur": dict(kuramoto=False, hormones=False),
    "full_nokur_reward": dict(kuramoto=False, hebb_bias=0.0),
}
ROLES_FULL = {"W_reward": dict(kuramoto=False, hebb_bias=0.0), "W_only": dict(kuramoto=False)}


def row(s, R):
    cov = np.array(s.coverage, float) if s.coverage else np.zeros(len(R))
    return dict(pre=float(R[T // 2 - 1000:T // 2].mean()), post=float(R[-1000:].mean()),
                cover_end=float(cov[-1000:].mean()))


out = {"T": T, "seeds": list(range(SEEDS)), "homogeneous": {}, "roles": {}, "full": {}}
for name, cfg in HOM.items():
    out["homogeneous"][name] = [row(*S.run_swarm(T, sd, **CLEAN, **cfg)[:2]) for sd in range(SEEDS)]
for name, cfg in ROLES.items():
    out["roles"][name] = [row(*S.run_swarm(T, sd, roles=True, **CLEAN, **cfg)[:2]) for sd in range(SEEDS)]
for name, cfg in FULL.items():
    out["full"][name] = [row(*S.run_swarm(T, sd, **cfg)[:2]) for sd in range(SEEDS)]
out["roles_full"] = {}
for name, cfg in ROLES_FULL.items():
    out["roles_full"][name] = [row(*S.run_swarm(T, sd, roles=True, **cfg)[:2]) for sd in range(SEEDS)]
out["linucb"] = []
for sd in range(SEEDS):
    R = S.run_linucb(T, sd)
    out["linucb"].append(dict(pre=float(R[:1000].mean()), post=float(R[-1000:].mean())))
json.dump(out, open(os.path.join(HERE, "golden", "legacy_T2000.json"), "w"), indent=1)
print("ok")
