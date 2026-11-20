"""Phase-0 acceptance: refactored package reproduces the legacy swarm.py numbers (T=2000, 5 seeds)
within ±0.01 per seed. In practice it is bit-exact because the RNG call order is preserved."""
import json, os

import pytest

from swarm import LEGACY, EnvConfig, run_swarm
from swarm.baselines import run_router, summary
from swarm.config import replace

HERE = os.path.dirname(os.path.abspath(__file__))
G = json.load(open(os.path.join(HERE, "golden", "legacy_T2000.json")))
T, SEEDS, TOL = G["T"], G["seeds"], 0.01
CLEAN = replace(LEGACY, sleep=False, fatigue=False)
VARIANTS = {
    "W+kuramoto": dict(),
    "W_only": dict(kuramoto=False),
    "kuramoto_only": dict(plastic_W=False),
    "neither": dict(kuramoto=False, plastic_W=False),
    "solo": dict(coupling=False, kuramoto=False, plastic_W=False),
    "W_reward": dict(kuramoto=False, hebb_bias=0.0),
    "W_reward+kuramoto": dict(hebb_bias=0.0),
    "W_reward_sim0": dict(kuramoto=False, hebb_bias=0.0, sim_w=0.0),
}
CASES = ([("homogeneous", n, CLEAN) for n in G["homogeneous"]]
         + [("roles", n, CLEAN) for n in G["roles"]]
         + [("full", n, LEGACY) for n in G["full"]])


@pytest.mark.parametrize("group,name,base", CASES, ids=[f"{g}-{n}" for g, n, _ in CASES])
def test_swarm_matches_legacy(group, name, base):
    over = VARIANTS.get(name, {"no_hormones": dict(hormones=False)}.get(name, {}))
    cfg = replace(base, **over)
    env = EnvConfig(kind="roles" if group == "roles" else "homogeneous", T=T)
    for sd, gold in zip(SEEDS, G[group][name]):
        s = run_swarm(env, cfg, sd).summary()
        assert abs(s["pre"] - gold["pre"]) <= TOL, (sd, s["pre"], gold["pre"])
        assert abs(s["post"] - gold["post"]) <= TOL, (sd, s["post"], gold["post"])


def test_linucb_matches_legacy():
    env = EnvConfig(kind="homogeneous", T=T)
    for sd, gold in zip(SEEDS, G["linucb"]):
        s = summary(run_router(env, sd, model="linucb"), T)
        assert abs(s["pre"] - gold["pre"]) <= TOL
        assert abs(s["post"] - gold["post"]) <= TOL
