"""Phase-0 acceptance: the refactored package reproduces legacy/swarm.py (T=2000, 5 seeds) within ±0.01
per seed. In practice bit-exact: the RNG call order is preserved.

Kuramoto was removed from the package after H6, so only legacy variants with kuramoto=False are checked.
Golden numbers: tests/golden/legacy_T2000.json (regenerate with tests/make_golden.py)."""
import json, os

import pytest

from swarm import LEGACY, EnvConfig, run_swarm
from swarm.baselines import run_router, summary
from swarm.config import replace

HERE = os.path.dirname(os.path.abspath(__file__))
G = json.load(open(os.path.join(HERE, "golden", "legacy_T2000.json")))
T, SEEDS, TOL = G["T"], G["seeds"], 0.01
CLEAN = replace(LEGACY, sleep=False, fatigue=False)
# (golden group, golden name) -> (env kind, package config)
CASES = {
    ("homogeneous", "W_only"): ("homogeneous", CLEAN),
    ("homogeneous", "neither"): ("homogeneous", replace(CLEAN, plastic_W=False)),
    ("homogeneous", "solo"): ("homogeneous", replace(CLEAN, coupling=False, plastic_W=False)),
    ("roles", "W_only"): ("roles", CLEAN),
    ("roles", "neither"): ("roles", replace(CLEAN, plastic_W=False)),
    ("roles", "solo"): ("roles", replace(CLEAN, coupling=False, plastic_W=False)),
    ("roles", "W_reward"): ("roles", replace(CLEAN, hebb_bias=0.0)),
    ("roles", "W_reward_sim0"): ("roles", replace(CLEAN, hebb_bias=0.0, sim_w=0.0)),
    ("full", "full_nokur"): ("homogeneous", LEGACY),
    ("full", "no_hormones_nokur"): ("homogeneous", replace(LEGACY, hormones=False)),
    ("full", "full_nokur_reward"): ("homogeneous", replace(LEGACY, hebb_bias=0.0)),
    ("roles_full", "W_reward"): ("roles", replace(LEGACY, hebb_bias=0.0)),
    ("roles_full", "W_only"): ("roles", LEGACY),
}


@pytest.mark.parametrize("key", list(CASES), ids=[f"{g}-{n}" for g, n in CASES])
def test_swarm_matches_legacy(key):
    kind, cfg = CASES[key]
    env = EnvConfig(kind=kind, T=T)
    for sd, gold in zip(SEEDS, G[key[0]][key[1]]):
        s = run_swarm(env, cfg, sd).summary()
        assert abs(s["pre"] - gold["pre"]) <= TOL, (sd, s["pre"], gold["pre"])
        assert abs(s["post"] - gold["post"]) <= TOL, (sd, s["post"], gold["post"])


def test_minimal_architecture_is_default():
    # default = legacy clean config + per-pair mean W (H8c) + conditional recruitment with neutral prior (H9d)
    assert replace(CLEAN, hebb_bias=0.0, w_rule="mean_succ", recruit_rule="conditional", w_prior="mean") == type(LEGACY)()


def test_linucb_matches_legacy():
    env = EnvConfig(kind="homogeneous", T=T)
    for sd, gold in zip(SEEDS, G["linucb"]):
        s = summary(run_router(env, sd, model="linucb"), T)
        assert abs(s["pre"] - gold["pre"]) <= TOL
        assert abs(s["post"] - gold["post"]) <= TOL
