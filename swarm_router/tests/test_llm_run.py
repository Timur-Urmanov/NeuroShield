"""End-to-end dry runs (SimProvider, offline) of the LLM experiment scripts."""
import importlib.util
import json
import os

import numpy as np

from swarm import Swarm, SwarmConfig
from swarm.envs import LLMEnv
from swarm.llm import build_client, load_config
from swarm.llm.embed import HashedEmbedder
from swarm.llm.tasks import load_builtin, make_stream
from swarm.llm.tasks.checkers import check

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "experiments", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_llm_run_dry(tmp_path):
    mod = _load("llm_run")
    path = mod.main(["--dry-run", "--n-tasks", "15", "--seeds", "2", "--out", str(tmp_path),
                     "--cache-dir", str(tmp_path / "cache"), "--name", "t"])
    d = json.load(open(path))
    assert d["status"] == "ok" and d["mode"] == "dry_run" and d["seeds"] == [0, 1]
    assert {"git_commit", "argv", "config", "swarm_config", "experiment"} <= set(d)
    names = set(d["variants"])
    assert {"linucb", "swarm", "single:opus", "single:sonnet", "single:haiku"} <= names
    assert any(n.startswith("pipeline:") for n in names)
    for v in d["variants"].values():
        assert "reward" in v and "reward_sd" in v and "usd_per_task" in v
        assert all(r["n_done"] == 15 for r in v["per_seed"])
    assert 0 < d["budget"]["spent_usd"] <= d["budget"]["limit_usd"]
    assert os.path.exists(d["call_log"])


def test_llm_run_estimate(capsys):
    _load("llm_run").main(["--estimate", "--n-tasks", "300", "--seeds", "5"])
    out = capsys.readouterr().out
    assert "swarm" in out and "linucb" in out and "TOTAL" in out


def test_budget_stops_dry_run(tmp_path):
    path = _load("llm_run").main(["--n-tasks", "15", "--seeds", "1", "--budget-usd", "0.02", "--out", str(tmp_path),
                                  "--cache-dir", str(tmp_path / "cache"), "--name", "b"])
    d = json.load(open(path))
    assert d["status"] == "stopped_budget"
    assert d["budget"]["spent_usd"] <= 0.02


def test_llmenv_swarm_uses_real_outcome(tmp_path):
    cfg = load_config()
    pool = load_builtin()
    client = build_client(cfg, pool, True, 5.0, str(tmp_path), None)
    env = LLMEnv(cfg, client, make_stream(pool, 12, 0), HashedEmbedder(dim=16), seed=0)
    assert env.X.shape == (12, 16) and np.allclose(np.linalg.norm(env.X, axis=1), 1)
    assert np.all(env.U == 0.5) and len(env.cost) == env.M
    s = Swarm(SwarmConfig(), 0, env.M, env.D)
    for t in range(env.T):
        r, info = s.step(t, env)
        rec = env.records[-1]
        assert rec["t"] == t and info["succ"] == float(rec["success"])
        assert rec["success"] == check(env.tasks[t], env._last.answer)
        assert np.isclose(r, rec["success"] - env.lam * rec["usd"] / env.cost_norm)


def test_context_and_role_scripts_dry(tmp_path):
    p1 = _load("llm_context_degradation").main(["--n-tasks", "4", "--seeds", "2", "--k", "0,2", "--out",
                                                str(tmp_path), "--cache-dir", str(tmp_path / "c")])
    d1 = json.load(open(p1))
    assert d1["status"] == "ok" and "0" in d1["results"]["haiku"]["summary"]
    p2 = _load("llm_role_spread").main(["--n-tasks", "4", "--seeds", "2", "--out", str(tmp_path),
                                        "--cache-dir", str(tmp_path / "c")])
    d2 = json.load(open(p2))
    assert d2["status"] == "ok" and set(d2["summary"]) == {"none", "executor", "planner", "verifier"}
