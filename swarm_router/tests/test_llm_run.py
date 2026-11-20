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


# ---------------------------------------------------------------- review fixes
def test_real_requires_explicit_budget(tmp_path, monkeypatch, capsys):
    from swarm.llm.estimate import SPEND_ENV, confirm_real, spend_authorised
    monkeypatch.delenv(SPEND_ENV, raising=False)
    assert not spend_authorised(5.0, False)[0]
    assert spend_authorised(5.0, True)[0]
    monkeypatch.setenv(SPEND_ENV, "4.99")
    assert not spend_authorised(5.0, False)[0]
    monkeypatch.setenv(SPEND_ENV, "5")
    assert spend_authorised(5.0, False)[0]
    monkeypatch.delenv(SPEND_ENV)
    assert not confirm_real(lambda: None, yes=True, budget_usd=5.0, budget_explicit=False, isatty=False)
    assert not confirm_real(lambda: None, yes=False, budget_usd=5.0, budget_explicit=True, isatty=False)
    assert confirm_real(lambda: None, yes=True, budget_usd=5.0, budget_explicit=True, isatty=False)
    # --real --yes alone: aborts before building any client (no results file, no call log)
    out = _load("llm_run").main(["--real", "--yes", "--n-tasks", "3", "--seeds", "1", "--out", str(tmp_path),
                                 "--cache-dir", str(tmp_path / "c")])
    assert out is None and not list(tmp_path.iterdir())
    assert "aborting without any API call" in capsys.readouterr().out


def test_estimates_have_both_columns(capsys):
    _load("llm_run").main(["--estimate", "--n-tasks", "30", "--seeds", "2"])
    _load("llm_context_degradation").main(["--estimate"])
    _load("llm_role_spread").main(["--estimate"])
    out = capsys.readouterr().out
    assert out.count("USD opt") == 3 and out.count("USD cons") == 3 and "embeddings" in out


def test_error_keeps_partial_results(tmp_path, monkeypatch):
    import swarm.envs.llm as envmod
    real = envmod.run_team
    n = {"i": 0}

    def flaky(*a, **k):
        n["i"] += 1
        if n["i"] == 7:
            raise RuntimeError("simulated provider failure")
        return real(*a, **k)
    monkeypatch.setattr(envmod, "run_team", flaky)
    path = _load("llm_run").main(["--n-tasks", "10", "--seeds", "2", "--out", str(tmp_path),
                                  "--cache-dir", str(tmp_path / "c"), "--name", "e"])
    d = json.load(open(path))
    sw = d["variants"]["swarm"]
    assert sw["per_seed"][0]["status"].startswith("error: RuntimeError")
    assert sw["per_seed"][0]["n_done"] == 6 and sw.get("partial")
    assert d["status"] == "partial" and "linucb" in d["variants"]        # later variants still ran


def test_variant_share_stops_only_that_variant(tmp_path):
    path = _load("llm_run").main(["--n-tasks", "15", "--seeds", "1", "--budget-usd", "0.06", "--out", str(tmp_path),
                                  "--cache-dir", str(tmp_path / "c"), "--name", "s"])
    d = json.load(open(path))
    sts = {k: v["per_seed"][0]["status"] for k, v in d["variants"].items()}
    assert any(s.startswith("stopped_budget[share]") for s in sts.values())
    assert list(d["variants"])[0] == "swarm"
    assert d["budget"]["spent_usd"] <= 0.06
    for k, v in d["variants"].items():
        assert v["per_seed"][0]["charged_usd"] <= v["share_usd"] + 1e-12


def test_pilot_dry(tmp_path, capsys):
    path = _load("llm_run").main(["--pilot", "6", "--pilot-only", "--n-tasks", "30", "--seeds", "2", "--out",
                                  str(tmp_path), "--cache-dir", str(tmp_path / "c"), "--name", "p"])
    d = json.load(open(path))
    assert d["status"] == "pilot_only"
    assert set(d["pilot"]) == {"opus", "sonnet", "haiku"}
    assert set(d["pilot"]["opus"]) == {"executor", "planner", "verifier"}
    assert "after pilot" in capsys.readouterr().out
