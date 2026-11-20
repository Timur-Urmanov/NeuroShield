"""Offline tests: response cache and the hard budget guard (SimProvider, no network)."""
import pytest

from swarm.llm import Budget, BudgetExceeded, CallLog, LLMClient, ResponseCache, cache_key, load_config
from swarm.llm.providers import Completion, Provider, SimOracle
from swarm.llm.tasks import load_builtin, task_message


@pytest.fixture
def setup(tmp_path):
    cfg = load_config()
    pool = load_builtin()

    def make(limit=5.0):
        cache = ResponseCache(str(tmp_path / "c.sqlite"))
        log = CallLog(str(tmp_path / "log.jsonl"))
        return LLMClient(cfg, Budget(limit), cache, log, dry_run=True, oracle=SimOracle(pool))
    return cfg, pool, make


def test_cache_key_stable_and_sensitive():
    ident = dict(provider="anthropic", model="m", params={"a": 1, "b": 2})
    msgs = [{"role": "user", "content": "hi"}]
    k1 = cache_key(ident, "sys", msgs, dict(max_tokens=10, sample_id=0))
    k2 = cache_key(dict(params={"b": 2, "a": 1}, model="m", provider="anthropic"), "sys",
                   [{"content": "hi", "role": "user"}], dict(sample_id=0, max_tokens=10))
    assert k1 == k2 and len(k1) == 64
    assert k1 != cache_key(ident, "sys", msgs, dict(max_tokens=11, sample_id=0))
    assert k1 != cache_key(ident, "sys2", msgs, dict(max_tokens=10, sample_id=0))
    assert k1 != cache_key(dict(ident, model="m2"), "sys", msgs, dict(max_tokens=10, sample_id=0))
    assert k1 != cache_key(ident, "sys", msgs, dict(max_tokens=10, sample_id=1))


def test_cache_hit_costs_zero(setup, tmp_path):
    cfg, pool, make = setup
    c = make()
    msgs = [{"role": "user", "content": task_message(pool[0])}]
    a = c.call("opus", "S", msgs, 256)
    assert not a.cache_hit and a.usd_charged > 0
    spent = c.budget.spent
    b = c.call("opus", "S", msgs, 256)
    assert b.cache_hit and b.usd_charged == 0.0 and b.text == a.text
    assert b.usd_nominal == pytest.approx(a.usd_nominal)
    assert c.budget.spent == spent
    # persists on disk: a fresh client on the same file also hits
    c2 = make()
    assert c2.call("opus", "S", msgs, 256).cache_hit
    lines = open(tmp_path / "log.jsonl").read().splitlines()
    assert len(lines) == 3


class _Greedy(Provider):
    """Fake provider that always uses the full max_tokens (worst case for the guard)."""
    kind = "fake"

    def __init__(self, spec):
        super().__init__(spec)
        self.n = 0

    def complete(self, system, messages, max_tokens):
        self.n += 1
        return Completion("x", input_tokens=len(messages[0]["content"]) // 4 + 1, output_tokens=max_tokens)


def test_budget_stops_before_exceeding(setup):
    cfg, pool, make = setup
    limit = 0.05
    c = make(limit)
    fake = _Greedy(cfg.model("opus"))
    c.providers["opus"] = fake
    with pytest.raises(BudgetExceeded):
        for i in range(10_000):
            c.call("opus", None, [{"role": "user", "content": f"task {i} " + "y" * 400}], 512)
    assert c.budget.spent <= limit
    assert fake.n == c.budget.n_calls          # the refused call was never sent
    assert c.budget.spent + c.estimate_usd("opus", None, [{"role": "user", "content": "y" * 410}], 512) > limit


def test_budget_unit():
    b = Budget(1.0)
    b.check(0.6); b.charge(0.6)
    with pytest.raises(BudgetExceeded):
        b.check(0.5)
    b.check(0.4)
