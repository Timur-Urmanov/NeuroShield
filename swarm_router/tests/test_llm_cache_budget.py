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


def test_budget_share_caps_variant_and_parent():
    run = Budget(1.0)
    a = run.share(0.4, "a")
    a.check(0.3); a.charge(0.3)
    with pytest.raises(BudgetExceeded) as e:
        a.check(0.2)
    assert e.value.scope == "share"
    assert run.spent == pytest.approx(0.3)
    b = run.share(10.0, "b")                     # capped at the parent's remaining 0.7
    assert b.limit == pytest.approx(0.7)
    with pytest.raises(BudgetExceeded) as e:
        b.check(0.8)
    assert e.value.scope == "run"


# ---------------------------------------------------------------- HTTP retries (no network: urlopen patched)
class _Resp:
    def __init__(self, body): self.body = body
    def read(self): return self.body
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _http_error(code):
    import io
    import urllib.error
    return urllib.error.HTTPError("https://x", code, "err", {}, io.BytesIO(b"overloaded"))


def test_post_json_retries_then_succeeds(monkeypatch):
    import urllib.error
    from swarm.llm import providers
    seq = [_http_error(529), urllib.error.URLError("reset"), _http_error(503), _Resp(b'{"ok": 1}')]
    calls, sleeps = [], []

    def fake(req, timeout):
        calls.append(1)
        x = seq.pop(0)
        if isinstance(x, Exception):
            raise x
        return x
    monkeypatch.setattr(providers.urllib.request, "urlopen", fake)
    monkeypatch.setattr(providers, "_sleep", sleeps.append)
    assert providers._post_json("https://x", {}, {}) == {"ok": 1}
    assert len(calls) == 4 and sleeps == [1.0, 2.0, 4.0]


def test_post_json_gives_up_and_no_retry_on_400(monkeypatch):
    from swarm.llm import providers
    calls = []

    def always(code):
        def f(req, timeout):
            calls.append(code)
            raise _http_error(code)
        return f
    monkeypatch.setattr(providers, "_sleep", lambda s: None)
    monkeypatch.setattr(providers.urllib.request, "urlopen", always(429))
    with pytest.raises(providers.ProviderError):
        providers._post_json("https://x", {}, {})
    assert len(calls) == providers.MAX_TRIES
    calls.clear()
    monkeypatch.setattr(providers.urllib.request, "urlopen", always(400))
    with pytest.raises(providers.ProviderError):
        providers._post_json("https://x", {}, {})
    assert len(calls) == 1


def test_real_embeddings_without_price_fail_loudly():
    from swarm.llm.embed import embedding_price, make_embedder
    from swarm.llm.estimate import Estimator
    assert embedding_price({"provider": "hashed"}) == 0.0
    bad = {"provider": "openai_compatible", "base_url": "https://e", "model": "m", "api_key_env": "K"}
    for cfg_ in (bad, dict(bad, price_in_per_1m=0)):
        with pytest.raises(ValueError):
            make_embedder(cfg_, dry_run=False)
    cfg = load_config()
    cfg.embeddings = dict(bad)
    with pytest.raises(ValueError):
        Estimator(cfg, load_builtin())
    cfg.embeddings = dict(bad, price_in_per_1m=0.02)
    assert Estimator(cfg, load_builtin()).embed_usd(1000) > 0


def test_verdict_three_way():
    from swarm.llm.roles import parse_verdict
    assert parse_verdict("ok\nVERDICT: ACCEPT") == "accept"
    assert parse_verdict("VERDICT: ACCEPT\n...\nverdict: revise because") == "revise"
    assert parse_verdict("looks fine to me") == "none"
