"""Providers: Anthropic Messages API, OpenAI-compatible chat, and SimProvider (dry-run).

Real clients are imported lazily: the package works without `anthropic` installed, and nothing
touches the network unless a real provider's `complete()` is called. API keys are read from the
environment variable named in the config (`api_key_env`), never from the config itself.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

from .config import ModelSpec

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_VERSION = "2026-10-21"


class ProviderError(RuntimeError):
    pass


@dataclass
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    stop_reason: str = "end_turn"
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    extra: dict = field(default_factory=dict)


def completion_usd(spec: ModelSpec, c: Completion) -> float:
    # prompt-cache writes ~1.25x, reads ~0.1x the base input price (Anthropic); 0 for other providers
    eff_in = c.input_tokens + 1.25 * c.cache_write_tokens + 0.1 * c.cache_read_tokens
    return spec.usd(eff_in, c.output_tokens)


class Provider:
    kind = "base"

    def __init__(self, spec: ModelSpec):
        self.spec = spec

    def identity(self) -> dict:
        s = self.spec
        return dict(provider=self.kind, model=s.model, base_url=s.base_url, params=s.params)

    def complete(self, system: Optional[str], messages: list, max_tokens: int) -> Completion:
        raise NotImplementedError

    def _key(self) -> str:
        env = self.spec.api_key_env
        key = os.environ.get(env) if env else None
        if not key:
            raise ProviderError(f"model '{self.spec.name}': environment variable {env!r} is not set")
        return key


RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504, 529}
MAX_TRIES = 4
_sleep = time.sleep           # patched in tests


def _post_json(url: str, body: dict, headers: dict, timeout: float = 600.0, tries: int = MAX_TRIES) -> dict:
    """POST JSON; retries 408/409/429/5xx/529 and connection errors/timeouts with exponential backoff
    (1, 2, 4 s, or the server's retry-after if given), `tries` attempts in total."""
    data = json.dumps(body).encode()
    last = None
    for attempt in range(tries):
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers={"content-type": "application/json", **headers})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            msg = e.read().decode(errors="replace")[:500]
            last = ProviderError(f"HTTP {e.code} from {url}: {msg}")
            if e.code not in RETRY_STATUS and not 500 <= e.code < 600:
                raise last from e
            wait = 2.0 ** attempt
            try:
                wait = max(wait, float(e.headers.get("retry-after") or 0))
            except (TypeError, ValueError, AttributeError):
                pass
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = ProviderError(f"connection error to {url}: {e}")
            wait = 2.0 ** attempt
        if attempt < tries - 1:
            _sleep(min(wait, 60.0))
    raise last


class AnthropicProvider(Provider):
    """Messages API. Uses the `anthropic` SDK if importable, else raw HTTPS (urllib).
    Refusals (stop_reason == 'refusal') return empty text -> the checker scores a failure."""
    kind = "anthropic"

    def __init__(self, spec: ModelSpec):
        super().__init__(spec)
        self._client = None

    def _body(self, system, messages, max_tokens):
        body = dict(model=self.spec.model, max_tokens=int(max_tokens), messages=messages, **self.spec.params)
        if system:
            body["system"] = system
        return body

    def complete(self, system, messages, max_tokens):
        body = self._body(system, messages, max_tokens)
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return self._raw(body)
        if self._client is None:
            import anthropic
            # base_url is explicit: the SDK would otherwise honour ANTHROPIC_BASE_URL from the environment
            self._client = anthropic.Anthropic(api_key=self._key(), base_url="https://api.anthropic.com",
                                               max_retries=MAX_TRIES - 1)
        resp = self._client.messages.create(**body)
        text = "".join(b.text for b in resp.content if getattr(b, "type", "") == "text")
        u = resp.usage
        return Completion(text=text, input_tokens=int(u.input_tokens or 0), output_tokens=int(u.output_tokens or 0),
                          stop_reason=str(resp.stop_reason),
                          cache_write_tokens=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
                          cache_read_tokens=int(getattr(u, "cache_read_input_tokens", 0) or 0),
                          extra=dict(model=resp.model))

    def _raw(self, body):
        d = _post_json(ANTHROPIC_URL, body, {"x-api-key": self._key(), "anthropic-version": ANTHROPIC_VERSION})
        text = "".join(b.get("text", "") for b in d.get("content", []) if b.get("type") == "text")
        u = d.get("usage", {})
        return Completion(text=text, input_tokens=int(u.get("input_tokens") or 0),
                          output_tokens=int(u.get("output_tokens") or 0), stop_reason=str(d.get("stop_reason")),
                          cache_write_tokens=int(u.get("cache_creation_input_tokens") or 0),
                          cache_read_tokens=int(u.get("cache_read_input_tokens") or 0),
                          extra=dict(model=d.get("model")))


class OpenAICompatProvider(Provider):
    """Any /v1/chat/completions endpoint (raw HTTPS, no SDK dependency)."""
    kind = "openai_compatible"

    def complete(self, system, messages, max_tokens):
        if not self.spec.base_url:
            raise ProviderError(f"model '{self.spec.name}': base_url is required")
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        body = {"model": self.spec.model, "messages": msgs, self.spec.max_tokens_field: int(max_tokens),
                **self.spec.params}
        d = _post_json(self.spec.base_url.rstrip("/") + "/chat/completions", body,
                       {"authorization": f"Bearer {self._key()}"})
        ch = (d.get("choices") or [{}])[0]
        u = d.get("usage") or {}
        return Completion(text=(ch.get("message") or {}).get("content") or "",
                          input_tokens=int(u.get("prompt_tokens") or 0),
                          output_tokens=int(u.get("completion_tokens") or 0),
                          stop_reason=str(ch.get("finish_reason")))


# ----------------------------------------------------------------------------------- simulation
def hash_uniform(*parts) -> float:
    h = hashlib.sha256(json.dumps(parts, sort_keys=True, ensure_ascii=False).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2**64


_TASK_RE = re.compile(r"<task>\n(.*?)\n</task>", re.S)
_ANSWER_RE = re.compile(r"<answer>\n(.*)\n</answer>", re.S)
_PLANQ_RE = re.compile(r"sim-plan-bonus=([0-9.]+)")


class SimOracle:
    """Maps a task prompt (between <task> tags) back to the Task, so the simulated model can emit a
    correct or a wrong answer in the real output format. Only SimProvider uses it."""

    def __init__(self, tasks, check_fn=None):
        self.by_prompt = {t.prompt: t for t in tasks}
        from .tasks.checkers import check
        self.check = check_fn or check

    def find(self, text: str):
        m = _TASK_RE.findall(text or "")
        return self.by_prompt.get(m[-1]) if m else None

    @staticmethod
    def render(task, correct: bool, u: float) -> str:
        """Reply in the real output format; wrong replies are plausible (tasks/mutate.py)."""
        from .tasks.mutate import wrong_code, wrong_number, wrong_text
        if task.kind == "math":
            a = float(task.answer) if correct else wrong_number(task.answer, u)
            s = f"{a:.4f}".rstrip("0").rstrip(".")
            return f"Working through the quantities step by step.\nFinal answer: {s}"
        if task.kind == "code":
            code = task.reference if correct else wrong_code(task)
            return f"```python\n{code}\n```"
        return f"Considering the text.\nAnswer: {task.answer if correct else wrong_text(task, u)}"


class SimProvider(Provider):
    """Deterministic simulated model with per-task-kind accuracy and token costs (config `sim:`).

    The correctness draw is a hash of (model, system, messages, sample_id), so it is reproducible and
    identical prompts give identical replies, exactly like a cached real model. Modifiers:
      role_skill[role]   additive accuracy change for the system prompt used as solver
      plan bonus         planner's `plan_bonus` if a plan is present in the executor message
      context decay      - context_decay_per_1k * input_tokens / 1000
    Verifier: P(reject | wrong) = verify_detect, P(reject | correct) = false_reject (uses the checker).
    Token counts: input = chars/4, output = out_tokens[role] * U(0.7, 1.3).
    """
    kind = "sim"

    def __init__(self, spec: ModelSpec, oracle: SimOracle):
        super().__init__(spec)
        self.oracle = oracle
        self.sp = spec.sim

    def identity(self) -> dict:
        return dict(provider="sim", model=self.spec.model, sim=self.sp)

    def complete(self, system, messages, max_tokens, sample_id: int = 0):
        from .roles import PROMPT_TO_ROLE
        sp = self.sp
        role = PROMPT_TO_ROLE.get(system or "", "none")
        last = messages[-1]["content"]
        n_in = int(sum(len(m["content"]) for m in messages) + len(system or "")) // 4 + 1
        u = hash_uniform(self.spec.model, system, messages, sample_id)
        u_tok = hash_uniform("tok", self.spec.model, system, messages, sample_id)
        task = self.oracle.find(last) or self.oracle.find(messages[0]["content"])

        def out(text, r):
            n_out = max(1, int(sp["out_tokens"].get(r, 100) * (0.7 + 0.6 * u_tok)))
            n_out = min(n_out, int(max_tokens))
            return Completion(text=text, input_tokens=n_in, output_tokens=n_out,
                              stop_reason="max_tokens" if n_out == int(max_tokens) else "end_turn")

        if task is None:
            return out("I could not identify the task.", "executor")
        verify_mode = role == "verifier" or bool(_ANSWER_RE.search(last))
        if role == "planner" and not verify_mode:
            return out(f"1. Restate what is asked.\n2. Work it out step by step.\n3. Check the result.\n"
                       f"(sim-plan-bonus={sp['plan_bonus']:.3f})", "planner")
        if verify_mode:
            m = _ANSWER_RE.findall(last)
            ok = bool(m) and self.oracle.check(task, m[-1])
            if hash_uniform("noverdict", u) < sp.get("no_verdict", 0.0):
                return out("The answer seems mostly reasonable.", "verifier")
            reject = u < (sp["false_reject"] if ok else sp["verify_detect"])
            return out("VERDICT: REVISE\nThe answer does not hold up; recheck the computation."
                       if reject else "VERDICT: ACCEPT", "verifier")
        # solve (executor, or another prompt used as a solver)
        p = sp["acc"].get(task.kind, 0.5) + sp["role_skill"].get(role, 0.0)
        pb = _PLANQ_RE.findall(" ".join(m["content"] for m in messages if m["role"] == "user"))
        if pb:
            p += float(pb[-1])
        p -= sp["context_decay_per_1k"] * n_in / 1000.0
        return out(SimOracle.render(task, u < p, hash_uniform("w", u)), "executor")


class LlamaCppProvider(Provider):
    """Local GGUF model on CPU via llama-cpp-python (no network, no money).

    `model` = path to the .gguf file (relative to the project root). Prices in the config are a
    COMPUTE PROXY (pseudo-USD proportional to parameter count), not money; they keep the budget guard
    and the cost term of the reward meaningful. Sampling: params.temperature (default 0.7) with a
    seed derived from (model, prompt, sample_id), so each sample_id is reproducible.
    """
    kind = "llama_cpp"
    _loaded: dict = {}

    def _llm(self):
        from .config import ROOT
        path = self.spec.model if os.path.isabs(self.spec.model) else os.path.join(ROOT, self.spec.model)
        if path not in LlamaCppProvider._loaded:
            from llama_cpp import Llama
            p = self.spec.params
            LlamaCppProvider._loaded[path] = Llama(model_path=path, n_ctx=int(p.get("n_ctx", 16384)),
                                                   n_threads=int(p.get("n_threads", os.cpu_count() or 4)),
                                                   verbose=False)
        return LlamaCppProvider._loaded[path]

    def identity(self) -> dict:
        p = {k: v for k, v in self.spec.params.items() if k != "n_threads"}   # threads do not change output
        return dict(provider=self.kind, model=self.spec.model, params=p)

    def complete(self, system, messages, max_tokens, sample_id: int = 0):
        msgs = ([{"role": "system", "content": system}] if system else []) + list(messages)
        seed = int(hash_uniform("seed", self.spec.model, system, messages, sample_id) * 2**31)
        r = self._llm().create_chat_completion(messages=msgs, max_tokens=int(max_tokens),
                                               temperature=float(self.spec.params.get("temperature", 0.7)),
                                               seed=seed)
        ch = r["choices"][0]; u = r.get("usage", {})
        return Completion(text=ch["message"].get("content") or "", input_tokens=int(u.get("prompt_tokens", 0)),
                          output_tokens=int(u.get("completion_tokens", 0)),
                          stop_reason="max_tokens" if ch.get("finish_reason") == "length" else "end_turn")


def make_provider(spec: ModelSpec, dry_run: bool, oracle: Optional[SimOracle] = None) -> Provider:
    if dry_run or spec.provider == "sim":
        if oracle is None:
            raise ValueError("SimProvider needs a SimOracle")
        return SimProvider(spec, oracle)
    if spec.provider == "anthropic":
        return AnthropicProvider(spec)
    if spec.provider == "openai_compatible":
        return OpenAICompatProvider(spec)
    if spec.provider == "llama_cpp":
        return LlamaCppProvider(spec)
    raise ValueError(f"unknown provider {spec.provider!r}")
