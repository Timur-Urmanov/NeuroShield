"""Checkers accept correct and reject wrong answers; the code sandbox enforces its timeout."""
import time

import pytest

from swarm.llm.providers import SimOracle
from swarm.llm.tasks import Task, load_builtin
from swarm.llm.tasks.checkers import check, check_code, check_math, check_text, extract_last_number
from swarm.llm.tasks.sandbox import run_tests

POOL = load_builtin()


def test_pool_sizes():
    kinds = [t.kind for t in POOL]
    assert all(kinds.count(k) >= 20 for k in ("math", "code", "text"))
    assert len({t.prompt for t in POOL}) == len(POOL)


def test_math():
    assert extract_last_number("we get 1,250 apples, so $8.25.") == 8.25
    assert check_math("12 + 3 = 15\nFinal answer: 15", 15)
    assert check_math("Final answer: -4", -4.0)
    assert check_math("so 0.70", 0.7)
    assert not check_math("Final answer: 16", 15)
    assert not check_math("no number here", 15)


def test_text():
    t = Task(id="x", kind="text", prompt="p", answer="Bluefin Analytics")
    assert check_text("Reasoning...\nAnswer: bluefin analytics.", t)
    assert check_text("**Answer:** \"Bluefin Analytics\"", t)
    assert not check_text("Answer: Northwind Labs", t)
    e = Task(id="y", kind="text", prompt="p", answer="support-team@example.org")
    assert check_text("Answer: support-team@example.org", e)
    assert not check_text("Answer: support@example.org", e)


CODE = Task(id="c", kind="code", prompt="p", tests=["assert add(2, 3) == 5", "assert add(-1, 1) == 0"],
            entry_point="add")


def test_code_accept_reject():
    assert check_code("Here:\n```python\ndef add(a, b):\n    return a + b\n```", CODE)
    assert not check_code("```python\ndef add(a, b):\n    return a - b\n```", CODE)
    assert not check_code("```python\ndef add(a, b) return\n```", CODE)          # syntax error
    assert not check_code("```python\nimport sys\nsys.exit(0)\n```", CODE)      # exits before the tests


def test_code_timeout_enforced():
    t0 = time.time()
    r = run_tests("def add(a, b):\n    while True:\n        pass", CODE.tests, timeout=1.0)
    assert r.timed_out and not r.passed
    assert time.time() - t0 < 5


def test_code_memory_limit():
    r = run_tests("x = bytearray(4 * 1024**3)\ndef add(a, b):\n    return a + b", CODE.tests, timeout=5, mem_mb=256)
    assert not r.passed


@pytest.mark.parametrize("task", POOL, ids=[t.id for t in POOL])
def test_builtin_reference_passes_and_wrong_fails(task):
    assert check(task, SimOracle.render(task, True, 0.3)), task.id
    assert not check(task, SimOracle.render(task, False, 0.3)), task.id
