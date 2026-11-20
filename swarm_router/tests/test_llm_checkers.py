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


# ---------------------------------------------------------------- review fixes: parsing edge cases
def test_math_final_answer_marker_first():
    assert check_math("Final answer: 7\n(Note: 35 sold out of 42.)", 7)
    assert check_math("The trip takes 90 minutes.\nFinal answer: 1.5 hours (90 minutes)", 1.5)
    assert not check_math("Final answer: 1.5 hours (90 minutes)", 90)
    assert check_math("FINAL ANSWER: 12", 12)


def test_math_number_formats():
    assert check_math("3/4", 0.75)
    assert check_math("Final answer: 3/4 cup", 0.75)
    assert check_math("−7", -7)                  # Unicode minus
    assert check_math("Final answer: −4 degrees", -4)
    assert check_math("1e3", 1000)
    assert check_math("Final answer: 2.5E-1", 0.25)
    assert not check_math("1e3", 1)


def test_text_answer_prefix_word():
    t = Task(id="v", kind="text", prompt="p", answer="Vienna")
    assert check_text("Final answer: Vienna", t)
    assert check_text("The answer: Vienna", t)
    assert check_text("**Final Answer:** Vienna", t)
    assert not check_text("Final answer: Berlin", t)


def test_code_picks_block_with_entry_point():
    out = ("Solution:\n```python\ndef add(a, b):\n    return a + b\n```\nUsage:\n```python\n"
           "print(add(2, 3))\nassert add(1, 1) == 3  # wrong example\n```")
    assert check_code(out, CODE)
    helper = "```python\ndef _h(a, b):\n    return a + b\n```\n```python\ndef add(a, b):\n    return _h(a, b)\n```"
    assert check_code(helper, CODE)                    # defs-only helper block is kept
    usage_first = "```python\nassert add(1, 1) == 3\n```\n```python\ndef add(a, b):\n    return a + b\n```"
    assert check_code(usage_first, CODE)               # usage block (top-level code) is dropped
    split = "```python\nimport math\n```\n```python\nX = 0\n```"
    from swarm.llm.tasks.checkers import extract_code
    assert extract_code(split, "add") == "import math\n\n\nX = 0\n"


def test_plausible_wrong_code_is_a_mutant():
    from swarm.llm.tasks.mutate import wrong_code
    for t in POOL:
        if t.kind == "code":
            w = wrong_code(t)
            assert "*args, **kwargs" not in w, t.id           # a real mutant, not the stub
            assert not check(t, f"```python\n{w}\n```"), t.id
