"""Automatic checkers. check(task, output) -> bool."""
from __future__ import annotations

import hashlib
import re
import string

from .sandbox import run_tests

_NUM = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?|-?\.\d+")


def extract_last_number(text: str):
    nums = _NUM.findall(text or "")
    if not nums:
        return None
    try:
        return float(nums[-1].replace(",", ""))
    except ValueError:
        return None


def check_math(output: str, answer: float, rel_tol: float = 1e-6, abs_tol: float = 1e-6) -> bool:
    x = extract_last_number(output)
    return x is not None and abs(x - float(answer)) <= max(abs_tol, rel_tol * abs(float(answer)))


_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.S)


def extract_code(output: str) -> str:
    blocks = _FENCE.findall(output or "")
    py = [b for lang, b in blocks if lang.lower() in ("python", "py", "python3")]
    if py:
        return py[-1]
    if blocks:
        return blocks[-1][1]
    return output or ""


def check_code(output: str, task, timeout: float = 5.0, mem_mb: int = 512) -> bool:
    return run_tests(extract_code(output), task.tests, task.setup, timeout, mem_mb).passed


_STRIP = string.whitespace + string.punctuation + "“”‘’«»"   # stripped at the ends only


def normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(_STRIP)
    s = re.sub(r"^(the|a|an) ", "", s)
    return s


_ANS = re.compile(r"^\s*\**\s*answer\s*\**\s*[:\-]\s*(.+?)\s*$", re.I | re.M)


def extract_text_answer(output: str) -> str:
    m = _ANS.findall(output or "")
    if m:
        return m[-1]
    lines = [ln for ln in (output or "").splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def check_text(output: str, task) -> bool:
    got = normalize(extract_text_answer(output))
    return got in {normalize(a) for a in [task.answer, *task.aliases]}


_MEMO: dict = {}


def check(task, output: str, timeout: float = 5.0, mem_mb: int = 512) -> bool:
    """Memoised on (task id, output): identical outputs (cache hits, sim) are checked once."""
    key = (task.source, task.id, hashlib.sha256((output or "").encode()).hexdigest())
    if key in _MEMO:
        return _MEMO[key]
    if task.kind == "math":
        ok = check_math(output, task.answer)
    elif task.kind == "code":
        ok = check_code(output, task, timeout, mem_mb)
    elif task.kind == "text":
        ok = check_text(output, task)
    else:
        raise ValueError(task.kind)
    if len(_MEMO) > 100_000:
        _MEMO.clear()
    _MEMO[key] = ok
    return ok
