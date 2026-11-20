"""Automatic checkers. check(task, output) -> bool."""
from __future__ import annotations

import hashlib
import re
import string

from .sandbox import run_tests

_NUM_RE = re.compile(r"-?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?(?:\s*/\s*-?\d+(?:\.\d+)?)?")
_FINAL = re.compile(r"final\s+answer\s*\**\s*[:\-]?", re.I)
_MINUS = str.maketrans({"−": "-", "–": "-", "‒": "-", "—": "-", "﹣": "-", "－": "-"})


def _to_float(tok: str):
    tok = tok.replace(",", "").replace(" ", "")
    try:
        if "/" in tok:
            a, b = tok.split("/", 1)
            return float(a) / float(b) if float(b) != 0 else None
        return float(tok)
    except ValueError:
        return None


def _numbers(text: str):
    return [x for x in (_to_float(t) for t in _NUM_RE.findall(text)) if x is not None]


def extract_number(text: str):
    """First number after the last 'Final answer' marker (case-insensitive); else the last number."""
    text = (text or "").translate(_MINUS)
    marks = list(_FINAL.finditer(text))
    if marks:
        nums = _numbers(text[marks[-1].end():])
        if nums:
            return nums[0]
    nums = _numbers(text)
    return nums[-1] if nums else None


def extract_last_number(text: str):
    nums = _numbers((text or "").translate(_MINUS))
    return nums[-1] if nums else None


def check_math(output: str, answer: float, rel_tol: float = 1e-6, abs_tol: float = 1e-6) -> bool:
    x = extract_number(output)
    return x is not None and abs(x - float(answer)) <= max(abs_tol, rel_tol * abs(float(answer)))


_FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[^\n]*\n(.*?)```", re.S)


def _defs_only(code: str) -> bool:
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    ok = (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    return bool(tree.body) and all(isinstance(n, ok) for n in tree.body)


def extract_code(output: str, entry_point: str = "") -> str:
    """The fenced python block that defines `entry_point` (last such block, preceded by any other blocks
    that contain only imports/defs/classes, i.e. helpers; usage-example blocks are dropped), else all
    python blocks concatenated, else the last fenced block, else the raw text."""
    blocks = _FENCE.findall(output or "")
    py = [b for lang, b in blocks if lang.lower() in ("python", "py", "python3", "")]
    if entry_point:
        pat = re.compile(rf"^\s*(?:async\s+)?def\s+{re.escape(entry_point)}\s*\(", re.M)
        defs = [b for b in py if pat.search(b)]
        if defs:
            main = defs[-1]
            # keep other blocks that only define helpers/imports (no top-level calls such as usage examples)
            helpers = [b for b in py if b is not main and not pat.search(b) and _defs_only(b)]
            return "\n\n".join(helpers + [main])
    if py:
        return "\n\n".join(py)
    if blocks:
        return blocks[-1][1]
    return output or ""


def check_code(output: str, task, timeout: float = 5.0, mem_mb: int = 512) -> bool:
    return run_tests(extract_code(output, task.entry_point), task.tests, task.setup, timeout, mem_mb).passed


_STRIP = string.whitespace + string.punctuation + "“”‘’«»"   # stripped at the ends only


def normalize(s: str) -> str:
    s = (s or "").strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(_STRIP)
    s = re.sub(r"^(the|a|an) ", "", s)
    return s


# "Answer: x", "Final answer: x", "The answer: x", "**Final Answer:** x"
_ANS = re.compile(r"^[\s*_#>]*(?:[A-Za-z]+\s+)?answer\s*(?:is)?\s*\**\s*[:\-]\s*\**\s*(.+?)\s*$", re.I | re.M)


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
