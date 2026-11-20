"""Plausible wrong answers with known labels (verifier probes in llm_role_spread, SimProvider errors).

code: a deterministic AST mutant of the reference (off-by-one constant, flipped comparison,
      swapped branch / boolean operator) that is verified to FAIL the task's tests;
math: off-by-one / slip of the true value;
text: another label, or another entity from the task text for extraction items.
"""
from __future__ import annotations

import ast
import copy
import re

_CMP_SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt, ast.Eq: ast.NotEq,
             ast.NotEq: ast.Eq, ast.In: ast.NotIn, ast.NotIn: ast.In, ast.Is: ast.IsNot, ast.IsNot: ast.Is}
_BIN_SWAP = {ast.Add: ast.Sub, ast.Sub: ast.Add, ast.Mult: ast.Add, ast.FloorDiv: ast.Mod, ast.Mod: ast.FloorDiv}


_UNWRAP = {"abs", "sorted", "set", "list", "reversed", "tuple", "int", "str"}
_ORDER = {"cmp": 0, "branch": 1, "star": 2, "unwrap": 3, "const": 4, "bin": 5, "bool": 6}


def _kind(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return "const"
    if isinstance(node, ast.Compare):
        return "cmp"
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_SWAP:
        return "bin"
    if isinstance(node, ast.If) and node.orelse:
        return "branch"
    if isinstance(node, ast.BoolOp):
        return "bool"
    if isinstance(node, ast.Starred):
        return "star"
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _UNWRAP
            and len(node.args) == 1 and not node.keywords):
        return "unwrap"
    return None


class _Mutator(ast.NodeTransformer):
    """Counts mutation sites in visiting order; replaces the `target`-th one (target=None: count only)."""

    def __init__(self, target=None):
        self.target, self.i, self.kinds = target, 0, []

    def visit(self, node):
        k = _kind(node)
        if k is not None:
            i = self.i; self.i += 1; self.kinds.append(k)
            if i == self.target:
                return self._apply(k, node)
        return self.generic_visit(node)

    @staticmethod
    def _apply(kind, node):
        if kind == "const":
            node.value = node.value + 1
        elif kind == "cmp":
            node.ops = [(_CMP_SWAP.get(type(o), type(o)))() for o in node.ops]
        elif kind == "bin":
            node.op = _BIN_SWAP[type(node.op)]()
        elif kind == "branch":
            node.body, node.orelse = node.orelse, node.body
        elif kind == "bool":
            node.op = ast.Or() if isinstance(node.op, ast.And) else ast.And()
        elif kind == "star":
            return node.value
        elif kind == "unwrap":
            return node.args[0]
        return node


_MEMO: dict = {}


def code_mutants(reference: str):
    """All single-site mutants of the reference, deterministic order (comparisons/branches first)."""
    try:
        tree = ast.parse(reference)
    except SyntaxError:
        return []
    counter = _Mutator(); counter.visit(copy.deepcopy(tree))
    idx = sorted(range(counter.i), key=lambda i: (_ORDER[counter.kinds[i]], i))
    out = []
    for i in idx:
        t2 = _Mutator(i).visit(copy.deepcopy(tree))
        try:
            out.append(ast.unparse(ast.fix_missing_locations(t2)))
        except Exception:
            continue
    return out


def wrong_code(task, check_fn=None) -> str:
    """First mutant that fails the tests (memoised per task); stub only if no mutant fails."""
    key = (task.source, task.id)
    if key not in _MEMO:
        from .checkers import check_code
        chk = check_fn or (lambda t, out: check_code(out, t, timeout=3.0))
        _MEMO[key] = None
        for m in code_mutants(task.reference):
            if not chk(task, f"```python\n{m}\n```"):
                _MEMO[key] = m
                break
        if _MEMO[key] is None:
            _MEMO[key] = f"def {task.entry_point or 'solution'}(*args, **kwargs):\n    return None"
    return _MEMO[key]


def wrong_number(answer: float, u: float) -> float:
    a = float(answer)
    if a == int(a):
        return a + (1 if u < 0.5 else -1) * (1 if u < 0.75 else 2) if abs(a) > 0 else a + 1
    return round(a * (1.1 if u < 0.5 else 0.9), 2)


def wrong_text(task, u: float) -> str:
    ans = str(task.answer)
    labels = [l for l in (task.labels or []) if str(l).lower() != ans.lower()]
    if labels:
        return labels[int(u * len(labels)) % len(labels)]
    if ans.lower() in _LANGS:
        return next(l for l in sorted(_LANGS, key=lambda l: hash_rank(l, u)) if l != ans.lower()).capitalize()
    if "@" in ans:
        local, dom = ans.split("@", 1)
        return re.split(r"[-._]", local)[0] + "@" + dom if re.search(r"[-._]", local) else "info@" + dom
    quoted = " ".join(re.findall(r'"([^"]+)"', task.prompt)) or task.prompt
    if re.search(r"\d", ans):
        cands = re.findall(r"[\w.\-]*[\d@][\w.\-@]*", quoted)
    else:
        cands = re.findall(r"\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b", quoted)
    cands = [c for c in cands if c not in _STOP]
    cands = [c.strip(".") for c in cands if c.strip(".").lower() != ans.lower() and c.strip(".").lower() not in ans.lower()]
    return cands[int(u * len(cands)) % len(cands)] if cands else "unknown"


_LANGS = {"french", "spanish", "italian", "german", "portuguese", "english", "dutch"}
_STOP = {"The", "A", "An", "After", "Before", "Hi", "Questions", "Write", "Je", "It", "This", "Please", "I",
         "Dr", "Two", "On", "In", "When", "We", "My", "Our"}


def hash_rank(s: str, u: float) -> float:
    import hashlib
    return int.from_bytes(hashlib.sha256(f"{s}|{u:.6f}".encode()).digest()[:4], "big")
