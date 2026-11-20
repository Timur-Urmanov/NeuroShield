"""Task sets with automatic checking: math (numeric answer), code (unit tests), text (exact match).

Built-in items live in data/{math,code,text}.jsonl (hand-written, GSM8K/MBPP style).
External files in the same formats can be dropped into `<repo>/data/`:
    data/gsm8k*.jsonl  {"question": ..., "answer": "... #### 42"}       (GSM8K)
    data/mbpp*.jsonl   {"task_id", "text", "code", "test_list", "test_setup_code"}   (MBPP)
    data/text*.jsonl   {"id", "prompt", "answer", "labels"?, "aliases"?}
`Task.prompt` is what the models see and what is embedded; the task *kind* is never given to the
router (it only gets the embedding of `prompt`).
"""
from __future__ import annotations

import glob
import json
import os
import re
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
KINDS = ("math", "code", "text")

FORMAT = {
    "math": "Solve the problem. End your reply with a line of the form 'Final answer: <number>'.",
    "code": ("Write the Python function. Reply with a single ```python code block containing the "
             "complete function definition (plus any imports it needs). Do not include tests or examples."),
    "text": "Reply with a final line of the form 'Answer: <your answer>' and nothing after it.",
}


@dataclass
class Task:
    id: str
    kind: str                       # math | code | text  (hidden from the router)
    prompt: str
    answer: object = None           # float for math, str for text
    tests: list = field(default_factory=list)
    setup: str = ""
    reference: str = ""             # reference solution (code); used by SimProvider + self-tests
    entry_point: str = ""
    labels: Optional[list] = None
    aliases: list = field(default_factory=list)
    source: str = "builtin"

    @property
    def kind_idx(self) -> int:
        return KINDS.index(self.kind)


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _num(s) -> float:
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s)
    if "####" in s:
        s = s.split("####")[-1]
    return float(s.strip().replace(",", "").replace("$", ""))


def load_gsm8k(path, source=None) -> list:
    out = []
    for i, r in enumerate(_read_jsonl(path)):
        out.append(Task(id=str(r.get("id", f"{os.path.basename(path)}:{i}")), kind="math",
                        prompt=r["question"].strip(), answer=_num(r["answer"]), source=source or path))
    return out


def _entry_point(r) -> str:
    if r.get("entry_point"):
        return r["entry_point"]
    m = re.search(r"assert\s+(?:not\s+)?\(?\s*([A-Za-z_]\w*)\s*\(", r["test_list"][0])
    if m:
        return m.group(1)
    m = re.search(r"def\s+([A-Za-z_]\w*)\s*\(", r.get("code", ""))
    return m.group(1) if m else ""


def load_mbpp(path, source=None) -> list:
    out = []
    for i, r in enumerate(_read_jsonl(path)):
        ep = _entry_point(r)
        prompt = (f"{r['text'].strip()}\nThe function must be named `{ep}`. It must pass tests such as:\n"
                  f"{r['test_list'][0]}")
        out.append(Task(id=str(r.get("task_id", f"{os.path.basename(path)}:{i}")), kind="code", prompt=prompt,
                        tests=list(r["test_list"]) + list(r.get("challenge_test_list") or []),
                        setup=r.get("test_setup_code") or "", reference=r.get("code", ""), entry_point=ep,
                        source=source or path))
    return out


def load_text(path, source=None) -> list:
    out = []
    for i, r in enumerate(_read_jsonl(path)):
        prompt = r["prompt"].strip()
        out.append(Task(id=str(r.get("id", f"{os.path.basename(path)}:{i}")), kind="text", prompt=prompt,
                        answer=str(r["answer"]), labels=r.get("labels"), aliases=list(r.get("aliases") or []),
                        source=source or path))
    return out


def load_builtin() -> list:
    return (load_gsm8k(os.path.join(DATA, "math.jsonl"), "builtin")
            + load_mbpp(os.path.join(DATA, "code.jsonl"), "builtin")
            + load_text(os.path.join(DATA, "text.jsonl"), "builtin"))


def load_external(data_dir: str) -> list:
    out = []
    if not data_dir or not os.path.isdir(data_dir):
        return out
    for p in sorted(glob.glob(os.path.join(data_dir, "gsm8k*.jsonl"))):
        out += load_gsm8k(p)
    for p in sorted(glob.glob(os.path.join(data_dir, "mbpp*.jsonl"))):
        out += load_mbpp(p)
    for p in sorted(glob.glob(os.path.join(data_dir, "text*.jsonl"))):
        out += load_text(p)
    return out


def load_pool(task_cfg: dict, root: str) -> list:
    pool = load_builtin() if task_cfg.get("builtin", True) else []
    ext = task_cfg.get("external_dir")
    if ext:
        pool += load_external(ext if os.path.isabs(ext) else os.path.join(root, ext))
    if not pool:
        raise ValueError("empty task pool")
    return pool


def make_stream(pool: list, n: int, seed: int) -> list:
    """n tasks: concatenated seeded permutations of the pool (items repeat only if n > len(pool))."""
    rng = np.random.default_rng(seed + 50_000)
    out = []
    while len(out) < n:
        out += [pool[i] for i in rng.permutation(len(pool))]
    return out[:n]


def task_message(task: Task) -> str:
    """User message for the executor. The <task> delimiters are also used by SimProvider."""
    return f"<task>\n{task.prompt}\n</task>\n\n{FORMAT[task.kind]}"
