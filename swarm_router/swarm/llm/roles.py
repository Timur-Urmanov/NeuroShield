"""Role system prompts and the team protocol plan -> execute -> verify (one revision max).

Roles by team size (team_roles = "position", members taken in execution order):
    1 -> [executor]
    2 -> [executor, verifier]
    3+ -> [planner, executor, verifier]   (members beyond 3 make no calls)
All members of one step use the same model m (as in the simulation).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .tasks import FORMAT, task_message

ROLE_PROMPTS = {
    "planner": ("You are the PLANNER in a small team. Read the task and write a short numbered plan "
                "(at most 6 steps) that another model will follow to solve it. Point out pitfalls. "
                "Do not solve the task and do not give a final answer."),
    "executor": ("You are the EXECUTOR in a small team. Solve the task carefully. If a plan is given, use it "
                 "when it helps. Follow the required output format exactly."),
    "verifier": ("You are the VERIFIER in a small team. You get a task and a proposed answer. Check the answer "
                 "independently and critically. Reply with 'VERDICT: ACCEPT' if it is correct and in the "
                 "required format, otherwise 'VERDICT: REVISE' followed by a short explanation of the problem. "
                 "Do not write the corrected solution yourself."),
}
PROMPT_TO_ROLE = {v: k for k, v in ROLE_PROMPTS.items()}


def roles_for_team(n_members: int) -> list:
    if n_members <= 1:
        return ["executor"]
    if n_members == 2:
        return ["executor", "verifier"]
    return ["planner", "executor", "verifier"]


def executor_message(task, plan: Optional[str] = None) -> str:
    msg = task_message(task)
    if plan:
        msg += f"\n\n<plan>\n{plan.strip()}\n</plan>"
    return msg


def planner_message(task) -> str:
    return f"<task>\n{task.prompt}\n</task>\n\nWrite the plan."


def verifier_message(task, candidate: str) -> str:
    return (f"<task>\n{task.prompt}\n</task>\n\nRequired output format: {FORMAT[task.kind]}\n\n"
            f"<answer>\n{candidate}\n</answer>\n\nCheck the proposed answer.")


def revision_message(feedback: str) -> str:
    return (f"A reviewer flagged a problem with your reply:\n<feedback>\n{feedback.strip()}\n</feedback>\n"
            "Fix it and give your complete final reply again, in the required format.")


_VERDICT = re.compile(r"VERDICT\s*:\s*(ACCEPT|REVISE)", re.I)


def parse_verdict(text: str) -> bool:
    """True = accept. Last VERDICT line wins; no verdict -> accept (do not spend a revision on noise)."""
    m = _VERDICT.findall(text or "")
    return not (m and m[-1].upper() == "REVISE")


@dataclass
class TeamResult:
    success: bool
    answer: str
    usd_nominal: float = 0.0          # cost as if nothing were cached (used in the reward)
    usd_charged: float = 0.0          # what the budget was actually charged
    n_calls: int = 0
    n_cache_hits: int = 0
    revised: bool = False
    roles: list = field(default_factory=list)
    transcript: list = field(default_factory=list)   # [(role, user_msg, reply)]

    def add(self, call, role, user_msg):
        self.usd_nominal += call.usd_nominal
        self.usd_charged += call.usd_charged
        self.n_calls += 1
        self.n_cache_hits += int(call.cache_hit)
        self.transcript.append((role, user_msg, call.text))


def run_team(client, model: str, task, roles: list, max_tokens: dict, sample_id: int = 0,
             history: Optional[list] = None, check_fn=None) -> TeamResult:
    """history: optional prior messages (user/assistant turns) prepended to the executor conversation."""
    from .tasks.checkers import check
    check_fn = check_fn or check
    res = TeamResult(False, "", roles=list(roles))
    plan = None
    if "planner" in roles:
        um = planner_message(task)
        c = client.call(model, ROLE_PROMPTS["planner"], [{"role": "user", "content": um}],
                        max_tokens["planner"], sample_id=sample_id, tag="planner")
        res.add(c, "planner", um); plan = c.text
    um = executor_message(task, plan)
    conv = list(history or []) + [{"role": "user", "content": um}]
    c = client.call(model, ROLE_PROMPTS["executor"], conv, max_tokens["executor"], sample_id=sample_id,
                    tag="executor")
    res.add(c, "executor", um); answer = c.text
    if "verifier" in roles:
        vm = verifier_message(task, answer)
        v = client.call(model, ROLE_PROMPTS["verifier"], [{"role": "user", "content": vm}],
                        max_tokens["verifier"], sample_id=sample_id, tag="verifier")
        res.add(v, "verifier", vm)
        if not parse_verdict(v.text):
            rm = revision_message(v.text)
            conv2 = conv + [{"role": "assistant", "content": answer or "(empty)"}, {"role": "user", "content": rm}]
            c2 = client.call(model, ROLE_PROMPTS["executor"], conv2, max_tokens["executor"],
                             sample_id=sample_id, tag="executor_revision")
            res.add(c2, "executor_revision", rm); answer = c2.text; res.revised = True
    res.answer = answer
    res.success = bool(check_fn(task, answer))
    return res
