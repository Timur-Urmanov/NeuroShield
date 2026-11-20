"""LLMEnv: the Swarm's env interface backed by real (or simulated) LLM calls.

How a real outcome reaches the unchanged Swarm.step():
    succ = float(u < env.p_success(t, m, exec_order) * (1 - 0.3 * fat))
  * U[t] = 0.5 for every t;
  * p_success() actually EXECUTES the team on task t with model m (plan -> execute -> verify by team
    size, see swarm/llm/roles.py), runs the checker and returns 1.0 or 0.0.
  So u < p reproduces the checker verdict. (With the artificial `fatigue` flag on, the factor
  1 - 0.3*fat >= 0.7 > 0.5 keeps the verdict unchanged; fatigue is off in SwarmConfig() anyway.)
  * reward(succ, m, members) is called right after, for the same step, and uses the actual cost of
    that execution: succ - lam * usd_nominal / cost_norm_usd.
  Swarm.step calls p_success exactly once per step; `execute()` is the same path for baselines.

X[t] = embedding of the task prompt (task kind is never exposed to the swarm; `ks` exists only for
analysis, like in the simulated envs). `cost[m]` = expected normalised cost of one executor call.
"""
from __future__ import annotations

from functools import partial

import numpy as np

from ..llm.roles import ROLE_PROMPTS, TeamResult, roles_for_team, run_team
from ..llm.tasks import task_message
from ..llm.tasks.checkers import check


class LLMEnv:
    kind = "llm"
    member_cost = 0.0

    def __init__(self, cfg, client, tasks: list, embedder, seed: int, sample_id: int = 0):
        self.cfg, self.client, self.tasks, self.seed = cfg, client, list(tasks), seed
        self.T = len(self.tasks)
        self.models = cfg.names
        self.M = len(self.models)
        self.lam = float(cfg.run["lam"])
        self.cost_norm = float(cfg.run["cost_norm_usd"])
        self.max_tokens = dict(cfg.run["max_tokens"])
        self.sample_id = int(sample_id)
        self.X = embedder.embed([t.prompt for t in self.tasks])
        self.D = self.X.shape[1]
        self.U = np.full(self.T, 0.5)
        self.ks = np.array([t.kind_idx for t in self.tasks])      # analysis only
        self.N = None
        tc = cfg.tasks
        self.check_fn = partial(check, timeout=float(tc["code_timeout_s"]), mem_mb=int(tc["code_mem_mb"]))
        self.cost = self.expected_costs()
        self.records: list = []
        self._last: TeamResult | None = None

    # ------------------------------------------------------------------ cost model
    def expected_costs(self) -> np.ndarray:
        """Normalised expected cost of one executor call per model: mean prompt length (chars/4) of
        this task stream + sim out_tokens['executor'] as the output-length assumption."""
        n_in = np.mean([(len(task_message(t)) + len(ROLE_PROMPTS["executor"])) / 4 for t in self.tasks])
        return np.array([m.usd(n_in, m.sim["out_tokens"].get("executor", 300)) / self.cost_norm
                         for m in self.cfg.models])

    # ------------------------------------------------------------------ execution
    def execute(self, t: int, m: int, n_members: int, tag: str = "") -> TeamResult:
        roles = roles_for_team(n_members)
        res = run_team(self.client, self.models[m], self.tasks[t], roles, self.max_tokens,
                       sample_id=self.sample_id, check_fn=self.check_fn)
        self._last = res
        self.records.append(dict(t=t, task=self.tasks[t].id, kind=self.tasks[t].kind, model=self.models[m],
                                 roles=roles, success=res.success, usd=res.usd_nominal, charged=res.usd_charged,
                                 calls=res.n_calls, hits=res.n_cache_hits, revised=res.revised, variant=tag,
                                 reward=self.step_reward(res), verdict=res.verdict,
                                 max_tokens_stops=res.n_max_tokens, refusals=res.n_refusals))
        return res

    def step_reward(self, res: TeamResult) -> float:
        return float(res.success) - self.lam * res.usd_nominal / self.cost_norm

    # ------------------------------------------------------------------ Swarm interface
    def p_success(self, t: int, m: int, members) -> float:
        return 1.0 if self.execute(t, m, len(members), tag="swarm").success else 0.0

    def reward(self, succ: float, m: int, members) -> float:
        usd = self._last.usd_nominal if self._last is not None else 0.0
        return float(succ) - self.lam * usd / self.cost_norm

    def team_factor(self, members, t=None) -> float:
        return 1.0

    def available(self, t: int):          # no capacity limits on primitives (all available)
        return None

    def commit(self, t: int, members) -> None:
        pass

    def coverage(self, members):
        return None

