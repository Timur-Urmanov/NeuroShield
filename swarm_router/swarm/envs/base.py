"""Simulated task stream + simulated LLMs (shared by all simulated envs).

Task type k is hidden; the swarm sees only a noisy unit embedding x in R^D.
U[t] is the common random number deciding the outcome of task t; it is shared by all
variants run on the same seed (common random numbers for paired ablations).
"""
from __future__ import annotations

import numpy as np

D, K, M = 8, 3, 3
# ACC[model, type]
ACC0 = np.array([[0.90, 0.55, 0.60],   # A: large, strong on "math"
                 [0.55, 0.88, 0.62],   # B: large, strong on "code"
                 [0.70, 0.70, 0.80]])  # C: cheap, strong on "text"
ACC1 = ACC0.copy(); ACC1[0, 0] = 0.60; ACC1[2, 0] = 0.85   # after the regime shift
COST = np.array([1.0, 1.0, 0.3])
LAM = 0.15
MIX = np.array([0.4, 0.3, 0.3])
TASK_NOISE = 0.35


def make_stream(T: int, seed: int):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(K, D)); C /= np.linalg.norm(C, axis=1, keepdims=True)
    ks = rng.choice(K, size=T, p=MIX)
    X = C[ks] + rng.normal(scale=TASK_NOISE, size=(T, D))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    U = rng.random(T)
    return ks, X, U


def expected(acc, policy, q: float = 1.0):
    """Expected reward of policy[k] -> distribution over models, team quality factor q."""
    val = acc * q - LAM * COST[:, None]
    return float(sum(MIX[k] * (policy[k] @ val[:, k]) for k in range(K)))


class SimEnv:
    """Homogeneous env: team composition does not change success probability."""
    kind = "homogeneous"
    D, K, M = D, K, M
    cost = COST
    lam = LAM

    member_cost = 0.0   # cost per extra team member (each member = one more LLM call); legacy: 0

    def __init__(self, T: int, seed: int, N: int = 40, member_cost: float = 0.0):
        self.T, self.seed, self.N = T, seed, N
        self.member_cost = member_cost
        self.ks, self.X, self.U = make_stream(T, seed)

    def acc(self, t: int):
        return ACC0 if t < (getattr(self, 'shift_at', 0) or self.T // 2) else ACC1

    def team_factor(self, members, t=None) -> float:
        return 1.0

    # capacity: primitives busy after a task (None = no capacity limit)
    def available(self, t: int):
        return None

    def commit(self, t: int, members) -> None:
        pass

    # distributed setting (H19): None = single node
    def home_mask(self, t: int):
        return None

    def remote_pen(self, t: int):
        return None

    def clone(self, src: int, dst: int, mutate: bool, rng) -> None:
        """Inheritance (H12): primitive dst becomes a copy of src. No hidden properties here."""
        pass

    def p_success(self, t: int, m: int, members) -> float:
        """Success probability before the swarm's own (artificial) fatigue penalty."""
        return self.acc(t)[m, self.ks[t]] * self.team_factor(members, t)

    def reward(self, succ: float, m: int, members) -> float:
        return succ - self.lam * self.cost[m] - self.member_cost * (len(members) - 1)

    def coverage(self, members, t=None):
        return None

    # analytic baselines ---------------------------------------------------
    def oracle_value(self, phase: str, q: float = 1.0) -> float:
        acc = ACC0 if phase == "pre" else ACC1
        val = acc * q - LAM * COST[:, None]
        return expected(acc, [np.eye(M)[a] for a in val.argmax(0)], q)

    def best_single_value(self, phase: str, q: float = 1.0) -> float:
        acc = ACC0 if phase == "pre" else ACC1
        return max(expected(acc, [np.eye(M)[m]] * K, q) for m in range(M))
