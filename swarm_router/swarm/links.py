"""Computed links (H11): the link of a pair is a function of per-primitive vectors, not a stored number.

  w_ab = p0 + b_a + b_b + 0.5 * (p_a . q_b + p_b . q_a)
p ("what I offer") and q ("what I need") are d-vectors; in the contextual variant they are linear in the
task embedding x: p_a(x) = P_a @ x. Learning is local SGD on err = success - w_ab for each pair of the
team; only the team members' parameters change. Used by the swarm (w_rule=factor|factor_ctx) and by the
centralised baselines (team=mf|mf_ctx).
"""
from __future__ import annotations

import numpy as np


class FactorLinks:
    def __init__(self, N: int, D: int, d: int = 4, lr: float = 0.05, ctx: bool = False, seed: int = 0,
                 init: float = 0.1):
        rng = np.random.default_rng(seed + 6000)          # separate stream
        self.N, self.d, self.lr, self.ctx = N, d, lr, ctx
        shape = (N, d, D) if ctx else (N, d)
        self.P = init * rng.normal(size=shape)
        self.Q = init * rng.normal(size=shape)
        self.b = np.zeros(N)

    def vecs(self, idx, x):
        """(p, q) for primitives idx (None = all) on task x -> arrays (len, d)."""
        P = self.P if idx is None else self.P[idx]
        Q = self.Q if idx is None else self.Q[idx]
        return (P @ x, Q @ x) if self.ctx else (P, Q)

    def rows(self, idx, x, p0: float) -> np.ndarray:
        """Predicted link of each primitive in idx with every primitive: (len(idx), N). O(N d)."""
        idx = np.atleast_1d(idx)
        pa, qa = self.vecs(idx, x)
        pall, qall = self.vecs(None, x)
        R = p0 + self.b[idx][:, None] + self.b[None, :] + 0.5 * (pa @ qall.T + qa @ pall.T)
        R[np.arange(len(idx)), idx] = 0
        return R

    def full(self, x, p0: float) -> np.ndarray:
        """All pairs (N, N): only the centralised baseline needs this. O(N^2 d)."""
        p, q = self.vecs(None, x)
        A = p @ q.T
        return p0 + self.b[:, None] + self.b[None, :] + 0.5 * (A + A.T)

    def update(self, team, succ: float, p0: float, x) -> None:
        team = np.asarray(team)
        if len(team) < 2:
            return
        p, q = self.vecs(team, x)                       # values before the update
        gb = np.zeros(len(team)); gp = np.zeros_like(p); gq = np.zeros_like(q)
        for i in range(len(team)):
            for j in range(i + 1, len(team)):
                err = succ - (p0 + self.b[team[i]] + self.b[team[j]] + 0.5 * (p[i] @ q[j] + p[j] @ q[i]))
                gb[i] += err; gb[j] += err
                gp[i] += 0.5 * err * q[j]; gq[j] += 0.5 * err * p[i]
                gp[j] += 0.5 * err * q[i]; gq[i] += 0.5 * err * p[j]
        self.b[team] += self.lr * gb
        if self.ctx:
            self.P[team] += self.lr * gp[:, :, None] * x[None, None, :]
            self.Q[team] += self.lr * gq[:, :, None] * x[None, None, :]
        else:
            self.P[team] += self.lr * gp
            self.Q[team] += self.lr * gq
