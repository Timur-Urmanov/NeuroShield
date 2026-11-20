"""Centralised baselines on the same task stream (same seeds, same U => paired comparison).

Model policies:
  linucb   - discounted LinUCB over models on the embedding x (legacy, exact port)
  oracle   - knows the task type and current accuracies, picks argmax acc*q - lam*cost
  single:<m> - always model m

Team policies (only matter in roles/ordered envs; team size = `team`):
  solo      - one random primitive
  random    - `team` distinct random primitives
  agent_ucb - top-`team` primitives by individual UCB on success (individual credit)
  pair_ucb  - combinatorial bandit over pairs: best pair by UCB, then the third member maximising
              the sum of pair-UCBs with the first two. Learns which *pairs* work, like W does,
              but centrally and with explicit exploration.
  oracle    - a team covering all roles (and in the right order for the ordered env)
"""
from __future__ import annotations

import numpy as np

from .envs import make_env


class LinUCB:
    def __init__(self, M, D, alpha=0.4, gamma=0.995):
        self.M, self.alpha, self.gamma = M, alpha, gamma
        self.d = D + 1
        self.A = np.stack([np.eye(self.d)] * M); self.b = np.zeros((M, self.d))

    def choose(self, x):
        z = np.append(x, 1.0)
        sc = []
        for m in range(self.M):
            Ai = np.linalg.inv(self.A[m]); th = Ai @ self.b[m]
            sc.append(th @ z + self.alpha * np.sqrt(z @ Ai @ z))
        return int(np.argmax(sc))

    def update(self, x, m, r):
        z = np.append(x, 1.0)
        self.A = self.gamma * self.A + (1 - self.gamma) * np.eye(self.d)[None]
        self.b = self.gamma * self.b
        self.A[m] += np.outer(z, z); self.b[m] += r * z


class TeamPolicy:
    def __init__(self, kind, N, team, rng, env, c=0.3):
        self.kind, self.N, self.team, self.rng, self.env, self.c = kind, N, team, rng, env, c
        self.t = 0
        if kind == "pair_ucb":
            self.n = np.zeros((N, N)); self.s = np.zeros((N, N))
        if kind == "agent_ucb":
            self.n = np.zeros(N); self.s = np.zeros(N)

    def _ucb(self):
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(self.n > 0, self.s / np.maximum(self.n, 1)
                         + self.c * np.sqrt(np.log(self.t + 2) / np.maximum(self.n, 1)), 1e6)
        return u + 1e-9 * self.rng.random(u.shape)       # random tie-break

    def choose(self):
        N, k = self.N, self.team
        if self.kind == "solo":
            return np.array([self.rng.integers(N)])
        if self.kind == "random":
            return self.rng.choice(N, size=k, replace=False)
        if self.kind == "oracle":
            role = self.env.role
            return np.array([self.rng.choice(np.where(role == j)[0]) for j in range(k)])
        if self.kind == "agent_ucb":
            return np.argsort(-self._ucb())[:k]
        if self.kind == "pair_ucb":
            U = self._ucb(); np.fill_diagonal(U, -np.inf)
            a, b = np.unravel_index(int(np.argmax(U)), U.shape)
            team = [a, b]
            while len(team) < k:
                sc = U[team].sum(0); sc[team] = -np.inf
                team.append(int(np.argmax(sc)))
            return np.array(team)
        raise ValueError(self.kind)

    def update(self, team, succ):
        self.t += 1
        if self.kind == "agent_ucb":
            self.n[team] += 1; self.s[team] += succ
        elif self.kind == "pair_ucb":
            for a in team:
                for b in team:
                    if a != b:
                        self.n[a, b] += 1; self.s[a, b] += succ


def run_router(env_cfg, seed, N=40, model="linucb", team="solo", team_size=3, c=0.3,
               alpha=0.4, gamma=0.995):
    """Returns dict(R=rewards, q=team factor, cover=coverage)."""
    env = make_env(env_cfg, seed, N)
    T = env_cfg.T
    rng = np.random.default_rng(seed + 7)
    lin = LinUCB(env.M, env.D, alpha, gamma) if model == "linucb" else None
    tp = TeamPolicy(team, N, team_size, rng, env, c) if env.kind != "homogeneous" else None
    R = np.zeros(T); Q = np.zeros(T); COV = np.full(T, np.nan)
    for t in range(T):
        x, k, u = env.X[t], env.ks[t], env.U[t]
        members = tp.choose() if tp else np.array([0])
        q = env.team_factor(members)
        if model == "linucb":
            m = lin.choose(x)
        elif model == "oracle":
            m = int(np.argmax(env.acc(t)[:, k] * q - env.lam * env.cost))
        elif model.startswith("single:"):
            m = int(model.split(":")[1])
        else:
            raise ValueError(model)
        succ = float(u < env.acc(t)[m, k] * q)
        r = env.reward(succ, m, members)
        if lin:
            lin.update(x, m, r)
        if tp:
            tp.update(members, succ)
            c_ = env.coverage(members)
            COV[t] = float(c_)
        R[t], Q[t] = r, q
    return dict(R=R, q=Q, cover=COV)


def summary(out, T):
    R, cov = out["R"], out["cover"]
    has = not np.all(np.isnan(cov))
    return dict(pre=float(R[T // 2 - 1000:T // 2].mean()), post=float(R[-1000:].mean()),
                cover_start=float(np.nanmean(cov[:1000])) if has else None,
                cover_end=float(np.nanmean(cov[-2000:])) if has else None,
                q_end=float(out["q"][-2000:].mean()))
