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
  cluster_pair_ucb - online k-means on x (told the number of task types) + one pair_ucb per cluster.
  seq_ucb   - order-aware variant: UCB over *directed* pairs (a acts before b); the sequence is grown
              greedily at the head or tail. Team order = returned order.
All team policies respect the env's capacity mask (busy primitives).
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


class OnlineClusters:
    """Online competitive k-means on unit vectors (the baseline is told the number of clusters)."""
    def __init__(self, C, lr=0.05):
        self.C, self.lr, self.c = C, lr, []

    def assign(self, x):
        if len(self.c) < self.C and (not self.c or max(float(c @ x) for c in self.c) < 0.6):
            self.c.append(x.copy())
            return len(self.c) - 1
        j = int(np.argmax([c @ x for c in self.c]))
        self.c[j] = self.c[j] + self.lr * (x - self.c[j]); self.c[j] /= np.linalg.norm(self.c[j])
        return j


class TeamPolicy:
    """kind in solo | random | oracle | agent_ucb | pair_ucb | cluster_pair_ucb."""
    def __init__(self, kind, N, team, rng, env, c=0.3, n_clusters=3):
        self.kind, self.N, self.team, self.rng, self.env, self.c = kind, N, team, rng, env, c
        self.t = 0
        self.ctx = 0
        C = n_clusters if kind == "cluster_pair_ucb" else 1
        self.clusters = OnlineClusters(C) if kind == "cluster_pair_ucb" else None
        if kind in ("pair_ucb", "cluster_pair_ucb", "seq_ucb"):
            self.n = np.zeros((C, N, N)); self.s = np.zeros((C, N, N))
        if kind == "agent_ucb":
            self.n = np.zeros((1, N)); self.s = np.zeros((1, N))

    def _ucb(self):
        n, s = self.n[self.ctx], self.s[self.ctx]
        u = np.where(n > 0, s / np.maximum(n, 1) + self.c * np.sqrt(np.log(self.t + 2) / np.maximum(n, 1)), 1e6)
        return u + 1e-9 * self.rng.random(u.shape)       # random tie-break

    def choose(self, t, x, avail=None):
        N, k = self.N, self.team
        free = np.arange(N) if avail is None else np.where(avail)[0]
        if self.clusters is not None:
            self.ctx = self.clusters.assign(x)
        if self.kind == "solo":
            return np.array([self.rng.integers(N)]) if avail is None else np.array([self.rng.choice(free)])
        if self.kind == "random":
            return self.rng.choice(N, size=k, replace=False) if avail is None else self.rng.choice(free, size=k, replace=False)
        if self.kind == "oracle":
            role = self.env.roles_at(t)
            team = []
            for j in range(k):
                cand = np.setdiff1d(free[role[free] == j], team)
                if len(cand) == 0:
                    cand = np.setdiff1d(free, team)
                team.append(int(self.rng.choice(cand)))
            return np.array(team)
        if self.kind == "agent_ucb":
            u = self._ucb()
            if avail is not None:
                u = np.where(avail, u, -np.inf)
            return np.argsort(-u)[:k]
        if self.kind in ("pair_ucb", "cluster_pair_ucb"):
            U = self._ucb(); np.fill_diagonal(U, -np.inf)
            if avail is not None:
                U[~avail] = -np.inf; U[:, ~avail] = -np.inf
            a, b = np.unravel_index(int(np.argmax(U)), U.shape)
            team = [a, b]
            while len(team) < k:
                sc = U[team].sum(0); sc[team] = -np.inf
                team.append(int(np.argmax(sc)))
            return np.array(team)
        if self.kind == "seq_ucb":
            # directed pairs: n[a, b] counts teams where a acts before b
            U = self._ucb(); np.fill_diagonal(U, -np.inf)
            if avail is not None:
                U[~avail] = -np.inf; U[:, ~avail] = -np.inf
            a, b = np.unravel_index(int(np.argmax(U)), U.shape)
            seq = [int(a), int(b)]
            while len(seq) < k:
                tail = U[seq].sum(0); head = U[:, seq].sum(1)
                tail[seq] = -np.inf; head[seq] = -np.inf
                if tail.max() >= head.max():
                    seq.append(int(np.argmax(tail)))
                else:
                    seq.insert(0, int(np.argmax(head)))
            return np.array(seq)
        raise ValueError(self.kind)

    def update(self, team, succ):
        self.t += 1
        if not hasattr(self, "n"):
            return
        n, s = self.n[self.ctx], self.s[self.ctx]
        if self.kind == "agent_ucb":
            n[team] += 1; s[team] += succ
        elif self.kind == "seq_ucb":
            for i in range(len(team)):
                for j in range(i + 1, len(team)):
                    n[team[i], team[j]] += 1; s[team[i], team[j]] += succ
        elif self.kind in ("pair_ucb", "cluster_pair_ucb"):
            for a in team:
                for b in team:
                    if a != b:
                        n[a, b] += 1; s[a, b] += succ


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
        members = tp.choose(t, x, env.available(t)) if tp else np.array([0])
        env.commit(t, members)
        q = env.team_factor(members, t)
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
            COV[t] = float(env.coverage(members, t))
        R[t], Q[t] = r, q
    return dict(R=R, q=Q, cover=COV)


def summary(out, T):
    R, cov = out["R"], out["cover"]
    has = not np.all(np.isnan(cov))
    return dict(pre=float(R[T // 2 - 1000:T // 2].mean()), post=float(R[-1000:].mean()),
                cover_start=float(np.nanmean(cov[:1000])) if has else None,
                cover_end=float(np.nanmean(cov[-2000:])) if has else None,
                q_end=float(out["q"][-2000:].mean()))
