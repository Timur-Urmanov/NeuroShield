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
  mf / mf_ctx - centralised version of the swarm's computed links (H11, swarm/links.py).
  pair_mean - centralised version of the swarm's link estimator (neutral prior, greedy, conditional third).
  cluster_pair_mean / cluster_agent_ucb - the same with separate statistics per online cluster of x.
All team policies respect the env's capacity mask (busy primitives).
  oracle    - a team covering all roles (and in the right order for the ordered env)
"""
from __future__ import annotations

import numpy as np

from .envs import make_env
from .links import FactorLinks


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


class SparsePairMean:
    """cluster_pair_mean on sparse statistics (H17): only visited pairs are stored. Unvisited pairs all have the
    estimate p0, so the dense argmax is reproduced by: best visited free pair if its estimate > p0, else a random
    free unvisited pair (same for the third member). Work per step ~ number of visited pairs, not N^2."""
    def __init__(self, N, team, rng, n_clusters=3):
        self.N, self.team, self.rng = N, team, rng
        self.clusters = OnlineClusters(n_clusters)
        self.C = n_clusters
        self.idx = [dict() for _ in range(n_clusters)]            # (a, b), a < b -> row
        self.nb = [dict() for _ in range(n_clusters)]             # node -> {partner: row}
        self.A = [np.zeros(64, int) for _ in range(n_clusters)]; self.B = [np.zeros(64, int) for _ in range(n_clusters)]
        self.s = [np.zeros(64) for _ in range(n_clusters)]; self.n = [np.zeros(64) for _ in range(n_clusters)]
        self.m = [0] * n_clusters
        self.succ_sum, self.succ_n, self.ctx = 0.0, 0, 0

    @property
    def state_size(self):
        return int(sum(self.m))

    def choose(self, t, x, avail=None):
        c = self.ctx = self.clusters.assign(x)
        N, m, rng = self.N, self.m[c], self.rng
        av = np.ones(N, bool) if avail is None else avail
        p0 = self.succ_sum / self.succ_n if self.succ_n else 0.5
        team = None
        if m:
            A, B = self.A[c][:m], self.B[c][:m]
            est = np.where(av[A] & av[B], (self.s[c][:m] + p0) / (self.n[c][:m] + 1), -np.inf)
            j = int(np.argmax(est))
            if est[j] > p0:
                team = [int(A[j]), int(B[j])]
        if team is None:
            free = np.where(av)[0]
            for _ in range(50):                                   # a random free unvisited pair
                a, b = (int(v) for v in rng.choice(free, size=2, replace=False))
                if (min(a, b), max(a, b)) not in self.idx[c]:
                    break
            team = [a, b]
        while len(team) < self.team:
            sc = {}
            for a in team:
                for j, row in self.nb[c].get(a, {}).items():
                    if av[j] and j not in team:
                        sc.setdefault(j, {})[a] = (self.s[c][row] + p0) / (self.n[c][row] + 1)
            best, bv = None, -np.inf
            for j, d in sc.items():
                v = (sum(d.values()) + p0 * (len(team) - len(d))) / len(team)
                if v > bv:
                    best, bv = j, v
            if best is None or bv <= p0:
                free = np.where(av)[0]
                for _ in range(50):                               # a random free node unknown to the team
                    j = int(rng.choice(free))
                    if j not in team and j not in sc:
                        best = j
                        break
                if best is None:
                    best = int(next(j for j in free if j not in team))
            team.append(best)
        return np.array(team)

    def update(self, team, succ):
        c = self.ctx
        self.succ_sum += succ; self.succ_n += 1
        for i in range(len(team)):
            for k in range(i + 1, len(team)):
                a, b = int(min(team[i], team[k])), int(max(team[i], team[k]))
                row = self.idx[c].get((a, b))
                if row is None:
                    row = self.m[c]; self.m[c] += 1
                    if row == len(self.A[c]):
                        for arr in (self.A, self.B, self.s, self.n):
                            arr[c] = np.concatenate([arr[c], np.zeros_like(arr[c])])
                    self.idx[c][(a, b)] = row; self.A[c][row] = a; self.B[c][row] = b
                    self.nb[c].setdefault(a, {})[b] = row; self.nb[c].setdefault(b, {})[a] = row
                self.s[c][row] += succ; self.n[c][row] += 1


class TeamPolicy:
    """kind in solo | random | oracle | agent_ucb | pair_ucb | cluster_pair_ucb."""
    def __init__(self, kind, N, team, rng, env, c=0.3, n_clusters=3, lr=0.05, d=4, seed=0,
                 gamma=1.0, reset_at=None):
        self.gamma, self.reset_at = gamma, reset_at       # H16: discounted statistics / informed reset
        clustered = kind in ("cluster_pair_mean", "cluster_agent_ucb", "cluster_pair_ucb")
        if kind in ("cluster_pair_mean", "cluster_agent_ucb"):    # H16 addendum: per-cluster statistics
            kind = kind[len("cluster_"):]
        self.kind, self.N, self.team, self.rng, self.env, self.c = kind, N, team, rng, env, c
        self.fl = FactorLinks(N, env.D, d, lr, kind == "mf_ctx", seed) if kind in ("mf", "mf_ctx") else None
        self.t = 0
        self.ctx = 0
        C = n_clusters if clustered else 1
        self.clusters = OnlineClusters(C) if clustered else None
        self.succ_sum, self.succ_n = 0.0, 0
        if kind in ("pair_ucb", "cluster_pair_ucb", "seq_ucb", "pair_mean"):
            self.n = np.zeros((C, N, N)); self.s = np.zeros((C, N, N))
        if kind == "agent_ucb":
            self.n = np.zeros((C, N)); self.s = np.zeros((C, N))

    def _ucb(self):
        n, s = self.n[self.ctx], self.s[self.ctx]
        if self.gamma < 1.0:          # discounted UCB: effective horizon 1/(1-gamma); decayed-out arms are new again
            lg = np.log(min(self.t + 2, 1.0 / (1.0 - self.gamma)))
            nn = np.maximum(n, 0.05)
            u = np.where(n > 0.05, s / nn + self.c * np.sqrt(lg / nn), 1e6)
            return u + 1e-9 * self.rng.random(u.shape)
        u = np.where(n > 0, s / np.maximum(n, 1) + self.c * np.sqrt(np.log(self.t + 2) / np.maximum(n, 1)), 1e6)
        return u + 1e-9 * self.rng.random(u.shape)       # random tie-break

    def choose(self, t, x, avail=None):
        N, k = self.N, self.team
        if self.reset_at is not None and t == self.reset_at:      # told about the shift: forget everything
            self.succ_sum, self.succ_n = 0.0, 0
            if hasattr(self, "n"):
                self.n[:] = 0; self.s[:] = 0
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
        if self.kind == "pair_mean":
            # centralised counterpart of the swarm's link rule (H10): estimate = (s + p0) / (n + 1) with the
            # neutral prior p0 = running mean team success; greedy best free pair, then the member that
            # maximises the mean estimate to the current team (= the swarm's conditional recruitment).
            p0 = self.succ_sum / self.succ_n if self.succ_n else 0.5
            n, s_ = self.n[self.ctx], self.s[self.ctx]
            E = (s_ + p0) / (n + 1) + 1e-9 * self.rng.random(n.shape)
            np.fill_diagonal(E, -np.inf)
            if avail is not None:
                E[~avail] = -np.inf; E[:, ~avail] = -np.inf
            a, b = np.unravel_index(int(np.argmax(E)), E.shape)
            team = [int(a), int(b)]
            while len(team) < k:
                sc = E[team].mean(0); sc[team] = -np.inf
                team.append(int(np.argmax(sc)))
            return np.array(team)
        if self.kind in ("mf", "mf_ctx"):
            # centralised counterpart of the swarm's computed links (H11): same model and update rule,
            # but the best free pair is chosen globally, then the third by mean link to the team.
            p0 = self.succ_sum / self.succ_n if self.succ_n else 0.5
            E = self.fl.full(x, p0)          # real-valued predictions: no tie-break noise needed (O(N^2) saved)
            np.fill_diagonal(E, -np.inf)
            if avail is not None:
                E[~avail] = -np.inf; E[:, ~avail] = -np.inf
            a, b = np.unravel_index(int(np.argmax(E)), E.shape)
            team = [int(a), int(b)]
            while len(team) < k:
                sc = E[team].mean(0); sc[team] = -np.inf
                team.append(int(np.argmax(sc)))
            self._x = x
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
        if self.kind in ("mf", "mf_ctx"):
            self.fl.update(team, succ, self.succ_sum / self.succ_n if self.succ_n else 0.5, self._x)
        self.succ_sum += succ; self.succ_n += 1
        if not hasattr(self, "n"):
            return
        if self.gamma < 1.0:
            self.n *= self.gamma; self.s *= self.gamma
            self.succ_sum *= self.gamma; self.succ_n *= self.gamma
        n, s = self.n[self.ctx], self.s[self.ctx]
        if self.kind == "agent_ucb":
            n[team] += 1; s[team] += succ
        elif self.kind == "seq_ucb":
            for i in range(len(team)):
                for j in range(i + 1, len(team)):
                    n[team[i], team[j]] += 1; s[team[i], team[j]] += succ
        elif self.kind in ("pair_ucb", "cluster_pair_ucb", "pair_mean"):
            for a in team:
                for b in team:
                    if a != b:
                        n[a, b] += 1; s[a, b] += succ


def run_router(env_cfg, seed, N=40, model="linucb", team="solo", team_size=3, c=0.3,
               alpha=0.4, gamma=0.995, lr=0.05, team_gamma=1.0, reset_at=None, role_k=None):
    """Returns dict(R=rewards, q=team factor, cover=coverage).
    team_gamma / reset_at: forgetting in the team policy (H16). role_k: replace the hidden roles (K x N) by a
    given population, e.g. the one a swarm with inheritance ended with (H16c)."""
    env = make_env(env_cfg, seed, N)
    if role_k is not None:
        env.role_k = np.array(role_k); env.role = env.role_k[0].copy()
    T = env_cfg.T
    rng = np.random.default_rng(seed + 7)
    lin = LinUCB(env.M, env.D, alpha, gamma) if model == "linucb" else None
    if team == "cluster_pair_mean_sparse":
        tp = SparsePairMean(N, team_size, rng)
    else:
        tp = TeamPolicy(team, N, team_size, rng, env, c, lr=lr, seed=seed, gamma=team_gamma, reset_at=reset_at) \
            if env.kind != "homogeneous" else None
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
    return dict(R=R, q=Q, cover=COV, policy=tp)


def summary(out, T):
    R, cov = out["R"], out["cover"]
    has = not np.all(np.isnan(cov))
    return dict(pre=float(R[T // 2 - 1000:T // 2].mean()), post=float(R[-1000:].mean()),
                cover_start=float(np.nanmean(cov[:1000])) if has else None,
                cover_end=float(np.nanmean(cov[-2000:])) if has else None,
                q_end=float(out["q"][-2000:].mean()))
