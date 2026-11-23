"""Complementary roles: each primitive has a hidden strong role (planner / executor / verifier).

Team quality = prod over roles of the best member quality in that role (strong if someone in the team
has the role, weak otherwise). Role identity is an env property; the swarm never sees it.
"""
from __future__ import annotations


import numpy as np

from .base import SimEnv

N_ROLES = 3


def assign_roles(N: int, fractions=None, seed: int = 0) -> np.ndarray:
    if fractions is None:
        return np.arange(N) % N_ROLES          # legacy: exact thirds, interleaved
    f = np.asarray(fractions, float); f = f / f.sum()
    counts = np.floor(f * N).astype(int)
    for i in np.argsort(-(f * N - counts))[: N - counts.sum()]:
        counts[i] += 1
    role = np.repeat(np.arange(N_ROLES), counts)
    # separate RNG stream: does not touch the task stream or the swarm RNG
    return np.random.default_rng(seed + 2000).permutation(role)


class RolesEnv(SimEnv):
    kind = "roles"

    def __init__(self, T, seed, N=40, role_fractions=None, strong=0.95, weak=0.6, member_cost=0.0,
                 role_by_type=False, busy=0, role_shift=0.0,
                 nodes=0, comm_cost=0.0, node_skew=0.0):
        super().__init__(T, seed, N, member_cost)
        self.role = assign_roles(N, role_fractions, seed)
        self.strong, self.weak = strong, weak
        # role_by_type: the strong role of a primitive differs per task type (independent permutation
        # per type, separate RNG). role_k[k] is the role vector for tasks of type k.
        self.role_by_type = role_by_type
        if role_by_type:
            rr = np.random.default_rng(seed + 2500)
            self.role_k = np.stack([rr.permutation(self.role) for _ in range(self.K)])
        else:
            self.role_k = np.stack([self.role] * self.K)
        # busy: a primitive that worked on task t is unavailable for tasks t+1 .. t+busy
        self.busy = busy
        self.busy_until = np.full(N, -1)
        # role_shift (H16): at T/2 the hidden roles of this fraction of primitives are permuted among them
        self.role_shift, self._shifted = role_shift, False
        # distributed setting (H19)
        self.nodes, self.comm_cost = nodes, comm_cost
        if nodes:
            self.node = np.arange(N) * nodes // N
            self.home = np.random.default_rng(seed + 2700).integers(nodes, size=T)
            self.remote_calls = np.zeros(T, int); self._last_remote = 0
            if node_skew:
                rs = np.random.default_rng(seed + 2800)
                for k in range(self.K):
                    for h in range(nodes):
                        idx = rs.permutation(np.where(self.node == h)[0])
                        miss = (h + (k if role_by_type else 0)) % N_ROLES
                        present = [r for r in range(N_ROLES) if r != miss]
                        self.role_k[k, idx] = np.array(present)[np.arange(len(idx)) % 2]
                self.role = self.role_k[0].copy()

    def home_mask(self, t):
        return self.node == self.home[t] if self.nodes else None

    def remote_pen(self, t):
        return self.comm_cost * (self.node != self.home[t]) if self.nodes else None

    def reward(self, succ, m, members):
        r = super().reward(succ, m, members)
        return r - self.comm_cost * self._last_remote if self.nodes else r

    def _shift(self):
        self._shifted = True
        rr = np.random.default_rng(self.seed + 2600)
        idx = rr.choice(self.N, size=int(round(self.role_shift * self.N)), replace=False)
        if self.role_by_type:
            for k in range(self.K):
                self.role_k[k, idx] = self.role_k[k, rr.permutation(idx)]
        else:
            self.role[idx] = self.role[rr.permutation(idx)]
            self.role_k[:] = self.role[None]

    def roles_at(self, t):
        if self.role_shift and not self._shifted and t is not None and t >= self.T // 2:
            self._shift()
        if t is None or not self.role_by_type:
            return self.role
        return self.role_k[self.ks[t]]

    def team_factor(self, members, t=None) -> float:
        role = self.roles_at(t)
        comp = np.where(role[members][:, None] == np.arange(N_ROLES)[None], self.strong, self.weak)
        return float(np.prod(comp.max(0)))

    def coverage(self, members, t=None) -> bool:
        return len(set(self.roles_at(t)[members].tolist())) == N_ROLES

    def clone(self, src, dst, mutate, rng):
        """dst inherits the hidden role(s) of src; a mutated child gets a random role instead."""
        if mutate:
            self.role_k[:, dst] = rng.integers(N_ROLES, size=self.K) if self.role_by_type else rng.integers(N_ROLES)
        else:
            self.role_k[:, dst] = self.role_k[:, src]
        if not self.role_by_type:
            self.role[dst] = self.role_k[0, dst]

    def available(self, t):
        if not self.busy:
            return None
        return self.busy_until < t

    def commit(self, t, members):
        if self.nodes:            # remote calls of the team just committed; charged by reward() for this task
            self._last_remote = int((self.node[np.asarray(members)] != self.home[t]).sum())
            self.remote_calls[t] = self._last_remote
        if self.busy:
            self.busy_until[np.asarray(members)] = t + self.busy

    @property
    def q_full(self) -> float:
        return self.strong ** N_ROLES

    def expected_random_team_q(self, size: int = 3) -> float:
        """Exact E[team_factor] for a uniformly random team of `size` distinct primitives."""
        counts = np.bincount(self.role, minlength=N_ROLES)
        tot, num = 0.0, 0
        # exact via combinatorics: iterate over how many of each role are drawn
        from math import comb
        for a in range(size + 1):
            for b in range(size + 1 - a):
                c = size - a - b
                w = comb(counts[0], a) * comb(counts[1], b) * comb(counts[2], c)
                if w == 0:
                    continue
                q = np.prod([self.strong if n > 0 else self.weak for n in (a, b, c)])
                tot += w * q; num += w
        return tot / num
