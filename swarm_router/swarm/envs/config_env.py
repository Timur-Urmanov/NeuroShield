"""H20: primitives with a rewritable configuration instead of a fixed hidden role.

A configuration is L genes with A values each. For every task type k and role j there is a hidden target string;
the skill of a primitive in role j on type k is weak + (strong - weak) * (matches / L) ** 2. Team quality is the
product over roles of the best skill in the team (as in RolesEnv). The system may rewrite a configuration:
copy another primitive (optionally with a one-gene mutation) or draw a random one.
"""
from __future__ import annotations

import numpy as np

from .roles import N_ROLES, RolesEnv


class ConfigEnv(RolesEnv):
    kind = "config"

    def __init__(self, T, seed, N=40, strong=0.95, weak=0.6, member_cost=0.0, busy=0, genes_L=6, genes_A=4):
        super().__init__(T, seed, N, None, strong, weak, member_cost, False, busy)
        self.L, self.A = genes_L, genes_A
        self.genes = np.random.default_rng(seed + 2900).integers(genes_A, size=(N, genes_L))
        self.targets = np.random.default_rng(seed + 2901).integers(genes_A, size=(self.K, N_ROLES, genes_L))
        self.skill = np.zeros((self.K, N, N_ROLES))
        self._recompute(np.arange(N))

    def _recompute(self, idx):
        m = (self.genes[idx][None, :, None, :] == self.targets[:, None, :, :]).sum(-1)      # K x n x roles
        self.skill[:, idx, :] = self.weak + (self.strong - self.weak) * (m / self.L) ** 2

    def team_factor(self, members, t=None) -> float:
        k = self.ks[t] if t is not None else 0
        return float(np.prod(self.skill[k, np.asarray(members)].max(0)))

    def coverage(self, members, t=None) -> float:
        """Not a boolean here: team quality relative to a perfect team."""
        return self.team_factor(members, t) / self.q_full

    def roles_at(self, t):
        return self.skill[self.ks[t] if t is not None else 0].argmax(1)

    def clone(self, src, dst, mutate, rng):
        self.genes[dst] = self.genes[src]
        if mutate:                                   # one-gene mutation to a different value
            g = int(rng.integers(self.L))
            self.genes[dst, g] = (self.genes[dst, g] + rng.integers(1, self.A)) % self.A
        self._recompute(np.array([dst]))

    def randomize(self, dst, rng):
        self.genes[dst] = rng.integers(self.A, size=self.L)
        self._recompute(np.array([dst]))
