"""Ordered steps: plan -> execute -> verify. Step j is done by the member at position j of the
executed order (positions wrap around for teams smaller than 3). Team quality = prod_j quality of
the member doing step j in role j. Same roles as RolesEnv, but the *order* of members now matters:
a team with all three roles in the wrong order is no better than a team missing roles.

Purpose: test the hypothesis "phase (Kuramoto) is useful for temporal coordination".
"""
from __future__ import annotations

import numpy as np

from .roles import N_ROLES, RolesEnv


class OrderedEnv(RolesEnv):
    kind = "ordered"

    def team_factor(self, members) -> float:
        members = np.asarray(members)
        L = len(members)
        who = members[np.arange(N_ROLES) % L]          # member doing step j
        return float(np.prod(np.where(self.role[who] == np.arange(N_ROLES), self.strong, self.weak)))

    def correct_order(self, members) -> bool:
        members = np.asarray(members)
        return len(members) == N_ROLES and bool(np.all(self.role[members] == np.arange(N_ROLES)))
