"""Configuration dataclasses. Every results file stores `asdict(cfg)` next to the numbers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Optional, Sequence


@dataclass(frozen=True)
class SwarmConfig:
    """Swarm mechanisms. Defaults = current minimal architecture (CLAUDE.md, section 2).

    Mechanisms are independent flags; do NOT merge them again (e.g. `coupling` != `kuramoto`).
    """
    N: int = 40
    team: int = 3                 # max team size (leader + team-1 recruits)
    coupling: bool = True         # teams at all (False = solo leader)
    plastic_W: bool = True        # learn pairwise link matrix W
    hebb_bias: float = 0.0        # unconditional Hebbian increment (legacy: 0.3); 0 = reward-only Hebb
    sim_w: float = 0.5            # s in score = (1-s)*W[leader] + s*sim
    recruit_thr: float = 0.25     # minimum recruit score
    hormones: bool = True         # NE stress hormone (temperature + forgetting)
    kuramoto: bool = False        # phase oscillators
    phase_filter: Optional[bool] = None  # phase gating in recruitment; None -> same as `kuramoto`
    fatigue: bool = False         # built-in (artificial) quality degradation of busy primitives
    sleep: bool = False           # sleep: offline consolidation of working memory
    sleep_prune: bool = True      # on waking, zero links W < prune_thr (only matters if sleep & plastic_W)
    prune_thr: float = 0.05
    # ordering of team members for envs where step order matters (envs/ordered.py)
    #   recruit  - leader first, then by recruitment score (what the swarm does implicitly)
    #   random   - random permutation
    #   position - learned per-primitive scalar, reward-modulated (no oscillators)
    #   phase    - order by Kuramoto phase, same reward-modulated rule applied to theta
    order: str = "recruit"
    order_lr: float = 0.1

    @property
    def use_phase_filter(self) -> bool:
        return self.kuramoto if self.phase_filter is None else self.phase_filter


# Defaults of the original swarm.py constructor. Used by the regression test and to
# reproduce the original tables.
LEGACY = SwarmConfig(kuramoto=True, fatigue=True, sleep=True, hebb_bias=0.3)


@dataclass(frozen=True)
class EnvConfig:
    kind: str = "homogeneous"     # homogeneous | roles | ordered
    T: int = 8000
    role_fractions: Optional[Sequence[float]] = None  # None -> arange(N) % 3 (exact thirds)
    strong: float = 0.95          # quality of a primitive in its own role
    weak: float = 0.6             # quality in a foreign role
    member_cost: float = 0.0      # reward penalty per extra team member (legacy: 0)


def to_dict(*cfgs) -> dict:
    out = {}
    for c in cfgs:
        d = asdict(c)
        if "role_fractions" in d and d["role_fractions"] is not None:
            d["role_fractions"] = list(d["role_fractions"])
        out[type(c).__name__] = d
    return out


__all__ = ["SwarmConfig", "EnvConfig", "LEGACY", "replace", "to_dict", "field"]
