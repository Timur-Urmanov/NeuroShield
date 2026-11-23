"""Configuration dataclasses. Every results file stores `asdict(cfg)` next to the numbers."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Optional, Sequence


@dataclass(frozen=True)
class SwarmConfig:
    """Swarm mechanisms. Defaults = current minimal architecture (CLAUDE.md, section 2).

    Mechanisms are independent flags; do NOT merge them again (e.g. `coupling` != `plastic_W`).
    """
    N: int = 40
    team: int = 3                 # max team size (leader + team-1 recruits)
    coupling: bool = True         # teams at all (False = solo leader)
    plastic_W: bool = True        # learn pairwise link matrix W
    hebb_bias: float = 0.0        # unconditional Hebbian increment (legacy: 0.3); 0 = reward-only Hebb
    # W learning rule (H8): mean_succ (default since H8c: per-pair mean success) | mean_r |
    #   hebb_r (Hebb on r - EMA(r); default up to H8, used by every experiment before H8c) | hebb_succ
    #   factor | factor_ctx (H11): links computed from per-primitive vectors (swarm/links.py)
    w_rule: str = "mean_succ"
    factor_d: int = 4
    factor_lr: float = 0.05
    # H9: recruit_rule independent (each recruit by W[leader]) | conditional (one by one, mean W to the
    #     current team); w_prior (mean_* rules): w0 (random initial W) | mean (running mean team success)
    #     defaults since H9d: conditional + mean; every experiment before H9d is pinned to independent + w0
    recruit_rule: str = "conditional"
    w_prior: str = "mean"
    sim_w: float = 0.5            # s in score = (1-s)*W[leader] + s*sim
    recruit_thr: float = 0.25     # minimum recruit score
    # exploration bonus in recruitment (H7): none | ucb (count-based, directed) | noise (undirected control)
    # recruit="oracle": diagnostic upper bound (uses hidden roles; post-hoc to H7, not a mechanism)
    recruit: str = "learned"
    explore: str = "none"
    explore_c: float = 0.1
    hormones: bool = True         # NE stress hormone (temperature + forgetting)
    # kuramoto / phase_filter: removed after H6 (no gain for step ordering, harmful in recruitment)
    fatigue: bool = False         # built-in (artificial) quality degradation of busy primitives
    sleep: bool = False           # sleep: offline consolidation of working memory
    sleep_prune: bool = True      # on waking, zero links W < prune_thr (only matters if sleep & plastic_W)
    prune_thr: float = 0.05
    # ordering of team members for envs where step order matters (envs/ordered.py)
    #   recruit  - leader first, then by recruitment score (what the swarm does implicitly)
    #   random   - random permutation
    #   position - learned per-primitive scalar, reward-modulated
    #   oracle   - by hidden roles (upper bound, not a mechanism)
    order: str = "recruit"
    order_lr: float = 0.1
    # inheritance and selection (H12): none | full | nolinks | random
    inherit: str = "none"
    inherit_every: int = 100      # tasks between generations
    inherit_frac: float = 0.02    # fraction of primitives replaced per generation
    inherit_mut: float = 0.05     # probability that a child's hidden role is random


# Defaults of the original swarm.py constructor minus Kuramoto. Legacy rows with kuramoto=True are
# reproducible only with legacy/swarm.py or git commit d2d10b0.
LEGACY = SwarmConfig(fatigue=True, sleep=True, hebb_bias=0.3, w_rule="hebb_r", recruit_rule="independent",
                     w_prior="w0")


@dataclass(frozen=True)
class EnvConfig:
    kind: str = "homogeneous"     # homogeneous | roles | ordered
    T: int = 8000
    role_fractions: Optional[Sequence[float]] = None  # None -> arange(N) % 3 (exact thirds)
    strong: float = 0.95          # quality of a primitive in its own role
    weak: float = 0.6             # quality in a foreign role
    member_cost: float = 0.0      # reward penalty per extra team member (legacy: 0)
    role_by_type: bool = False    # strong role of a primitive depends on task type
    busy: int = 0                 # steps a primitive stays busy after a task (capacity); 0 = unlimited


def to_dict(*cfgs) -> dict:
    out = {}
    for c in cfgs:
        d = asdict(c)
        if "role_fractions" in d and d["role_fractions"] is not None:
            d["role_fractions"] = list(d["role_fractions"])
        out[type(c).__name__] = d
    return out


__all__ = ["SwarmConfig", "EnvConfig", "LEGACY", "replace", "to_dict", "field"]
