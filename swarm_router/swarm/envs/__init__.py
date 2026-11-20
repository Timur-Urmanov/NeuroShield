from .base import ACC0, ACC1, COST, D, K, LAM, M, MIX, SimEnv, expected, make_stream
from .homogeneous import HomogeneousEnv
from .llm import LLMEnv  # noqa: F401  (Phase 2; optional deps are imported lazily)
from .ordered import OrderedEnv
from .roles import RolesEnv, assign_roles


def make_env(env_cfg, seed: int, N: int):
    kw = dict(N=N, role_fractions=env_cfg.role_fractions, strong=env_cfg.strong, weak=env_cfg.weak,
              member_cost=env_cfg.member_cost, role_by_type=env_cfg.role_by_type, busy=env_cfg.busy)
    if env_cfg.kind == "homogeneous":
        return HomogeneousEnv(env_cfg.T, seed, N, env_cfg.member_cost)
    if env_cfg.kind == "roles":
        return RolesEnv(env_cfg.T, seed, **kw)
    if env_cfg.kind == "ordered":
        return OrderedEnv(env_cfg.T, seed, **kw)
    raise ValueError(env_cfg.kind)
