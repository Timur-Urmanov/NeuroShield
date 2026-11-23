"""Run one (env, swarm config, seed) and collect per-step traces."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import EnvConfig, SwarmConfig
from .core import Swarm
from .envs import make_env


@dataclass
class Trace:
    R: np.ndarray
    models: np.ndarray
    ks: np.ndarray
    leaders: np.ndarray
    team_size: np.ndarray
    coverage: np.ndarray          # NaN in envs without roles
    q: np.ndarray                 # team quality factor actually applied
    teams: list = field(default_factory=list)
    swarm: Swarm | None = None
    env: object = None            # the env instance after the run (keep=True); roles may have evolved (H12)

    def window(self, name: str, T: int) -> slice:
        return {"pre": slice(T // 2 - 1000, T // 2), "post": slice(T - 1000, T),
                "end2000": slice(T - 2000, T), "start1000": slice(0, 1000)}[name]

    def summary(self) -> dict:
        T = len(self.R)
        w = lambda n: self.window(n, T)
        cov = self.coverage
        return dict(pre=float(self.R[w("pre")].mean()), post=float(self.R[w("post")].mean()),
                    cover_start=float(np.nanmean(cov[w("start1000")])) if not np.all(np.isnan(cov)) else None,
                    cover_end=float(np.nanmean(cov[w("end2000")])) if not np.all(np.isnan(cov)) else None,
                    q_end=float(self.q[w("end2000")].mean()),
                    team_size=float(self.team_size[w("end2000")].mean()))


def run_swarm(env_cfg: EnvConfig, cfg: SwarmConfig, seed: int, keep=False) -> Trace:
    env = make_env(env_cfg, seed, cfg.N)
    s = Swarm(cfg, seed, env.M, env.D)
    T = env_cfg.T
    R = np.zeros(T); Ms = np.zeros(T, int); L = np.zeros(T, int); TS = np.zeros(T, int)
    COV = np.full(T, np.nan); Q = np.zeros(T)
    teams = []
    for t in range(T):
        r, info = s.step(t, env)
        R[t], Ms[t], L[t] = r, info["m"], info["leader"]
        TS[t] = len(info["members"])
        c = env.coverage(info["members"], t)
        if c is not None:
            COV[t] = float(c)
        Q[t] = env.team_factor(info["exec_order"], t)
        if keep:
            teams.append(info["exec_order"].tolist())
    return Trace(R, Ms, env.ks, L, TS, COV, Q, teams, s if keep else None, env if keep else None)
