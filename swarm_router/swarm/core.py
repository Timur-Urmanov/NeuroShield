"""Swarm of primitives routing tasks -> team -> model.

A primitive is a cheap stateful agent; it does not "think", it stores:
  - affinity vector aff (specialisation, competitive learning)
  - model value estimates Q[m] (long-term QL/nL + optional working memory sW/nW for sleep)
  - links W to other primitives (learned by reward-prediction-error Hebb)
  - optional: fatigue F, sleep state
Kuramoto phases were removed after H6 (notes/hypotheses.md); see legacy/swarm.py or commit d2d10b0.

Global slow variables ("hormones"):
  - dopa = reward prediction error (r - fast running mean)
  - NE   = stress: recent reward below long-term -> higher choice temperature, faster forgetting
  - mel  = mean fatigue -> sleep pressure (only with fatigue+sleep)

IMPORTANT: the order of calls to self.rng is the same as in legacy/swarm.py. Changing it breaks the
bit-exact regression test (tests/test_regression.py). New randomness must use a separate generator.
"""
from __future__ import annotations

import numpy as np

from .config import SwarmConfig
from .links import FactorLinks


class Swarm:
    def __init__(self, cfg: SwarmConfig, seed: int, M: int, D: int):
        self.cfg, self.M, self.D = cfg, M, D
        N = self.N = cfg.N
        self.rng = r = np.random.default_rng(seed + 1000)
        self.rng_order = np.random.default_rng(seed + 3000)   # separate stream: random ordering only
        # The legacy model drew Kuramoto phases and frequencies here. The draws are kept (values unused)
        # so that the RNG stream, and therefore every regression number, stays identical.
        r.uniform(0, 2 * np.pi, N)
        r.normal(1.0, 0.1, N)
        self.aff = r.normal(size=(N, D)); self.aff /= np.linalg.norm(self.aff, axis=1, keepdims=True)
        self.QL = np.full((N, M), 0.6); self.nL = np.ones((N, M))
        self.sW = np.zeros((N, M)); self.nW = np.zeros((N, M))
        W = r.uniform(0, 0.2, (N, N)); self.W = (W + W.T) / 2; np.fill_diagonal(self.W, 0)
        self.pos = self.rng_order.uniform(0, 1, N)               # learned ordering scalar (order=position)
        self.rng_explore = np.random.default_rng(seed + 4000)  # separate stream: explore="noise" only
        self.n_co = np.zeros((N, N))                           # co-work counts (explore="ucb")
        self.W0 = self.W.copy()                                # prior for w_rule=mean_* (H8)
        self.nW_pair = np.zeros((N, N)); self.sW_pair = np.zeros((N, N))
        self.s_fast = 0.5                                      # EMA of success (w_rule=hebb_succ)
        self.fl = (FactorLinks(N, D, cfg.factor_d, cfg.factor_lr, cfg.w_rule == "factor_ctx", seed)
                   if cfg.w_rule.startswith("factor") else None)
        self.team_succ_sum = 0.0; self.team_n = 0              # running mean team success (w_prior=mean)
        self.t_seen = 0
        self.F = np.zeros(N); self.asleep = np.zeros(N, bool); self.timer = np.zeros(N, int)
        self.r_fast = self.r_slow = 0.5
        self.NE = 0.0

    # ---------- helpers ----------
    def Qest(self, idx):
        return (self.QL[idx] * self.nL[idx] + self.sW[idx]) / (self.nL[idx] + self.nW[idx])

    def consolidate(self, i):
        """Sleep: working memory -> long-term memory; optionally prune weak links."""
        n = self.nL[i] + self.nW[i]
        self.QL[i] = (self.QL[i] * self.nL[i] + self.sW[i]) / n
        self.nL[i] = np.minimum(n, 100.0)
        self.sW[i] = 0; self.nW[i] = 0
        if not (self.cfg.plastic_W and self.cfg.sleep_prune):
            return
        weak = self.W[i] < self.cfg.prune_thr
        self.W[i, weak] = 0; self.W[weak, i] = 0

    def execution_order(self, members: np.ndarray, env, t: int) -> np.ndarray:
        o = self.cfg.order
        if o == "recruit" or len(members) == 1:
            return members
        if o == "random":
            return self.rng_order.permutation(members)
        if o == "position":
            return members[np.argsort(self.pos[members], kind="stable")]
        if o == "oracle":                     # upper bound, uses hidden roles
            return members[np.argsort(env.roles_at(t)[members], kind="stable")]
        raise ValueError(o)

    def learn_order(self, exec_order: np.ndarray, dopa: float):
        """Reward-modulated ordering: consecutive pair (a before b) is pushed apart if dopa > 0,
        towards a swap if dopa < 0. (H6: the same rule on a Kuramoto phase performed at chance level.)"""
        if len(exec_order) < 2 or self.cfg.order != "position":
            return
        step = self.cfg.order_lr * dopa
        for a, b in zip(exec_order[:-1], exec_order[1:]):
            self.pos[a] -= step; self.pos[b] += step

    # ---------- one step = one task ----------
    def step(self, t: int, env):
        cfg, N, M = self.cfg, self.N, self.M
        x, u = env.X[t], env.U[t]
        awake = ~self.asleep
        avail = env.available(t)                     # capacity mask from the env (None = all)
        free = awake if avail is None else awake & avail
        NE = self.NE if cfg.hormones else 0.0

        # 1. bids: how much the task looks like "mine"
        bids = (self.aff @ x) * (1 - 0.5 * self.F) + 0.02 * self.rng.normal(size=N)
        bids[~free] = -np.inf
        leader = int(np.argmax(bids))

        # 2. team: score = (1-s) W[leader] + s sim   (legacy multiplied by a phase gate; `* 1.0` keeps
        #    the float result bit-identical to the legacy no-Kuramoto path)
        members = [leader]
        if cfg.coupling:
            sim = np.clip(self.aff @ x, 0, None)
            neutral = cfg.w_prior == "mean" and cfg.w_rule.startswith("mean")
            factor = self.fl is not None
            p0 = (self.team_succ_sum / self.team_n if self.team_n else 0.5) if (neutral or factor) else None

            def Wrows(idx):
                """Rows idx of the effective link matrix (untried pairs -> p0 under the neutral prior).
                Row-wise on purpose: building the full N x N matrix every step is O(N^2)."""
                idx = np.atleast_1d(idx)
                if factor:
                    return self.fl.rows(idx, x, p0)
                if not neutral:
                    return self.W[idx]
                R = np.where(self.nW_pair[idx] > 0, self.W[idx], p0)
                R[np.arange(len(idx)), idx] = 0
                return R
            Wl = Wrows(leader)[0]
            sc = ((1 - cfg.sim_w) * Wl + cfg.sim_w * sim) * 1.0
            if cfg.explore == "ucb":
                sc = sc + cfg.explore_c * np.sqrt(np.log(self.t_seen + 2) / (self.n_co[leader] + 1))
            elif cfg.explore == "noise":
                sc = sc + cfg.explore_c * np.abs(self.rng_explore.normal(size=N))
            elif cfg.explore != "none":
                raise ValueError(cfg.explore)
            sc[~free] = -1; sc[leader] = -1
            if cfg.recruit == "oracle":
                # upper bound for leader-centred teams: fill missing roles with free primitives
                role = env.roles_at(t)
                for need in [k for k in range(3) if k != role[leader]][: cfg.team - 1]:
                    c = np.where(free & (role == need))[0]
                    c = c[c != leader]
                    if len(c):
                        members.append(int(c[np.argmax(sc[c])]))
            elif cfg.recruit_rule == "conditional":
                bonus = sc - ((1 - cfg.sim_w) * Wl + cfg.sim_w * sim)       # exploration term, if any
                for _ in range(cfg.team - 1):
                    c = (1 - cfg.sim_w) * Wrows(members).mean(0) + cfg.sim_w * sim + bonus
                    c[~free] = -1; c[members] = -1
                    j = int(np.argmax(c))
                    if c[j] <= cfg.recruit_thr:
                        break
                    members.append(j)
            else:
                cand = np.argsort(-sc)[: cfg.team - 1]
                members += [int(j) for j in cand if sc[j] > cfg.recruit_thr]
        members = np.array(members)
        env.commit(t, members)
        self.t_seen += 1
        if cfg.explore == "ucb" and len(members) > 1:
            self.n_co[np.ix_(members, members)] += 1

        # 3. model choice from the team's shared value memory
        rel = np.clip(self.aff[members] @ x, 0.05, None)
        Qt = (rel[:, None] * self.Qest(members)).sum(0) / rel.sum()
        temp = 0.05 * (1 + 2 * NE) if cfg.hormones else 0.06
        p = np.exp((Qt - Qt.max()) / temp); p /= p.sum()
        m = int(self.rng.choice(M, p=p))

        # 4. outcome
        exec_order = self.execution_order(members, env, t)
        fat = self.F[members].mean() if cfg.fatigue else 0.0
        succ = float(u < env.p_success(t, m, exec_order) * (1 - 0.3 * fat))
        r = env.reward(succ, m, exec_order)

        # 5. hormones
        dopa = r - self.r_fast
        self.r_fast += (r - self.r_fast) / 30
        self.r_slow += (r - self.r_slow) / 400
        self.NE = float(np.clip((self.r_slow - self.r_fast) / 0.1, 0, 1))
        mel = self.F[awake].mean() if awake.any() else 0.0

        # 6. learning
        eta = 0.05 * (1 + NE) if cfg.hormones else 0.05
        for j in members:
            if cfg.sleep:
                self.sW[j, m] += r; self.nW[j, m] += 1
            else:                                   # online consolidation
                self.QL[j, m] = (self.QL[j, m] * self.nL[j, m] + r) / (self.nL[j, m] + 1)
                self.nL[j, m] = min(self.nL[j, m] + 1, 100.0)
            # competitive learning: pull towards own tasks, harder on success
            w_pull = (1.0 if j == leader else 0.5) * max(0.0, 0.5 + dopa)
            self.aff[j] += eta * w_pull * x
            self.aff[j] /= np.linalg.norm(self.aff[j])
        rule = cfg.w_rule
        if rule == "hebb_r":
            if len(members) > 1 and cfg.plastic_W:      # Hebb inside the team
                for a in members:
                    for b in members:
                        if a != b:
                            self.W[a, b] = np.clip(self.W[a, b] + 0.5 * eta * (dopa + cfg.hebb_bias), 0, 1)
            if cfg.plastic_W:
                self.W *= (1 - 0.0005)
        elif rule == "hebb_succ":
            ds = succ - self.s_fast
            if len(members) > 1 and cfg.plastic_W:
                for a in members:
                    for b in members:
                        if a != b:
                            self.W[a, b] = np.clip(self.W[a, b] + 0.5 * eta * (ds + cfg.hebb_bias), 0, 1)
            if cfg.plastic_W:
                self.W *= (1 - 0.0005)
        elif rule in ("mean_r", "mean_succ"):
            if len(members) > 1 and cfg.plastic_W:
                sig = r if rule == "mean_r" else succ
                ix = np.ix_(members, members)
                self.nW_pair[ix] += 1; self.sW_pair[ix] += sig
                np.fill_diagonal(self.nW_pair, 0); np.fill_diagonal(self.sW_pair, 0)
                prior = self.W0[ix]
                if cfg.w_prior == "mean":
                    prior = self.team_succ_sum / self.team_n if self.team_n else 0.5
                self.W[ix] = (self.sW_pair[ix] + prior) / (self.nW_pair[ix] + 1)
                np.fill_diagonal(self.W, 0)
        elif rule in ("factor", "factor_ctx"):
            if len(members) > 1 and cfg.plastic_W:
                self.fl.update(members, succ, self.team_succ_sum / self.team_n if self.team_n else 0.5, x)
        else:
            raise ValueError(rule)
        if len(members) > 1:
            self.team_succ_sum += succ; self.team_n += 1
        self.s_fast += (succ - self.s_fast) / 30
        if cfg.sleep:
            self.sW *= 0.998; self.nW *= 0.998      # working memory leaks
        if cfg.hormones and NE > 0:                 # stress -> revisit old estimates
            self.nL = np.maximum(1.0, self.nL * (1 - 0.02 * NE))
        self.learn_order(exec_order, dopa)

        # 7. fatigue and sleep
        if cfg.fatigue:
            self.F[members] += 0.04 * (1 - self.F[members])
            self.F[awake] *= 0.999
        if cfg.sleep:
            woke = self.asleep & (self.timer <= 0)
            for i in np.where(woke)[0]:
                self.asleep[i] = False; self.consolidate(i)
            self.timer[self.asleep] -= 1
            self.F[self.asleep] *= 0.9
            thr = 0.6 * (1 - 0.3 * mel) if cfg.hormones else 0.6
            for i in np.where(~self.asleep & (self.F > thr))[0]:
                if self.asleep.mean() < 0.3:
                    self.asleep[i] = True; self.timer[i] = 40

        info = dict(leader=leader, members=members, exec_order=exec_order, m=m, succ=succ)
        return r, info


def clusters(W, thr=0.4):
    N = len(W); seen = np.zeros(N, bool); out = []
    for s in range(N):
        if seen[s]:
            continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            i = stack.pop(); comp.append(i)
            for j in np.where((W[i] > thr) & ~seen)[0]:
                seen[j] = True; stack.append(j)
        if len(comp) >= 2:
            out.append(sorted(comp))
    return out
