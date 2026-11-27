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

    def choose(self, t, x, avail=None, pen=None):
        """pen (H19): per-primitive cost of a remote call; None = single node. Unvisited pairs are then drawn
        among zero-cost (local) primitives first, and every estimate is reduced by the cost of its members."""
        c = self.ctx = self.clusters.assign(x)
        N, m, rng = self.N, self.m[c], self.rng
        av = np.ones(N, bool) if avail is None else avail
        pz = np.zeros(N) if pen is None else pen
        pool = np.where(av & (pz == 0))[0]                        # where unvisited candidates come from
        if len(pool) < self.team:
            pool = np.where(av)[0]
        p0 = self.succ_sum / self.succ_n if self.succ_n else 0.5
        team = None
        if m:
            A, B = self.A[c][:m], self.B[c][:m]
            est = np.where(av[A] & av[B], (self.s[c][:m] + p0) / (self.n[c][:m] + 1) - pz[A] - pz[B], -np.inf)
            j = int(np.argmax(est))
            if est[j] > p0:
                team = [int(A[j]), int(B[j])]
        if team is None:
            for _ in range(50):                                   # a random free unvisited pair
                a, b = (int(v) for v in rng.choice(pool, size=2, replace=False))
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
                v = (sum(d.values()) + p0 * (len(team) - len(d))) / len(team) - pz[j]
                if v > bv:
                    best, bv = j, v
            if best is None or bv <= p0:
                cand = None
                for _ in range(50):                               # a random free node unknown to the team
                    j = int(rng.choice(pool))
                    if j not in team and j not in sc:
                        cand = j
                        break
                if cand is None and best is None:
                    cand = int(next(j for j in np.where(av)[0] if j not in team))
                best = cand if cand is not None else best
            team.append(best)
        return np.array(team)

    def update(self, team, succ, x=None):
        c = self.ctx if x is None else self.clusters.assign(x)    # x given: an outcome reported from another node
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


def _sp_set(b, c, a, x, s_, n_):
    """SparsePairMean: create or overwrite the statistics of pair (a, x) in cluster c."""
    a, x = (a, x) if a < x else (x, a)
    row = b.idx[c].get((a, x))
    if row is None:
        row = b.m[c]; b.m[c] += 1
        if row == len(b.A[c]):
            for arr in (b.A, b.B, b.s, b.n):
                arr[c] = np.concatenate([arr[c], np.zeros_like(arr[c])])
        b.idx[c][(a, x)] = row; b.A[c][row] = a; b.B[c][row] = x
        b.nb[c].setdefault(a, {})[x] = row; b.nb[c].setdefault(x, {})[a] = row
    b.s[c][row] = s_; b.n[c][row] = n_


def _sp_drop(b, i):
    """SparsePairMean: forget everything about primitive i (its configuration was rewritten)."""
    for c in range(b.C):
        for j, row in b.nb[c].pop(i, {}).items():
            b.nb[c][j].pop(i, None); b.idx[c].pop((min(i, j), max(i, j)), None)
            b.s[c][row] = 0.0; b.n[c][row] = 0.0       # dead row: estimate = p0, never preferred
        live = len(b.idx[c])
        if b.m[c] > 4096 and b.m[c] > 2 * live:         # compact dead rows (order kept, so choices are unchanged)
            rows = np.array(sorted(b.idx[c].values()), int)
            remap = {int(r): k for k, r in enumerate(rows)}
            size = max(64, 1 << int(np.ceil(np.log2(live + 1))))
            for arr in (b.A, b.B, b.s, b.n):
                new = np.zeros(size, arr[c].dtype); new[:live] = arr[c][rows]; arr[c] = new
            b.idx[c] = {k: remap[r] for k, r in b.idx[c].items()}
            b.nb[c] = {a: {j: remap[r] for j, r in d.items()} for a, d in b.nb[c].items()}
            b.m[c] = live


ES_PARAMS = dict(n_min=1, max_sub=3, rho=0.1, timeout=300, evict="crate", pq="equal", sub="config", k=1.0)      # frozen for H32 stage 2
# pq: parents by own LCB over all clusters (flat) or the same number from every cluster (equal: plain niching, no deficit signal)
# sub: a newcomer replaces its parent (parent) or any team member carrying the parent's configuration (config; the
#      system writes the configurations itself, so comparing them uses no hidden information)      # H32: trial mechanism (set by the experiment before a run)


# H33: self-regulation of the number of replacements per generation in mode colony (set by the experiment; None = off).
#   mode "g": rho -> 1 if the fast success estimate is below the best level the slow estimate ever reached by delta
#             (alarm) or exceeds the slow one by eps (still improving), otherwise rho *= 1 - eta; replacements = rho * 3R.
#             alarm=False: decay only (ablation).
#   mode "z": only primitives whose RECENT own record (forgetting factor gz per generation) is reliably below the rate
#             of their clusters (z < -z0) may be evicted.
CTL = dict(mode=None, tf=5000.0, ts=50000.0, eps=0.01, eta=0.1, delta=0.02, alarm=True, z0=2.0, gz=0.98)


class EvolvingBandit:
    """H20: sparse context bandit (fast layer) + a slow layer that rewrites configurations every `every` tasks.
      mode fixed      - no slow layer
      mode random     - least valuable primitives get random configurations
      mode individual - best primitives (LCB of own success, per cluster) are copied over the least valuable ones
      mode pattern    - best triples (LCB of the triple's success in a cluster) are copied as a unit
    A copy is mutated with probability pm (env.clone) and inherits half of the parent's pair statistics.
    Value of a primitive = its best per-cluster LCB. frac = share of the population rewritten per generation.
    nov > 0 (H22): score = LCB - nov * crowd for parents and victims alike, crowd(i) = share of the other
    primitives whose configuration differs from i's in at most one gene (pattern: mean over the triple)."""
    def __init__(self, mode, N, team, rng, env, frac=0.02, pm=0.1, every=100, seed=0, nov=0.0, beta=1.0, victim="noinherit"):
        self.mode, self.N, self.team, self.env, self.frac, self.pm, self.every = mode, N, team, env, frac, pm, every
        self.victim = victim      # H24: contrib | rate | nodrift | noinherit; default noinherit since H27 (contrib before)
        self.beta = beta          # H23: sharpness of the deficit quotas
        self.nov = nov            # H22: weight of the crowding penalty (MER Novelty) in parent and victim scores
        self.b = SparsePairMean(N, team, rng)
        self.rg = np.random.default_rng(seed + 8000)
        C = self.b.C
        self.is_ = np.zeros((C, N)); self.in_ = np.zeros((C, N))
        self.ctr = np.zeros(N); self.ih_s = np.zeros((C, N)); self.ih_n = np.zeros((C, N))      # H24
        self.cr = np.full(C, 0.5); self.cn = np.full(C, 1.0 / C)      # H23: recent success rate / task share per cluster
        self.tri = {}; self.epoch = np.zeros(N, int); self.t = 0; self.births = 0
        self.kids = "clone"       # H29: "random" = same quotas and eviction, but the freed slots get random configurations
        if mode == "colony_r":    #      (random search with the colony's eviction rule; replaces the defective mode "random")
            self.mode, self.kids = "colony", "random"
        # H32: trial of newcomers (probation + substitution for the parent). es = evolutionary search with trials and
        # eviction by own LCB; es_c = the same with the colony's eviction; colony_t = the colony plus the trial mechanism
        self.trial = mode in ("es", "es_c", "colony_t"); self.es = mode in ("es", "es_c"); self.es_evict = "lcb" if mode == "es" else "contrib"
        if mode == "colony_t":
            self.mode = "colony"
        if self.trial:
            self.tp = dict(ES_PARAMS); self.prob = np.zeros(N, bool); self.own_n = np.zeros(N); self.par = np.full(N, -1)
            self.born_t = np.zeros(N, int); self.kidq = {}; self.n_sub = 0; self.n_sub_tasks = 0; self.cfgq = {}; self.ref = {}; self.n_skip = 0
        self.ctl = dict(CTL) if CTL["mode"] and mode == "colony" else None      # H33
        self.rho, self.sf, self.ss, self.n_alarm, self.rho_log, self.sbest = 1.0, 0.5, 0.5, 0, [], 0.0
        if self.ctl is not None and self.ctl["mode"] == "z":
            self.zs = np.zeros((C, N)); self.zn = np.zeros((C, N))
        self.frozen = False       # H26: no further generations (the bandit keeps learning)
        self.lin = np.arange(N); self.trace_every = 0; self.trace = []      # H25: lineage ids and snapshots (measurement only)

    @property
    def state_size(self):
        return self.b.state_size

    def choose(self, t, x, avail=None):
        team = self.b.choose(t, x, avail)
        if self.trial and self.prob.any():           # H32: a newcomer on probation takes its parent's place in the team
            team = [int(i) for i in team]; done = 0
            for pos, p in enumerate(list(team)):
                if done >= self.tp["max_sub"]:
                    break
                for k in (self.kidq.get(p, ()) if self.tp["sub"] == "parent" else self.cfgq.get(self.env.genes[p].tobytes(), ())):
                    if self.prob[k] and k not in team and (avail is None or avail[k]):
                        team[pos] = k; done += 1
                        break
            if done < self.tp["max_sub"]:            # orphans and immigrants: in place of a random member
                free = np.where(self.prob & (self.par < 0) & (True if avail is None else avail))[0]
                free = [int(k) for k in free if k not in team]
                if free:
                    k = free[int(np.argmin(self.own_n[free]))]; pos = int(self.rg.integers(len(team)))
                    if not self.prob[team[pos]]:
                        team[pos] = k; done += 1
            self.n_sub += done; self.n_sub_tasks += done > 0
            team = np.array(team)
        return team

    def _end_trial(self, k):
        self.prob[k] = False; p = int(self.par[k])
        if p >= 0 and k in self.kidq.get(p, ()):
            self.kidq[p].remove(k)
        self.par[k] = -1
        key = self.ref.pop(k, None)
        if key is not None and k in self.cfgq.get(key, ()):
            self.cfgq[key].remove(k)
            if not self.cfgq[key]:
                del self.cfgq[key]

    def _start_trial(self, kid, parent):
        for k in self.kidq.pop(kid, ()):             # the slot was a parent itself: its kids on probation become orphans
            if self.tp["sub"] == "parent" or k not in self.ref:
                self.par[k] = -1
        if parent >= 0 and self.tp["sub"] == "config":
            key = self.ref_key; self.ref[kid] = key; self.cfgq.setdefault(key, []).append(kid)
        self.prob[kid] = True; self.own_n[kid] = 0; self.par[kid] = parent; self.born_t[kid] = self.t
        if parent >= 0:
            self.kidq.setdefault(parent, []).append(kid)

    def _gen_es(self):
        """H32: steady-state evolutionary search. Parents by own LCB (no quotas), kids without any inheritance,
        victims among the primitives that finished probation: lowest own LCB (es) or lowest own contribution (es_c)."""
        N, rg, env = self.N, self.rg, self.env
        for k in np.where(self.prob & (self.t - self.born_t > self.tp["timeout"]))[0]:
            self._end_trial(int(k))
        n_new = 3 * max(1, int(round(self.frac * N / 3))); n_imm = int(round(self.tp["rho"] * n_new))
        L = self._lcb(self.is_, self.in_) + 1e-9 * rg.random(self.is_.shape)
        parents, seen = [], set()
        if self.tp["pq"] == "equal":                 # plain niching: the same number of parents from every cluster
            order = [np.argsort(-L[c]) for c in range(L.shape[0])]; pos = [0] * L.shape[0]; c = 0; idle = 0
            while len(parents) < n_new - n_imm and idle < L.shape[0]:
                while pos[c] < N:
                    i = int(order[c][pos[c]]); pos[c] += 1
                    if i not in seen and self.in_[c, i] > 0 and not self.prob[i]:
                        seen.add(i); parents.append(i); idle = -1
                        break
                idle += 1; c = (c + 1) % L.shape[0]
        else:
            for flat in np.argsort(-L, axis=None):
                c, i = divmod(int(flat), N)
                if i not in seen and self.in_[c, i] > 0 and not self.prob[i]:
                    seen.add(i); parents.append(i)
                if len(parents) >= n_new - n_imm:
                    break
        if self.es_evict == "lcb" and self.tp["evict"] == "mean":      # own success rate over all uses, shrunk to the colony mean
            p0 = self.b.succ_sum / self.b.succ_n if self.b.succ_n else 0.5
            val = (self.is_.sum(0) + p0) / (self.in_.sum(0) + 1) + 1e-9 * rg.random(N)
        elif self.es_evict == "lcb" and self.tp["evict"] == "crate":   # own success per use relative to the rate of the cluster of each use
            val = (self.is_ - self.in_ * self.cr[:, None]).sum(0) / (self.in_.sum(0) + self.tp["k"]) + 1e-9 * rg.random(N)
        elif self.es_evict == "lcb":
            val = L.max(0)
        else:
            val = (self.is_ - self.in_ * self.cr[:, None]).sum(0) + 1e-9 * rg.random(N)
        val = val.copy(); val[self.prob] = np.inf; val[parents] = np.inf
        victims = [int(v) for v in np.argsort(val)[: len(parents) + n_imm] if np.isfinite(val[v])]
        self.n_skip += len(parents) + n_imm - len(victims)
        for j, kid in enumerate(victims):
            p = parents[j] if j < len(parents) else -1
            if p >= 0:
                self.ref_key = env.genes[p].tobytes()
                env.clone(p, kid, bool(rg.random() < self.pm), rg); self.lin[kid] = self.lin[p]
            else:
                env.randomize(kid, rg); self.lin[kid] = self.N + self.births
            self._rewrite(kid); self.ih_s[:, kid] = 0; self.ih_n[:, kid] = 0; self.ctr[kid] = 0
            self._start_trial(kid, p)

    def _lcb(self, s_, n_):
        p0 = self.b.succ_sum / self.b.succ_n if self.b.succ_n else 0.5
        return (s_ + p0) / (n_ + 1) - 1 / np.sqrt(n_ + 1)

    def update(self, team, succ):
        b = self.b; b.update(team, succ); c = b.ctx; self.t += 1
        team = [int(i) for i in team]
        self.is_[c, team] += succ; self.in_[c, team] += 1
        self.ctr[team] += succ - self.cr[c]
        self.cr[c] += 0.001 * (succ - self.cr[c]); self.cn *= 0.999; self.cn[c] += 0.001
        if self.ctl is not None and self.ctl["mode"] == "g":
            self.sf += (succ - self.sf) / min(self.t, self.ctl["tf"]); self.ss += (succ - self.ss) / min(self.t, self.ctl["ts"])
            if self.t >= self.ctl["ts"]:
                self.sbest = max(self.sbest, self.ss)
        elif self.ctl is not None:
            self.zs[c, team] += succ; self.zn[c, team] += 1
        if self.trial:
            self.own_n[team] += 1
            for k in team:
                if self.prob[k] and self.own_n[k] >= self.tp["n_min"]:
                    self._end_trial(k)
        if self.mode == "pattern" and len(team) == 3:
            key = (c, *sorted(team)); ep = tuple(int(self.epoch[i]) for i in key[1:])
            st = self.tri.get(key)
            if st is None or st[2] != ep:
                st = self.tri[key] = [0.0, 0.0, ep]
            st[0] += succ; st[1] += 1
        if self.mode != "fixed" and not self.frozen and self.t % self.every == 0:
            self._generation()
        if self.trace_every and self.t % self.trace_every == 0:
            self.trace.append((self.t, self.env.genes.astype(np.int8), self.lin.copy(), self.epoch.copy()))

    def _rewrite(self, kid):
        _sp_drop(self.b, kid); self.epoch[kid] += 1; self.births += 1
        self.is_[:, kid] = 0; self.in_[:, kid] = 0
        if self.ctl is not None and self.ctl["mode"] == "z":
            self.zs[:, kid] = 0; self.zn[:, kid] = 0

    def _generation(self):
        if self.es:
            return self._gen_es()
        N, rg, b, env = self.N, self.rg, self.b, self.env
        if self.trial:
            for k in np.where(self.prob & (self.t - self.born_t > self.tp["timeout"]))[0]:
                self._end_trial(int(k))
        R = max(1, int(round(self.frac * N / 3)))
        val = self._lcb(self.is_, self.in_).max(0) + 1e-9 * rg.random(N)
        crowd = None
        if self.nov:
            crowd = ((env.config_distance() <= 1).sum(1) - 1) / (N - 1)
            val = val - self.nov * crowd
        if self.mode == "random":
            # defective as random search (H29 review): a fresh slot has the lowest LCB and is rewritten again
            # until the bandit happens to try it; kept unchanged for H21-H22. Use mode "colony_r" instead.
            for kid in np.argsort(val)[: 3 * R]:
                env.randomize(int(kid), rg); self._rewrite(int(kid)); self.lin[kid] = self.N + self.births
            return
        if self.mode in ("individual", "colony", "colony_q", "colony_v"):
            L = self._lcb(self.is_, self.in_) + 1e-9 * rg.random(self.is_.shape)
            if crowd is not None:
                L = L - self.nov * crowd[None, :]
            parents, seen = [], set()
            if self.mode in ("colony", "colony_q"):      # copies go to the clusters where the colony fails most
                d = self.cn * (1 - self.cr) ** self.beta; d = 3 * R * d / d.sum()
                q = np.floor(d).astype(int)
                for c in np.argsort(-(d - q))[: 3 * R - q.sum()]:
                    q[c] += 1
                for c in np.argsort(-d):
                    got = 0
                    for i in np.argsort(-L[c]):
                        if got >= q[c]:
                            break
                        if int(i) not in seen and self.in_[c, i] > 0:
                            seen.add(int(i)); parents.append(int(i)); got += 1
            else:
                for flat in np.argsort(-L, axis=None):
                    c, i = divmod(int(flat), N)
                    if i not in seen and self.in_[c, i] > 0:
                        seen.add(i); parents.append(i)
                    if len(parents) >= 3 * R:
                        break
            if self.mode in ("colony", "colony_v"):      # victim: least contribution to the colony's successes
                val = (self.is_ - self.in_ * self.cr[:, None]).sum(0)
                if self.victim == "rate":            # per use: no usage weight
                    val = val / (self.in_.sum(0) + 1)
                elif self.victim == "nodrift":       # against the colony rate at the time of each task
                    val = self.ctr.copy()
                elif self.victim == "noinherit":     # own record only, without the inherited half
                    val = ((self.is_ - self.ih_s) - (self.in_ - self.ih_n) * self.cr[:, None]).sum(0)
                val = val + 1e-9 * rg.random(N)
            if self.ctl is not None:             # H33: fewer replacements than 3R
                if self.ctl["mode"] == "g":
                    d = self.sf - self.ss; low = self.ctl["alarm"] and self.sf < self.sbest - self.ctl["delta"]
                    if low or d > self.ctl["eps"]:
                        self.n_alarm += low; self.rho = 1.0
                    else:
                        self.rho *= 1 - self.ctl["eta"]
                    x = self.rho * len(parents); n = int(x) + int(rg.random() < x - int(x))
                else:
                    self.zs *= self.ctl["gz"]; self.zn *= self.ctl["gz"]
                    z = (self.zs - self.zn * self.cr[:, None]).sum(0) / np.sqrt((self.zn * (self.cr * (1 - self.cr))[:, None]).sum(0) + 1)
                    bad = z < -self.ctl["z0"]; bad[parents] = False
                    val[~bad] = np.inf; n = min(len(parents), int(bad.sum())); self.rho = n / max(1, len(parents))
                if n < len(parents):
                    parents = [parents[i] for i in sorted(rg.choice(len(parents), n, replace=False))]
                self.rho_log.append(n)
            val[parents] = np.inf
            if self.trial:                       # H32 (colony_t): primitives on probation cannot be evicted
                val[self.prob] = np.inf
            for p, kid in zip(parents, (int(v) for v in np.argsort(val)[: len(parents)])):
                if self.trial and not np.isfinite(val[kid]):      # nobody left to evict (everyone else is on probation)
                    self.n_skip += 1
                    continue
                if self.kids == "random":
                    env.randomize(kid, rg); self._rewrite(kid); self.lin[kid] = self.N + self.births
                    self.ih_s[:, kid] = 0; self.ih_n[:, kid] = 0; self.ctr[kid] = 0
                    continue
                env.clone(p, kid, bool(rg.random() < self.pm), rg); self._rewrite(kid); self.lin[kid] = self.lin[p]
                for c in range(b.C):
                    for j, row in list(b.nb[c].get(p, {}).items()):
                        if j != kid:
                            _sp_set(b, c, kid, j, b.s[c][row] / 2, b.n[c][row] / 2)
                self.is_[:, kid] = self.is_[:, p] / 2; self.in_[:, kid] = self.in_[:, p] / 2
                self.ih_s[:, kid] = self.is_[:, kid]; self.ih_n[:, kid] = self.in_[:, kid]; self.ctr[kid] = self.ctr[p] / 2
                if self.trial:
                    self.ref_key = env.genes[p].tobytes(); self._start_trial(kid, p)
            return
        pen = (lambda key: 0.0) if crowd is None else (lambda key: self.nov * float(crowd[list(key[1:])].mean()))
        cand = sorted(((self._lcb(st[0], st[1]) - pen(key), key) for key, st in self.tri.items()
                       if st[2] == tuple(int(self.epoch[i]) for i in key[1:])), key=lambda v: -v[0])
        parents, used = [], set()
        for _, key in cand:
            if len(parents) >= R:
                break
            if not used & set(key[1:]):
                used |= set(key[1:]); parents.append(key)
        if not parents:
            return
        val[list(used)] = np.inf
        victims = [int(v) for v in np.argsort(val)[: 3 * len(parents)]]
        for k, key in enumerate(parents):
            team, kids = key[1:], victims[3 * k: 3 * k + 3]
            st = self.tri[key]
            for p, kid in zip(team, kids):
                env.clone(p, kid, bool(rg.random() < self.pm), rg); self._rewrite(kid); self.lin[kid] = self.lin[p]
                self.is_[:, kid] = self.is_[:, p] / 2; self.in_[:, kid] = self.in_[:, p] / 2
            for c in range(b.C):                                  # the copy inherits the triple's pair statistics
                for u in range(3):
                    for v in range(u + 1, 3):
                        row = b.idx[c].get((min(team[u], team[v]), max(team[u], team[v])))
                        if row is not None and b.n[c][row] > 0:
                            _sp_set(b, c, kids[u], kids[v], b.s[c][row] / 2, b.n[c][row] / 2)
            ck = (key[0], *sorted(kids))
            self.tri[ck] = [st[0] / 2, st[1] / 2, tuple(int(self.epoch[i]) for i in ck[1:])]
        if len(self.tri) > 200_000:
            self.tri = {k: v for k, v in self.tri.items() if v[2] == tuple(int(self.epoch[i]) for i in k[1:])}


class NodeBandits:
    """Distributed setting (H19). kind:
      central_sparse - one SparsePairMean with all outcomes (needs a call to the centre per task; charged by the caller)
      node_sparse    - one per node over all primitives; a node learns from tasks that arrived at it or used its primitives
      shard_sparse   - one per node, local primitives only (no communication)
      node_random    - random free local team;  node_oracle - per role a free local primitive, else a remote one
    eps: probability of a decision that ignores the remote-call cost (exploration of remote partners)."""
    def __init__(self, kind, N, team, rng, env, eps=0.0):
        self.kind, self.N, self.team, self.rng, self.env, self.eps = kind, N, team, rng, env, eps
        P = 1 if kind == "central_sparse" else env.nodes
        self.b = [SparsePairMean(N, team, rng) for _ in range(P)] if kind.endswith("sparse") else []

    @property
    def state_size(self):
        return int(sum(b.state_size for b in self.b))

    def choose(self, t, x, avail=None):
        env, k = self.env, self.team
        h = int(env.home[t]); local = env.node == h
        av = np.ones(self.N, bool) if avail is None else avail
        self._h, self._x = h, x
        if self.kind == "node_random":
            free = np.where(av & local)[0]
            return self.rng.choice(free if len(free) >= k else np.where(av)[0], size=k, replace=False)
        if self.kind == "node_oracle":
            role, team = env.roles_at(t), []
            for j in range(k):
                for mask in (av & local & (role == j), av & (role == j), av & local, av):
                    cand = np.setdiff1d(np.where(mask)[0], team)
                    if len(cand):
                        break
                team.append(int(self.rng.choice(cand)))
            return np.array(team)
        if self.kind == "shard_sparse":
            return self.b[h].choose(t, x, av & local if (av & local).sum() >= k else av)
        pen = None if (self.eps and self.rng.random() < self.eps) else env.comm_cost * (~local)
        return self.b[0 if self.kind == "central_sparse" else h].choose(t, x, av, pen)

    def update(self, team, succ):
        if not self.b:
            return
        if self.kind == "central_sparse":
            return self.b[0].update(team, succ)
        self.b[self._h].update(team, succ)
        if self.kind == "node_sparse":
            for n in set(int(v) for v in self.env.node[np.asarray(team)]) - {self._h}:
                self.b[n].update(team, succ, self._x)


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
               alpha=0.4, gamma=0.995, lr=0.05, team_gamma=1.0, reset_at=None, role_k=None, eps=0.0, frac=0.02, pm=0.1, nov=0.0, beta=1.0, victim="noinherit", trace_every=0, hook=None):
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
    elif team.startswith("evo_"):
        tp = EvolvingBandit(team[4:], N, team_size, rng, env, frac, pm, seed=seed, nov=nov, beta=beta, victim=victim)
        tp.trace_every = trace_every
    elif team in ("central_sparse", "node_sparse", "shard_sparse", "node_random", "node_oracle"):
        tp = NodeBandits(team, N, team_size, rng, env, eps)
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
        if hook is not None:          # measurement / intervention point for experiments (H26)
            hook(t, tp, env, members, m, succ, q)
    return dict(R=R, q=Q, cover=COV, policy=tp, env=env)


def summary(out, T):
    R, cov = out["R"], out["cover"]
    has = not np.all(np.isnan(cov))
    return dict(pre=float(R[T // 2 - 1000:T // 2].mean()), post=float(R[-1000:].mean()),
                cover_start=float(np.nanmean(cov[:1000])) if has else None,
                cover_end=float(np.nanmean(cov[-2000:])) if has else None,
                q_end=float(out["q"][-2000:].mean()))
