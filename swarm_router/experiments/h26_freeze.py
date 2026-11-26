"""H26 stages 1-2: freezing the population of C at 400k/600k/800k and a per-task account of where team quality
is lost (notes/hypotheses.md). Usage: python h26_freeze.py VARIANT SEED  (VARIANT: C | NI | A | any key of VAR).
The C run forks at the freeze points: the child continues with the population frozen, the parent goes on evolving,
so the frozen runs share the exact prefix of C. Diagnostics read hidden simulator state and never feed the algorithm."""
import os, sys, time

import numpy as np

from common import ROOT
from swarm import EnvConfig
from swarm.baselines import run_router

T = int(os.environ.get("H26_T", 1_600_000)); SHIFT = int(os.environ.get("H26_SHIFT", 100_000))
FORKS = tuple(int(x) for x in os.environ.get("H26_FORKS", "400000,600000,800000").split(",") if x)
N, BUSY, GL, BLK = int(os.environ.get("H26_N", 640)), int(os.environ.get("H26_BUSY", 48)), int(os.environ.get("H26_L", 5)), T // 160      # H28: other environments
VAR = {"C": dict(team="evo_colony", frac=0.05, pm=0.1, beta=1.0, victim="contrib"), "NI": dict(team="evo_colony", frac=0.05, pm=0.1, beta=1.0, victim="noinherit"),
       "A": dict(team="evo_fixed"), "R": dict(team="evo_random", frac=0.05), "RC": dict(team="evo_colony_r", frac=0.05, pm=0.1, beta=1.0), "PN": dict(team="evo_pattern", frac=0.05, pm=0.1, nov=2.0)}      # H29: baselines with search
SWITCH = os.environ.get("H26_SWITCH", "") == "1"      # H27: also branch to the noinherit eviction rule at the fork points
KINDS = os.environ.get("H26_KINDS", "FS" if SWITCH else "F")      # branch kinds at the fork points (H30: F, O, G)
BASE = os.environ.get("H26_BASE", "C")                           # the variant that is branched
OUT = os.path.join(ROOT, ".cache", os.environ.get("H26_DIR", "h26"))
KEYS = ("succ", "q_pop", "q_avail", "q_team", "er", "er_best", "min_free", "cap_obs", "good_obs")


class Diag:
    def __init__(self, tag):
        self.tag, self.kids, self.t0 = tag, [], time.time()
        self.a = {k: np.zeros(160) for k in KEYS}
        self.cat_n = np.zeros((160, 4)); self.cat_s = np.zeros((160, 4)); self.role = np.zeros((160, 4))
        self.conf = np.zeros((160, 3, 3)); self.cum = np.zeros((3, 3))        # H30: cluster x hidden type
        self.split = np.zeros((160, 4, 2))                                   # routing loss and count: cluster ok | wrong | young member | no young member
        self.ep = None; self.born = None

    def __call__(self, t, tp, env, members, m, succ, q):
        b = t // BLK; k = env.ks[t]; sk = env.skill[k]; a = self.a
        av = env.busy_until < t; av[members] = True                      # availability at decision time
        bp = sk.max(0); ba = np.where(av[:, None], sk, 0.0).max(0); tm = sk[members].max(0)
        perf = sk >= env.strong - 1e-9; ex = perf.any(0); free = (perf & av[:, None]).sum(0); got = tm >= env.strong - 1e-9
        acc = env.acc(t)[:, k]; ev = acc * q - env.lam * env.cost
        a["succ"][b] += succ; a["q_pop"][b] += bp.prod(); a["q_avail"][b] += ba.prod(); a["q_team"][b] += q
        a["er"][b] += ev[m]; a["er_best"][b] += ev.max(); a["min_free"][b] += free.min()
        c = 0 if not ex.all() else 1 if (free == 0).any() else 2 if not got.all() else 3      # absent | busy | routing | full team
        self.cat_n[b, c] += 1; self.cat_s[b, c] += succ
        self.role[b] += [(~ex).mean(), (ex & (free == 0)).mean(), ((free > 0) & ~got).mean(), got.mean()]
        if hasattr(tp, "b"):                                             # H30: where the routing loss sits (hidden type and ages; diagnostics only)
            cc = tp.b.ctx; self.conf[b, cc, k] += 1; self.cum[cc, k] += 1; rl = ba.prod() - q
            if self.ep is None:
                self.ep = tp.epoch.copy(); self.born = np.full(len(self.ep), -10 ** 9)
            ch = tp.epoch != self.ep
            if ch.any():
                self.born[ch] = t; self.ep = tp.epoch.copy()
            i = 0 if self.cum[cc].argmax() == k else 1; j = 2 if (t - self.born[members] < 5000).any() else 3
            self.split[b, i] += (rl, 1); self.split[b, j] += (rl, 1)
        if hasattr(tp, "in_"):                                           # observable proxy: free primitives with above-average record
            cc = tp.b.ctx; n = tp.in_[cc]; good = (n >= 5) & (tp.is_[cc] > tp.cr[cc] * n)
            a["cap_obs"][b] += (good & av).sum(); a["good_obs"][b] += good.sum()
            if self.tag == BASE and tp.t in FORKS:                       # branch: the child continues with a frozen population
                for kind in KINDS:
                    pid = os.fork()
                    if pid == 0:
                        if kind in "FG":
                            tp.frozen = True
                        if kind == "S":
                            tp.victim = "noinherit"                      # H27: evolution continues under the other eviction rule
                        if kind in "OG":                                 # H30 (diagnostic, hidden information): clusters replaced by the true task type
                            tp.b.clusters = TrueType(env, t + 1, self.cum.argmax(0))
                        self.tag = f"{kind}{tp.t // 1000}"; self.kids = []
                        break
                    self.kids.append(pid)


class TrueType:
    """Stands in for OnlineClusters: returns the cluster that the hidden type of the current task mostly fell into."""
    def __init__(self, env, t, cluster_of_type):
        self.env, self.t, self.map = env, t, [int(c) for c in cluster_of_type]

    def assign(self, x):
        c = self.map[self.env.ks[self.t]]; self.t += 1
        return c


def done(v, seed):
    tags = [v] + ([f"{k}{f // 1000}" for f in FORKS for k in KINDS] if v == BASE else [])
    return all(os.path.exists(os.path.join(OUT, f"{g}_{seed}.npz")) for g in tags)


if __name__ == "__main__":
    v, seed = sys.argv[1], int(sys.argv[2]); os.makedirs(OUT, exist_ok=True)
    if done(v, seed):
        sys.exit(0)
    d = Diag(v)
    o = run_router(EnvConfig("config", T, busy=BUSY, genes_L=GL, genes_pow=1.0, shift_at=SHIFT), seed, N=N, model="linucb",
                   trace_every=T // 80, hook=d, **VAR[v])
    tp, env = o["policy"], o["env"]; f = os.path.join(OUT, f"{d.tag}_{seed}")
    np.savez_compressed(f + ".tmp.npz", R=o["R"].reshape(160, -1).mean(1), q=o["q"].reshape(160, -1).mean(1), blk=BLK,
                        t=np.array([s[0] for s in tp.trace]), genes=np.array([s[1] for s in tp.trace]), lin=np.array([s[2] for s in tp.trace]),
                        epoch=np.array([s[3] for s in tp.trace]), targets=env.targets, genes0=np.random.default_rng(seed + 2900).integers(env.A, size=(N, env.L)),
                        cat_n=d.cat_n, cat_s=d.cat_s, role=d.role / BLK, conf=d.conf, split=d.split, wall=time.time() - d.t0, state=tp.state_size, births=tp.births,
                        **{k: x / BLK for k, x in d.a.items()})
    os.replace(f + ".tmp.npz", f + ".npz")
    print(f"{d.tag} seed {seed}: {(time.time() - d.t0) / 60:.1f} min", flush=True)
    for pid in d.kids:
        os.waitpid(pid, 0)
    os._exit(0)
