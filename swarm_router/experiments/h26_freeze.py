"""H26 stages 1-2: freezing the population of C at 400k/600k/800k and a per-task account of where team quality
is lost (notes/hypotheses.md). Usage: python h26_freeze.py VARIANT SEED  (VARIANT: C | NI | A | any key of VAR).
The C run forks at the freeze points: the child continues with the population frozen, the parent goes on evolving,
so the frozen runs share the exact prefix of C. Diagnostics read hidden simulator state and never feed the algorithm."""
import os, sys, time

import numpy as np

from common import ROOT
from swarm import EnvConfig
import swarm.baselines as sb
from swarm.baselines import run_router

T = int(os.environ.get("H26_T", 1_600_000)); SHIFT = int(os.environ.get("H26_SHIFT", 100_000))
FORKS = tuple(int(x) for x in os.environ.get("H26_FORKS", "400000,600000,800000").split(",") if x)
N, BUSY, GL, BLK = int(os.environ.get("H26_N", 640)), int(os.environ.get("H26_BUSY", 48)), int(os.environ.get("H26_L", 5)), T // 160      # H28: other environments
VAR = {"C": dict(team="evo_colony", frac=0.05, pm=0.1, beta=1.0, victim="contrib"), "NI": dict(team="evo_colony", frac=0.05, pm=0.1, beta=1.0, victim="noinherit"),
       "A": dict(team="evo_fixed"), "R": dict(team="evo_random", frac=0.05), "RC": dict(team="evo_colony_r", frac=0.05, pm=0.1, beta=1.0),
       "ES": dict(team="evo_es", frac=0.05, pm=0.1), "EC": dict(team="evo_es_c", frac=0.05, pm=0.1), "NT": dict(team="evo_colony_t", frac=0.05, pm=0.1, beta=1.0),      # H32
       "N4": dict(team="evo_colony", frac=0.0125, pm=0.1, beta=1.0),      # H33: constant reduced intensity (9 replacements per generation)
       "PN": dict(team="evo_pattern", frac=0.05, pm=0.1, nov=2.0)}      # H29: baselines with search
SWITCH = os.environ.get("H26_SWITCH", "") == "1"      # H27: also branch to the noinherit eviction rule at the fork points
KINDS = os.environ.get("H26_KINDS", "FS" if SWITCH else "F")      # branch kinds at the fork points (H30: F, O, G)
VICT = os.environ.get("H26_VICT", "") == "1"                      # H31: who is evicted under each rule, on the same states
BASE = os.environ.get("H26_BASE", "C")                           # the variant that is branched
if os.environ.get("H32_PARAMS"):                                 # H32: n_min,max_sub,rho,timeout,evict of the trial mechanism
    a_ = os.environ["H32_PARAMS"].split(","); sb.ES_PARAMS.update(n_min=int(a_[0]), max_sub=int(a_[1]), rho=float(a_[2]), timeout=int(a_[3]), evict=a_[4])
    if len(a_) > 5:
        sb.ES_PARAMS.update(pq=a_[5], sub=a_[6])
    if len(a_) > 7:
        sb.ES_PARAMS.update(k=float(a_[7]))
if os.environ.get("H33_CTL"):                                    # H33: g,eps,eta[,noalarm] or z,z0,gz
    a_ = os.environ["H33_CTL"].split(",")
    sb.CTL.update(mode="g", eps=float(a_[1]), eta=float(a_[2]), alarm=len(a_) < 4) if a_[0] == "g" else sb.CTL.update(mode="z", z0=float(a_[1]), gz=float(a_[2]))
if os.environ.get("H33_SET"):                                    # H33 main test: parameters only, the controllers are switched on at a fork
    a_ = os.environ["H33_SET"].split(","); sb.CTL.update(eps=float(a_[0]), eta=float(a_[1]), z0=float(a_[2]), gz=float(a_[3]))
TSH = [int(x) for x in os.environ.get("H33_TSHIFT", "0,0").split(",")] + [0]      # H33: at task TSH[0] the targets of TSH[1] niches change:
#      redrawn (TSH[2] = 0) or TSH[2] genes of each replaced by other values
KMAP = dict(x.split(":") for x in os.environ.get("H33_KMAP", "").split(",") if x)      # H33: fork time -> kinds, e.g. 400000:HAZQ,600000:F
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
        self.ev = np.zeros((160, 6)); self.subs = np.zeros(160); self.uses = None; self.isperf = None; self.sub0 = 0      # H32: search accounting
        self.vict = np.zeros((160, 2, 4)); self.pop = np.zeros((160, 4)); self.prot = np.zeros((160, 4)); self.vn = np.zeros(160)
        self.vuse = np.zeros((160, 2)); self.vinh = np.zeros((160, 2))

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
                self.uses = np.zeros(len(self.ep)); self.isperf = (env.genes[:, None, None, :] == env.targets[None]).all(-1).any((1, 2))
            self.uses[members] += 1
            ch = tp.epoch != self.ep
            if ch.any():                                                 # H32: births, evictions, fate of perfect configurations (hidden; diagnostics)
                pn = (env.genes[ch][:, None, None, :] == env.targets[None]).all(-1).any((1, 2)); po = self.isperf[ch]; u = self.uses[ch]
                self.ev[b] += (ch.sum(), pn.sum(), (u == 0).sum(), u.sum(), po.sum(), (po & (t - self.born[ch] >= 5000)).sum())
                self.isperf[ch] = pn; self.uses[ch] = 0
                self.born[ch] = t; self.ep = tp.epoch.copy()
            if getattr(tp, "trial", False):
                self.subs[b] += tp.n_sub - self.sub0; self.sub0 = tp.n_sub
            i = 0 if self.cum[cc].argmax() == k else 1; j = 2 if (t - self.born[members] < 5000).any() else 3
            self.split[b, i] += (rl, 1); self.split[b, j] += (rl, 1)
        if TSH[0] and t + 1 == TSH[0]:                                   # H33: environment event, not known to the algorithm
            rg = np.random.default_rng(SEED + 3300); K, J, L = env.targets.shape
            for f in rg.choice(K * J, size=TSH[1], replace=False):
                old = env.targets[f // J, f % J].copy()
                if TSH[2]:
                    for g in rg.choice(L, size=TSH[2], replace=False):
                        env.targets[f // J, f % J, g] = (old[g] + rg.integers(1, env.A)) % env.A
                while (env.targets[f // J, f % J] == old).all():
                    env.targets[f // J, f % J] = rg.integers(env.A, size=L)
            env._recompute(np.arange(env.N))
            if self.isperf is not None:
                self.isperf = (env.genes[:, None, None, :] == env.targets[None]).all(-1).any((1, 2))
        if VICT and hasattr(tp, "ih_s") and tp.t % tp.every == tp.every - 1:      # H31: state one task before a generation
            eq = (env.genes[:, None, None, :] == env.targets[None]).all(-1); cnt = eq.sum(0); big = np.unravel_index(cnt.argmax(), cnt.shape)
            perf = eq.any((1, 2)); nc = np.where(eq, cnt[None], 0).max((1, 2))      # copies in the primitive's own niche
            cat = np.where(~perf, 3, np.where(eq[:, big[0], big[1]], 2, np.where(nc < 16, 0, 1)))      # scarce | abundant | dominant | imperfect
            own_n = (tp.in_ - tp.ih_n).sum(0); inh = (tp.ih_s - tp.ih_n * tp.cr[:, None]).sum(0)
            v0 = (tp.is_ - tp.in_ * tp.cr[:, None]).sum(0); v1 = v0 - inh; R3 = 3 * max(1, int(round(tp.frac * tp.N / 3)))
            self.pop[b] += np.bincount(cat, minlength=4); self.prot[b] += np.bincount(cat[(own_n < 0.5) & (inh > 0)], minlength=4); self.vn[b] += 1
            for r, val in enumerate((v0, v1)):
                vi = np.argsort(val + 1e-9 * np.arange(len(val)))[:R3]
                self.vict[b, r] += np.bincount(cat[vi], minlength=4); self.vuse[b, r] += own_n[vi].mean(); self.vinh[b, r] += inh[vi].mean()
        if hasattr(tp, "in_"):                                           # observable proxy: free primitives with above-average record
            cc = tp.b.ctx; n = tp.in_[cc]; good = (n >= 5) & (tp.is_[cc] > tp.cr[cc] * n)
            a["cap_obs"][b] += (good & av).sum(); a["good_obs"][b] += good.sum()
            if self.tag == BASE and tp.t in FORKS:                       # branch: the child continues with a frozen population
                for kind in KMAP.get(str(tp.t), KINDS):
                    pid = os.fork()
                    if pid == 0:
                        if kind in "HA":                                 # H33: global controller switched on (A: without the alarm)
                            tp.ctl = dict(sb.CTL, mode="g", alarm=kind == "H"); tp.sf = tp.ss = float((tp.cn * tp.cr).sum() / tp.cn.sum()); tp.rho = 1.0
                        if kind == "Z":                                  # H33: local controller switched on
                            tp.ctl = dict(sb.CTL, mode="z"); tp.zs = np.zeros_like(tp.is_); tp.zn = np.zeros_like(tp.in_)
                        if kind == "Q":                                  # H33: constant reduced intensity
                            tp.frac = 0.0125
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
    tags = [v] + ([f"{k}{f // 1000}" for f in FORKS for k in KMAP.get(str(f), KINDS)] if v == BASE else [])
    return all(os.path.exists(os.path.join(OUT, f"{g}_{seed}.npz")) for g in tags)


if __name__ == "__main__":
    v, seed = sys.argv[1], int(sys.argv[2]); os.makedirs(OUT, exist_ok=True); SEED = seed
    if done(v, seed):
        sys.exit(0)
    d = Diag(v); targets0 = np.random.default_rng(seed + 2901).integers(4, size=(3, 3, GL))      # targets before a possible H33 shift
    o = run_router(EnvConfig("config", T, busy=BUSY, genes_L=GL, genes_pow=1.0, shift_at=SHIFT), seed, N=N, model="linucb",
                   trace_every=T // 80, hook=d, **VAR[v])
    tp, env = o["policy"], o["env"]; f = os.path.join(OUT, f"{d.tag}_{seed}")
    np.savez_compressed(f + ".tmp.npz", R=o["R"].reshape(160, -1).mean(1), q=o["q"].reshape(160, -1).mean(1), blk=BLK,
                        t=np.array([s[0] for s in tp.trace]), genes=np.array([s[1] for s in tp.trace]), lin=np.array([s[2] for s in tp.trace]),
                        epoch=np.array([s[3] for s in tp.trace]), targets=env.targets, genes0=np.random.default_rng(seed + 2900).integers(env.A, size=(N, env.L)),
                        cat_n=d.cat_n, cat_s=d.cat_s, role=d.role / BLK, conf=d.conf, split=d.split, ev=d.ev, subs=d.subs, rho_log=np.array(getattr(tp, "rho_log", []), float), n_alarm=getattr(tp, "n_alarm", 0), targets0=targets0, n_skip=getattr(tp, "n_skip", 0), alive_perf_age=np.where(d.isperf, T - d.born, -1) if d.isperf is not None else np.zeros(1), vict=d.vict, pop=d.pop, prot=d.prot, vn=d.vn, vuse=d.vuse, vinh=d.vinh, wall=time.time() - d.t0, state=tp.state_size, births=tp.births,
                        **{k: x / BLK for k, x in d.a.items()})
    os.replace(f + ".tmp.npz", f + ".npz")
    print(f"{d.tag} seed {seed}: {(time.time() - d.t0) / 60:.1f} min", flush=True)
    for pid in d.kids:
        os.waitpid(pid, 0)
    os._exit(0)
