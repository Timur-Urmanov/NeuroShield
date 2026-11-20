"""H2 (sim weight s), H3 (team size x member cost), H4 (role fractions, N), H5 (hebb bias x sleep prune)."""
import argparse

from common import base_args, paired, run_variants, save, table
from swarm import EnvConfig, SwarmConfig

ap = base_args(argparse.ArgumentParser()); ap.add_argument("--only", default="h2,h3,h4,h5"); a = ap.parse_args()
seeds = list(range(a.seed0, a.seed0 + a.seeds))
only = a.only.split(",")
roles = EnvConfig("roles", a.T)
out = {}

if "h2" in only:
    res = run_variants({f"s={s}": ("swarm", roles, SwarmConfig(sim_w=s)) for s in (0.0, 0.25, 0.5, 0.75, 1.0)}, seeds, a.workers)
    print("## H2\n" + table(res)); best = max(res, key=lambda k: res[k]["post"])
    cmp = [paired(res, best, "s=0.0"), paired(res, best, "s=1.0")] + [paired(res, best, k) for k in res if k != best]
    for c in cmp: print(c)
    out["h2"] = dict(results=res, best=best, comparisons=cmp)

if "h3" in only:
    for mc in (0.0, 0.05):
        env = EnvConfig("roles", a.T, member_cost=mc)
        res = run_variants({f"team={k}": ("swarm", env, SwarmConfig(team=k)) for k in (2, 3, 4, 5)}, seeds, a.workers)
        print(f"## H3 member_cost={mc}\n" + table(res))
        cmp = [paired(res, "team=3", "team=2"), paired(res, "team=3", "team=5"), paired(res, "team=4", "team=3")]
        for c in cmp: print(c)
        out[f"h3_mc{mc}"] = dict(results=res, comparisons=cmp)

if "h4" in only:
    conds = {"frac 1/3": dict(N=40, f=None), "frac .5/.25/.25": dict(N=40, f=(0.5, 0.25, 0.25)),
             "frac .6/.2/.2": dict(N=40, f=(0.6, 0.2, 0.2)), "frac .8/.1/.1": dict(N=40, f=(0.8, 0.1, 0.1)),
             "N=20": dict(N=20, f=None), "N=80": dict(N=80, f=None)}
    out["h4"] = {}
    for cname, c in conds.items():
        env = EnvConfig("roles", a.T, role_fractions=c["f"])
        V = {"swarm W_reward": ("swarm", env, SwarmConfig(N=c["N"])),
             "swarm sim-only (no W)": ("swarm", env, SwarmConfig(N=c["N"], plastic_W=False)),
             "linucb + pair_ucb c=0.03": ("router", env, dict(N=c["N"], model="linucb", team="pair_ucb", c=0.03)),
             "linucb + pair_ucb c=0.1": ("router", env, dict(N=c["N"], model="linucb", team="pair_ucb", c=0.1)),
             "linucb + oracle team": ("router", env, dict(N=c["N"], model="linucb", team="oracle"))}
        res = run_variants(V, seeds, a.workers)
        print(f"## H4 {cname}\n" + table(res, ("post", "cover_end")))
        bb = max(["linucb + pair_ucb c=0.03", "linucb + pair_ucb c=0.1"], key=lambda k: res[k]["post"])
        cmp = [paired(res, "swarm W_reward", "swarm sim-only (no W)"), paired(res, "swarm W_reward", bb)]
        for x in cmp: print(x)
        out["h4"][cname] = dict(results=res, comparisons=cmp)

if "h5" in only:
    V = {f"bias={b}, prune={p}": ("swarm", roles, SwarmConfig(fatigue=True, sleep=True, hebb_bias=b, sleep_prune=p))
         for b in (0.0, 0.3) for p in (True, False)}
    V["no fatigue, no sleep (ref)"] = ("swarm", roles, SwarmConfig())
    V["fatigue, no sleep"] = ("swarm", roles, SwarmConfig(fatigue=True))
    res = run_variants(V, seeds, a.workers)
    print("## H5\n" + table(res, ("post", "cover_end")))
    cmp = [paired(res, "bias=0.0, prune=False", "bias=0.0, prune=True"),
           paired(res, "bias=0.3, prune=False", "bias=0.3, prune=True"),
           paired(res, "bias=0.0, prune=False", "bias=0.3, prune=False"),
           paired(res, "bias=0.0, prune=True", "bias=0.3, prune=True"),
           paired(res, "bias=0.0, prune=False", "bias=0.0, prune=True", "cover_end"),
           paired(res, "bias=0.0, prune=True", "fatigue, no sleep"),
           paired(res, "bias=0.0, prune=False", "fatigue, no sleep")]
    for c in cmp: print(c)
    out["h5"] = dict(results=res, comparisons=cmp)

save("h2_h5_ablations" + ("" if len(only) == 4 else "_" + "_".join(only)), dict(T=a.T, seeds=seeds, **out))
