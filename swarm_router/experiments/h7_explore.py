"""
NOTE: pinned to recruit_rule="independent", w_prior="w0" (defaults before H9d).

NOTE: pinned to w_rule="hebb_r" (the default when this experiment was run; default is mean_succ since H8c).
H7: exploration bonus in recruitment (notes/hypotheses.md).
Step 1 (--stage dev): sweep c on dev seeds 100-104 in the base roles cell.
Step 2 (--stage test --c_ucb X --c_noise Y): chosen c on seeds 0-4 in all H1b cells + ordered env."""
import argparse, json, os

from common import RESULTS, base_args, paired, run_variants, save, table
from swarm import EnvConfig, SwarmConfig

ap = base_args(argparse.ArgumentParser())
ap.add_argument("--stage", choices=["dev", "test"], required=True)
ap.add_argument("--c_ucb", type=float); ap.add_argument("--c_noise", type=float)
a = ap.parse_args()
CS = (0.02, 0.05, 0.1, 0.2, 0.4)

if a.stage == "dev":
    seeds = list(range(100, 105))
    env = EnvConfig("roles", a.T)
    V = {"no bonus": ("swarm", env, SwarmConfig(recruit_rule="independent", w_prior="w0", w_rule="hebb_r"))}
    for kind in ("ucb", "noise"):
        for c in CS:
            V[f"{kind} c={c}"] = ("swarm", env, SwarmConfig(recruit_rule="independent", w_prior="w0", w_rule="hebb_r", explore=kind, explore_c=c))
    res = run_variants(V, seeds, a.workers)
    print(table(res))
    best = {k: max([n for n in res if n.startswith(k)], key=lambda n: res[n]["post"]) for k in ("ucb", "noise")}
    print("best on dev:", best)
    save("h7_explore_dev", dict(T=a.T, seeds=seeds, results=res, best=best))
else:
    seeds = list(range(a.seed0, a.seed0 + a.seeds))
    h1b = json.load(open(os.path.join(RESULTS, "h1b_context_capacity.json")))["cells"]
    out = {}
    for rbt in (False, True):
        for busy in (0, 8):
            cell = f"role_by_type={rbt}, busy={busy}"
            env = EnvConfig("roles", a.T, role_by_type=rbt, busy=busy)
            V = {"no bonus": ("swarm", env, SwarmConfig(recruit_rule="independent", w_prior="w0", w_rule="hebb_r")),
                 "ucb": ("swarm", env, SwarmConfig(recruit_rule="independent", w_prior="w0", w_rule="hebb_r", explore="ucb", explore_c=a.c_ucb)),
                 "noise": ("swarm", env, SwarmConfig(recruit_rule="independent", w_prior="w0", w_rule="hebb_r", explore="noise", explore_c=a.c_noise))}
            res = run_variants(V, seeds, a.workers)
            # bandit numbers from H1b (same seeds, same streams)
            for k, v in h1b[cell]["results"].items():
                if "_ucb" in k:
                    res[k] = v
            best = max([k for k in res if "_ucb" in k], key=lambda k: res[k]["post"])
            pair_best = max([k for k in res if "linucb + pair_ucb" in k], key=lambda k: res[k]["post"])
            cmp = [paired(res, "ucb", "no bonus"), paired(res, "noise", "no bonus"), paired(res, "ucb", "noise"),
                   paired(res, "ucb", best), paired(res, "ucb", pair_best)]
            print(f"\n## {cell}\n" + table({k: res[k] for k in ("no bonus", "ucb", "noise", best)}))
            for c in cmp:
                print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})
            out[cell] = dict(results=res, best_bandit=best, comparisons=cmp)
    env = EnvConfig("ordered", a.T)
    V = {"position, no bonus": ("swarm", env, SwarmConfig(recruit_rule="independent", w_prior="w0", w_rule="hebb_r", order="position")),
         "position + ucb": ("swarm", env, SwarmConfig(recruit_rule="independent", w_prior="w0", w_rule="hebb_r", order="position", explore="ucb", explore_c=a.c_ucb)),
         "linucb + seq_ucb c=0.1": ("router", env, dict(model="linucb", team="seq_ucb", c=0.1))}
    res = run_variants(V, seeds, a.workers)
    cmp = [paired(res, "position + ucb", "position, no bonus"), paired(res, "position + ucb", "linucb + seq_ucb c=0.1")]
    print("\n## ordered\n" + table(res))
    for c in cmp:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()})
    out["ordered"] = dict(results=res, comparisons=cmp)
    save("h7_explore_test", dict(T=a.T, seeds=seeds, c_ucb=a.c_ucb, c_noise=a.c_noise, cells=out))
