"""H14: recruitment that uses confirmed links: {mean,min} x {fixed,evidence} x {no inheritance, team selection}.
(notes/hypotheses.md). Baselines mean+fixed come from the H12/H13 result files. Checkpointed."""
import argparse, os

from common import ROOT, base_args, save
from h13_team_selection import CELLS, collect, run
from swarm import EnvConfig, SwarmConfig

ap = base_args(argparse.ArgumentParser()); ap.add_argument("--Ns", default="40,160,640,2560")
a = ap.parse_args()
seeds = list(range(a.seed0, a.seed0 + a.seeds))
specs, keys = [], []
for cname, ck in CELLS.items():
    for N in [int(x) for x in a.Ns.split(",")]:
        env = EnvConfig(T=a.T, **ck)
        for inh in ("none", "team"):
            for agg_ in ("mean", "min"):
                for sw in ("fixed", "evidence"):
                    if agg_ == "mean" and sw == "fixed":
                        continue
                    cfg = SwarmConfig(N=N, inherit=inh, inherit_frac=0.02, recruit_agg=agg_, sim_weight=sw)
                    for sd in seeds:
                        specs.append((env, cfg, sd)); keys.append((cname, N, f"{inh}|{agg_}|{sw}", sd))
table = collect(run(specs, keys, os.path.join(ROOT, ".cache", f"h14_T{a.T}.jsonl"), a.workers))
for c, byN in table.items():
    print(f"\n## {c}")
    for N, byV in sorted(byN.items(), key=lambda x: int(x[0])):
        print(f"N={N:>5}: " + "  ".join(f"{v}={e['post']:.3f}" for v, e in byV.items()))
save("h14_recruit", dict(T=a.T, seeds=seeds, table=table))
