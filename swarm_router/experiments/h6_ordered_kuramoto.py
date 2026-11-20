"""H6: is phase (Kuramoto) useful for ordering steps plan -> execute -> verify? (notes/hypotheses.md)"""
import argparse

from common import base_args, paired, run_variants, save, table
from swarm import EnvConfig, SwarmConfig

ap = base_args(argparse.ArgumentParser()); a = ap.parse_args()
seeds = list(range(a.seed0, a.seed0 + a.seeds))
env = EnvConfig("ordered", a.T)
B = dict(phase_filter=False)
V = {
    "recruit": SwarmConfig(order="recruit", **B),
    "random": SwarmConfig(order="random", **B),
    "position": SwarmConfig(order="position", **B),
    "phase_static": SwarmConfig(order="phase", kuramoto=False, **B),
    "phase_kuramoto": SwarmConfig(order="phase", kuramoto=True, **B),
    "phase_kuramoto_filter": SwarmConfig(order="phase", kuramoto=True, phase_filter=True),
    "oracle": SwarmConfig(order="oracle", **B),
    # post-hoc (added after the pre-registered run): linear readout around the team mean phase
    "posthoc phase_mean_static": SwarmConfig(order="phase_mean", kuramoto=False, **B),
    "posthoc phase_mean_kuramoto": SwarmConfig(order="phase_mean", kuramoto=True, **B),
}
res = run_variants({k: ("swarm", env, v) for k, v in V.items()}, seeds, a.workers)
res["linucb + pair_ucb c=0.03 (order = selection order)"] = run_variants(
    {"x": ("router", env, dict(model="linucb", team="pair_ucb", c=0.03))}, seeds, a.workers)["x"]
for c in (0.03, 0.1, 0.3):
    res[f"linucb + seq_ucb c={c} (order-aware)"] = run_variants(
        {"x": ("router", env, dict(model="linucb", team="seq_ucb", c=c))}, seeds, a.workers)["x"]
res["linucb + oracle team"] = run_variants({"x": ("router", env, dict(model="linucb", team="oracle"))}, seeds, a.workers)["x"]
print(table(res, ("post", "cover_end", "q_end")))
cmp = [paired(res, "phase_kuramoto", "position"), paired(res, "phase_kuramoto", "phase_static"),
       paired(res, "phase_kuramoto_filter", "position"), paired(res, "position", "recruit"),
       paired(res, "position", "random"), paired(res, "oracle", "position"),
       paired(res, "posthoc phase_mean_kuramoto", "position"),
       paired(res, "posthoc phase_mean_kuramoto", "posthoc phase_mean_static")]
best_seq = max([k for k in res if "seq_ucb" in k], key=lambda k: res[k]["post"])
cmp.append(paired(res, "position", best_seq))
for c in cmp:
    print(c)
save("h6_ordered_kuramoto", dict(T=a.T, seeds=seeds, results=res, comparisons=cmp))
