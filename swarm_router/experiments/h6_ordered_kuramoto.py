"""
NOTE: pinned to recruit_rule="independent", w_prior="w0" (defaults before H9d).

NOTE: pinned to w_rule="hebb_r" (the default when this experiment was run; default is mean_succ since H8c).
H6: is phase (Kuramoto) useful for ordering steps plan -> execute -> verify? (notes/hypotheses.md)

The phase variants (phase_static, phase_kuramoto, phase_kuramoto_filter, posthoc phase_mean_*) need the
Kuramoto code, removed after this experiment: run them with `git checkout d2d10b0`. Their numbers are in
results/h6_ordered_kuramoto.json. On the current code this script runs the remaining variants."""
import argparse

from common import base_args, paired, run_variants, save, table
from swarm import EnvConfig, SwarmConfig

ap = base_args(argparse.ArgumentParser()); a = ap.parse_args()
seeds = list(range(a.seed0, a.seed0 + a.seeds))
env = EnvConfig("ordered", a.T)
B = {}
V = {
    "recruit": SwarmConfig(recruit_rule="independent", w_prior="w0", w_rule="hebb_r", order="recruit", **B),
    "random": SwarmConfig(recruit_rule="independent", w_prior="w0", w_rule="hebb_r", order="random", **B),
    "position": SwarmConfig(recruit_rule="independent", w_prior="w0", w_rule="hebb_r", order="position", **B),
    "oracle": SwarmConfig(recruit_rule="independent", w_prior="w0", w_rule="hebb_r", order="oracle", **B),
}
res = run_variants({k: ("swarm", env, v) for k, v in V.items()}, seeds, a.workers)
res["linucb + pair_ucb c=0.03 (order = selection order)"] = run_variants(
    {"x": ("router", env, dict(model="linucb", team="pair_ucb", c=0.03))}, seeds, a.workers)["x"]
for c in (0.03, 0.1, 0.3):
    res[f"linucb + seq_ucb c={c} (order-aware)"] = run_variants(
        {"x": ("router", env, dict(model="linucb", team="seq_ucb", c=c))}, seeds, a.workers)["x"]
res["linucb + oracle team"] = run_variants({"x": ("router", env, dict(model="linucb", team="oracle"))}, seeds, a.workers)["x"]
print(table(res, ("post", "cover_end", "q_end")))
cmp = [paired(res, "position", "recruit"), paired(res, "position", "random"), paired(res, "oracle", "position")]
best_seq = max([k for k in res if "seq_ucb" in k], key=lambda k: res[k]["post"])
cmp.append(paired(res, "position", best_seq))
for c in cmp:
    print(c)
save("h6_ordered_nokuramoto", dict(T=a.T, seeds=seeds, results=res, comparisons=cmp))
