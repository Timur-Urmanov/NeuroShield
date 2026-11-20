"""
NOTE: pinned to recruit_rule="independent", w_prior="w0" (defaults before H9d).

NOTE: pinned to w_rule="hebb_r" (the default when this experiment was run; default is mean_succ since H8c).
H1: swarm W_reward vs centralised baselines in the roles env (notes/hypotheses.md)."""
import argparse

from common import base_args, paired, run_variants, save, table
from swarm import EnvConfig, SwarmConfig
from swarm.envs import RolesEnv, SimEnv

ap = base_args(argparse.ArgumentParser()); a = ap.parse_args()
seeds = list(range(a.seed0, a.seed0 + a.seeds))
env = EnvConfig("roles", a.T)
S = SwarmConfig(recruit_rule="independent", w_prior="w0", w_rule="hebb_r")   # minimal architecture: W_reward, s=0.5, NE on, no kuramoto/fatigue/sleep
V = {
    "swarm W_reward": ("swarm", env, S),
    "swarm sim-only (no W)": ("swarm", env, SwarmConfig(recruit_rule="independent", w_prior="w0", w_rule="hebb_r", plastic_W=False)),
    "swarm solo": ("swarm", env, SwarmConfig(recruit_rule="independent", w_prior="w0", w_rule="hebb_r", coupling=False, plastic_W=False)),
    "linucb + solo": ("router", env, dict(model="linucb", team="solo")),
    "linucb + random3": ("router", env, dict(model="linucb", team="random")),
    "linucb + agent_ucb": ("router", env, dict(model="linucb", team="agent_ucb", c=0.3)),
    **{f"linucb + pair_ucb c={c}": ("router", env, dict(model="linucb", team="pair_ucb", c=c))
       for c in (0.03, 0.1, 0.3, 1.0)},
    "linucb + oracle team": ("router", env, dict(model="linucb", team="oracle")),
    "oracle model + random3": ("router", env, dict(model="oracle", team="random")),
    "oracle model + oracle team": ("router", env, dict(model="oracle", team="oracle")),
}
res = run_variants(V, seeds, a.workers)
print(table(res))
pair = [k for k in res if "pair_ucb" in k]
best = max(pair, key=lambda k: res[k]["post"])
cmp = [paired(res, "swarm W_reward", best), paired(res, "swarm W_reward", "linucb + random3"),
       paired(res, "swarm W_reward", "swarm sim-only (no W)"), paired(res, best, "linucb + random3")]
for c in cmp:
    print(c)
r = RolesEnv(10, 0); e = SimEnv(10, 0)
analytic = {"E[q] random team of 3": r.expected_random_team_q(3), "q full cover": r.q_full,
            "oracle model, E[q] random team (approx: oracle model at mean q)": e.oracle_value("post", r.expected_random_team_q(3)),
            "oracle model + full cover": e.oracle_value("post", r.q_full)}
print(analytic)
save("h1_roles_baselines", dict(T=a.T, seeds=seeds, results=res, best_pair_ucb=best, comparisons=cmp, analytic=analytic))
