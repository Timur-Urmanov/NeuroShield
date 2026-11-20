"""Phase 0: reproduce the tables of CLAUDE.md section 2 with the refactored package.
Kuramoto rows were removed with the Kuramoto code (H6); results/reproduce_T8000.json (commit d508726) has
them, reproduced bit-exactly before removal.
python3 experiments/reproduce.py [--T 8000 --seeds 5]"""
import argparse

from common import base_args, run_variants, save, table
from swarm import LEGACY, EnvConfig
from swarm.config import replace
from swarm.envs import RolesEnv, SimEnv

ap = base_args(argparse.ArgumentParser()); a = ap.parse_args()
seeds = list(range(a.seed0, a.seed0 + a.seeds))
CLEAN = replace(LEGACY, sleep=False, fatigue=False)
V = {"W_only": dict(), "neither": dict(plastic_W=False), "solo": dict(coupling=False, plastic_W=False)}
VR = dict(V, **{"W_reward": dict(hebb_bias=0.0), "W_reward_sim0": dict(hebb_bias=0.0, sim_w=0.0)})
hom, rol = EnvConfig("homogeneous", a.T), EnvConfig("roles", a.T)
exp1 = run_variants({**{n: ("swarm", hom, replace(CLEAN, **o)) for n, o in V.items()},
                     "linucb": ("router", hom, dict(model="linucb"))}, seeds, a.workers)
exp2 = run_variants({n: ("swarm", rol, replace(CLEAN, **o)) for n, o in VR.items()}, seeds, a.workers)
exp2_full = run_variants({n: ("swarm", rol, replace(LEGACY, **o)) for n, o in VR.items()}, seeds, a.workers)
e = SimEnv(10, 0); r = RolesEnv(10, 0)
analytic = dict(oracle_post=e.oracle_value("post"), best_single_post=e.best_single_value("post"),
                roles_oracle_post=e.oracle_value("post", r.q_full))
print("## exp1 homogeneous\n" + table(exp1, ("pre", "post")))
print("## exp2 roles clean\n" + table(exp2, ("post", "cover_end")))
print("## exp2 roles full (fatigue+sleep)\n" + table(exp2_full, ("post", "cover_end")))
print(analytic)
save(f"reproduce_nokuramoto_T{a.T}", dict(T=a.T, seeds=seeds, exp1=exp1, exp2=exp2, exp2_full=exp2_full, analytic=analytic))
