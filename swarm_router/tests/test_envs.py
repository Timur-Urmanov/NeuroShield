import numpy as np

from swarm import EnvConfig, SwarmConfig, run_swarm
from swarm.baselines import run_router, summary
from swarm.envs import OrderedEnv, RolesEnv, assign_roles, make_stream


def test_stream_is_shared_across_envs():
    a, b = RolesEnv(100, 3), OrderedEnv(100, 3, role_by_type=True, busy=5)
    ks, X, U = make_stream(100, 3)
    for e in (a, b):
        assert np.array_equal(e.ks, ks) and np.array_equal(e.X, X) and np.array_equal(e.U, U)


def test_roles_team_factor():
    e = RolesEnv(10, 0, N=6)                       # roles 0,1,2,0,1,2
    assert np.isclose(e.team_factor(np.array([0, 1, 2])), 0.95 ** 3)
    assert np.isclose(e.team_factor(np.array([0, 3])), 0.95 * 0.6 * 0.6)
    assert e.coverage(np.array([2, 0, 1])) and not e.coverage(np.array([0, 3, 1]))


def test_ordered_team_factor_depends_on_order():
    e = OrderedEnv(10, 0, N=6)
    assert np.isclose(e.team_factor(np.array([0, 1, 2])), 0.95 ** 3)
    assert np.isclose(e.team_factor(np.array([2, 1, 0])), 0.95 * 0.6 * 0.6)


def test_role_fractions():
    r = assign_roles(40, (0.8, 0.1, 0.1), seed=1)
    assert np.bincount(r, minlength=3).tolist() == [32, 4, 4]
    assert np.array_equal(assign_roles(9), np.arange(9) % 3)


def test_role_by_type_and_random_team_q():
    e = RolesEnv(50, 0, role_by_type=True)
    assert all(np.bincount(e.role_k[k], minlength=3).tolist() == np.bincount(e.role, minlength=3).tolist()
               for k in range(3))
    assert not np.array_equal(e.role_k[0], e.role_k[1])
    rng = np.random.default_rng(0)
    mc = np.mean([e.team_factor(rng.choice(40, 3, replace=False)) for _ in range(20000)])
    assert abs(mc - e.expected_random_team_q(3)) < 0.01


def test_busy_primitives_are_never_reused():
    env = EnvConfig("roles", 300, busy=6)
    tr = run_swarm(env, SwarmConfig(), 0, keep=True)
    last = {}
    for t, team in enumerate(tr.teams):
        for i in team:
            assert t - last.get(i, -100) > 6
            last[i] = t
    out = run_router(env, 0, team="pair_ucb")
    assert np.isfinite(out["R"]).all()


def test_oracle_team_bounds_router():
    env = EnvConfig("roles", 3000)
    o = summary(run_router(env, 0, model="oracle", team="oracle"), 3000)
    r = summary(run_router(env, 0, model="oracle", team="random"), 3000)
    assert o["cover_end"] == 1.0 and o["post"] > r["post"]
