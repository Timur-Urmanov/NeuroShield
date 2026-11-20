"""
Игрушечный прототип роя примитивов над несколькими LLM.

Примитив = дешёвый агент с состоянием:
  - фаза theta (Курамото) -> кто синхронен, тот работает в одной команде
  - affinity-вектор в пространстве задач -> специализация
  - оценки Q[m] полезности каждой модели (рабочая память + долговременная)
  - усталость F (аналог разбухшего контекста: качество падает)
  - сон: офлайн-консолидация рабочей памяти в долговременную, прунинг связей

Глобальные «гормоны» (медленные переменные):
  - dopamine  = ошибка предсказания награды (сила пластичности на шаге)
  - NE        = стресс: недавняя награда ниже долгосрочной -> больше шума,
                выше температура выбора, быстрее забывание старых оценок
  - melatonin = средняя усталость -> давление сна

LLM заменены симуляцией: у каждой модели своя точность по типу задачи и цена.
В середине прогона «модель A деградирует» (смена режима), чтобы проверить адаптацию.

Тип задачи рою НЕ сообщается: он видит только зашумлённый эмбеддинг x.
"""
import numpy as np
import json, argparse, time

D, K, M = 8, 3, 3
# ACC[model, type]
ACC0 = np.array([[0.90, 0.55, 0.60],   # A: большая, сильна в «математике»
                 [0.55, 0.88, 0.62],   # B: большая, сильна в «коде»
                 [0.70, 0.70, 0.80]])  # C: дешёвая, сильна в «тексте»
ACC1 = ACC0.copy(); ACC1[0, 0] = 0.60; ACC1[2, 0] = 0.85   # после сдвига
COST = np.array([1.0, 1.0, 0.3]); LAM = 0.15
MIX = np.array([0.4, 0.3, 0.3])
TASK_NOISE = 0.35


def acc_at(t, T):
    return ACC0 if t < T // 2 else ACC1


def expected(acc, policy):
    """Ожидаемая награда стратегии policy[k] -> model (или распределение)."""
    val = acc - LAM * COST[:, None]            # [m, k]
    return float(sum(MIX[k] * (policy[k] @ val[:, k]) for k in range(K)))


def make_stream(T, seed):
    rng = np.random.default_rng(seed)
    C = rng.normal(size=(K, D)); C /= np.linalg.norm(C, axis=1, keepdims=True)
    ks = rng.choice(K, size=T, p=MIX)
    X = C[ks] + rng.normal(scale=TASK_NOISE, size=(T, D))
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    U = rng.random(T)                          # общий источник исхода задачи
    return ks, X, U


class Swarm:
    def __init__(self, N=40, seed=0, sleep=True, hormones=True, coupling=True,
                 fatigue=True, team=3, kuramoto=True, plastic_W=True,
                 roles=False, hebb_bias=0.3, sim_w=0.5):
        self.rng = np.random.default_rng(seed + 1000)
        self.N, self.sleep, self.hormones = N, sleep, hormones
        self.coupling, self.fatigue, self.team = coupling, fatigue, team
        self.kuramoto, self.plastic_W = kuramoto, plastic_W
        self.hebb_bias, self.sim_w = hebb_bias, sim_w
        # скрытые роли: каждый примитив хорош в одной из 3 ролей (0.95), в остальных слаб (0.6)
        self.roles = roles
        self.role = np.arange(N) % 3
        self.coverage = []
        self.teams = []
        r = self.rng
        self.theta = r.uniform(0, 2 * np.pi, N)
        self.omega = r.normal(1.0, 0.1, N)
        self.aff = r.normal(size=(N, D)); self.aff /= np.linalg.norm(self.aff, axis=1, keepdims=True)
        self.QL = np.full((N, M), 0.6); self.nL = np.ones((N, M))
        self.sW = np.zeros((N, M)); self.nW = np.zeros((N, M))
        W = r.uniform(0, 0.2, (N, N)); self.W = (W + W.T) / 2; np.fill_diagonal(self.W, 0)
        self.F = np.zeros(N); self.asleep = np.zeros(N, bool); self.timer = np.zeros(N, int)
        self.r_fast = self.r_slow = 0.5
        self.NE = 0.0
        self.led_type = []                     # (t, leader, k, m) для анализа

    # ---------- помощники ----------
    def Qest(self, idx):
        return (self.QL[idx] * self.nL[idx] + self.sW[idx]) / (self.nL[idx] + self.nW[idx])

    def order_param(self, mask):
        if mask.sum() == 0: return 0.0
        return float(np.abs(np.exp(1j * self.theta[mask]).mean()))

    def consolidate(self, i):
        """Сон: рабочая память -> долговременная, прунинг слабых связей."""
        n = self.nL[i] + self.nW[i]
        self.QL[i] = (self.QL[i] * self.nL[i] + self.sW[i]) / n
        self.nL[i] = np.minimum(n, 100.0)
        self.sW[i] = 0; self.nW[i] = 0
        if not self.plastic_W: return
        weak = self.W[i] < 0.05
        self.W[i, weak] = 0; self.W[weak, i] = 0

    # ---------- один шаг = одна задача ----------
    def step(self, t, x, k, u, acc):
        N, awake = self.N, ~self.asleep
        NE = self.NE if self.hormones else 0.0

        # 1. ставки: насколько задача «похожа на мою»
        bids = (self.aff @ x) * (1 - 0.5 * self.F) + 0.02 * self.rng.normal(size=N)
        bids[~awake] = -np.inf
        leader = int(np.argmax(bids))

        # 2. команда: синхронные по фазе и сильно связанные соседи лидера
        members = [leader]
        if self.coupling:
            sim = np.clip(self.aff @ x, 0, None)
            ph = (1 + np.cos(self.theta - self.theta[leader])) / 2 if self.kuramoto else 1.0
            sc = ((1 - self.sim_w) * self.W[leader] + self.sim_w * sim) * ph
            sc[~awake] = -1; sc[leader] = -1
            cand = np.argsort(-sc)[: self.team - 1]
            members += [int(j) for j in cand if sc[j] > 0.25]
        members = np.array(members)

        # 3. выбор модели по общей памяти команды
        rel = np.clip(self.aff[members] @ x, 0.05, None)
        Qt = (rel[:, None] * self.Qest(members)).sum(0) / rel.sum()
        temp = 0.05 * (1 + 2 * NE) if self.hormones else 0.06
        p = np.exp((Qt - Qt.max()) / temp); p /= p.sum()
        m = int(self.rng.choice(M, p=p))

        # 4. исход
        fat = self.F[members].mean() if self.fatigue else 0.0
        q = 1.0
        if self.roles:
            comp = np.where(self.role[members][:, None] == np.arange(3)[None], 0.95, 0.6)
            q = float(np.prod(comp.max(0)))
            self.coverage.append(len(set(self.role[members].tolist())) == 3)
        succ = float(u < acc[m, k] * q * (1 - 0.3 * fat))
        r = succ - LAM * COST[m]

        # 5. гормоны
        dopa = r - self.r_fast                      # ошибка предсказания
        self.r_fast += (r - self.r_fast) / 30
        self.r_slow += (r - self.r_slow) / 400
        self.NE = float(np.clip((self.r_slow - self.r_fast) / 0.1, 0, 1))
        mel = self.F[awake].mean() if awake.any() else 0.0

        # 6. обучение
        eta = 0.05 * (1 + NE) if self.hormones else 0.05
        for j in members:
            if self.sleep:
                self.sW[j, m] += r; self.nW[j, m] += 1
            else:                                   # онлайн-консолидация
                self.QL[j, m] = (self.QL[j, m] * self.nL[j, m] + r) / (self.nL[j, m] + 1)
                self.nL[j, m] = min(self.nL[j, m] + 1, 100.0)
            # конкурентное обучение: тянемся к своим задачам, сильнее при успехе
            w_pull = (1.0 if j == leader else 0.5) * max(0.0, 0.5 + dopa)
            self.aff[j] += eta * w_pull * x
            self.aff[j] /= np.linalg.norm(self.aff[j])
        if len(members) > 1 and self.plastic_W:     # Хебб внутри команды
            for a in members:
                for b in members:
                    if a != b:
                        self.W[a, b] = np.clip(self.W[a, b] + 0.5 * eta * (dopa + self.hebb_bias), 0, 1)
        if self.plastic_W: self.W *= (1 - 0.0005)
        if self.sleep:
            self.sW *= 0.998; self.nW *= 0.998      # рабочая память протекает
        if self.hormones and NE > 0:                # стресс -> пересматриваем старое
            self.nL = np.maximum(1.0, self.nL * (1 - 0.02 * NE))

        # 7. усталость и сон
        if self.fatigue:
            self.F[members] += 0.04 * (1 - self.F[members])
            self.F[awake] *= 0.999
        if self.sleep:
            woke = self.asleep & (self.timer <= 0)
            for i in np.where(woke)[0]:
                self.asleep[i] = False; self.consolidate(i)
            self.timer[self.asleep] -= 1
            self.F[self.asleep] *= 0.9
            thr = 0.6 * (1 - 0.3 * mel) if self.hormones else 0.6
            for i in np.where(~self.asleep & (self.F > thr))[0]:
                if self.asleep.mean() < 0.3:
                    self.asleep[i] = True; self.timer[i] = 40

        # 8. фазовая динамика Курамото
        self.led_type.append((t, leader, k, m)); self.teams.append(members.tolist())
        if not self.kuramoto:
            return r, m
        aw = ~self.asleep
        R = self.order_param(aw)
        sigma = 0.1 * (1 + 1.0 * (1 - R)) * (1 + NE) if self.hormones else 0.1
        Kc = 2.0 if self.coupling else 0.0
        diff = np.sin(self.theta[None, :] - self.theta[:, None])
        coup = Kc / N * ((self.W * aw[None, :]) * diff).sum(1)
        om = np.where(aw, self.omega, 0.3 * self.omega)
        self.theta += 0.1 * (om + coup * aw) + np.sqrt(0.1) * sigma * self.rng.normal(size=N)
        # синхронизация подтягивает фазы тех, кто вместе удачно работал
        if len(members) > 1 and dopa > 0:
            mean_phase = np.angle(np.exp(1j * self.theta[members]).mean())
            self.theta[members] += 0.1 * np.sin(mean_phase - self.theta[members])

        return r, m


def clusters(W, thr=0.4):
    N = len(W); seen = np.zeros(N, bool); out = []
    for s in range(N):
        if seen[s]: continue
        stack, comp = [s], []
        seen[s] = True
        while stack:
            i = stack.pop(); comp.append(i)
            for j in np.where((W[i] > thr) & ~seen)[0]:
                seen[j] = True; stack.append(j)
        if len(comp) >= 2: out.append(sorted(comp))
    return out


def run_swarm(T, seed, **cfg):
    ks, X, U = make_stream(T, seed)
    s = Swarm(seed=seed, **cfg)
    R = np.zeros(T); Ms = np.zeros(T, int)
    for t in range(T):
        R[t], Ms[t] = s.step(t, X[t], ks[t], U[t], acc_at(t, T))
    return s, R, Ms, ks


def run_linucb(T, seed, alpha=0.4, gamma=0.995):
    """Сильная базовая линия: контекстный бандит (дисконтированный LinUCB) на тех же x."""
    ks, X, U = make_stream(T, seed)
    rng = np.random.default_rng(seed + 7)
    d = D + 1
    A = np.stack([np.eye(d)] * M); b = np.zeros((M, d))
    R = np.zeros(T)
    for t in range(T):
        z = np.append(X[t], 1.0)
        sc = []
        for m in range(M):
            Ai = np.linalg.inv(A[m]); th = Ai @ b[m]
            sc.append(th @ z + alpha * np.sqrt(z @ Ai @ z))
        m = int(np.argmax(sc))
        acc = acc_at(t, T)
        r = float(U[t] < acc[m, ks[t]]) - LAM * COST[m]
        A = gamma * A + (1 - gamma) * np.eye(d)[None]
        b = gamma * b
        A[m] += np.outer(z, z); b[m] += r * z
        R[t] = r
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--T", type=int, default=8000)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default="results.json")
    a = ap.parse_args()
    T = a.T
    variants = {
        "full":         dict(),
        "no_sleep":     dict(sleep=False, fatigue=False),   # идеал: без усталости, онлайн-память
        "no_sleep_fat": dict(sleep=False, fatigue=True),    # усталость есть, сна нет
        "no_hormones":  dict(hormones=False),
        "no_coupling":  dict(coupling=False),
    }
    res = {"T": T, "variants": {}, "baselines": {}}
    curves = {}
    for name, cfg in variants.items():
        pre, post, rec = [], [], []
        t0 = time.time()
        for sd in range(a.seeds):
            s, R, Ms, ks = run_swarm(T, sd, **cfg)
            pre.append(R[T // 2 - 1000:T // 2].mean()); post.append(R[-1000:].mean())
            curves.setdefault(name, []).append(R.tolist())
            if name == "full" and sd == 0:
                np.save("W_full.npy", s.W); np.save("aff_full.npy", s.aff)
                cl = clusters(s.W)
                led = np.array(s.led_type)
                info = []
                win = led[(led[:, 0] >= T // 2 - 1500) & (led[:, 0] < T // 2)]
                win2 = led[led[:, 0] >= T - 1500]
                for c in cl:
                    for tag, w in (("pre", win), ("post", win2)):
                        sel = w[np.isin(w[:, 1], c)]
                        if len(sel) == 0: continue
                        tc = np.bincount(sel[:, 2], minlength=K)
                        dom = int(tc.argmax())
                        mc = np.bincount(sel[sel[:, 2] == dom][:, 3], minlength=M)
                        info.append(dict(phase=tag, size=len(c), members=[int(v) for v in c], tasks=int(len(sel)),
                                         dominant_type=dom, purity=float(tc[dom] / tc.sum()),
                                         model_for_dominant=mc.tolist()))
                res["clusters_seed0"] = info
                res["n_clusters_seed0"] = len(cl)
        res["variants"][name] = dict(pre=float(np.mean(pre)), pre_sd=float(np.std(pre)),
                                     post=float(np.mean(post)), post_sd=float(np.std(post)),
                                     sec=round(time.time() - t0, 1))
        print(name, res["variants"][name], flush=True)
    lu_pre, lu_post, lu_curves = [], [], []
    for sd in range(a.seeds):
        R = run_linucb(T, sd)
        lu_pre.append(R[T // 2 - 1000:T // 2].mean()); lu_post.append(R[-1000:].mean())
        lu_curves.append(R.tolist())
    curves["linucb"] = lu_curves
    res["baselines"]["linucb"] = dict(pre=float(np.mean(lu_pre)), post=float(np.mean(lu_post)))
    oh = lambda ms: [np.eye(M)[mm] for mm in ms]
    uni = [np.full(M, 1 / M)] * K
    for tag, acc in (("pre", ACC0), ("post", ACC1)):
        val = acc - LAM * COST[:, None]
        res["baselines"].setdefault("oracle", {})[tag] = expected(acc, oh(val.argmax(0)))
        best_single = max(expected(acc, oh([mm] * K)) for mm in range(M))
        res["baselines"].setdefault("best_single", {})[tag] = best_single
        res["baselines"].setdefault("random", {})[tag] = expected(acc, uni)
    print(json.dumps(res["baselines"], indent=1))
    json.dump(res, open(a.out, "w"), indent=1)
    np.save("curves.npy", {k: np.array(v) for k, v in curves.items()}, allow_pickle=True)


if __name__ == "__main__":
    main()
