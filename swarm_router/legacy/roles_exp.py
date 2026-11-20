"""Многошаговые задачи с комплементарными ролями: нужен ли ChannelMatrix / Курамото?"""
import numpy as np, json, sys
import swarm as S
T, SEEDS = 8000, 5
cfgname = sys.argv[1] if len(sys.argv) > 1 else "clean"
base = dict(roles=True, sleep=False, fatigue=False) if cfgname == "clean" else dict(roles=True)
V = {
 "W + Курамото":       dict(),
 "только W":           dict(kuramoto=False),
 "только Курамото":    dict(plastic_W=False),
 "ни того ни другого": dict(kuramoto=False, plastic_W=False),
 "соло":               dict(coupling=False, kuramoto=False, plastic_W=False),
 "W, Хебб только по награде": dict(kuramoto=False, hebb_bias=0.0),
 "W(награда) + Курамото": dict(hebb_bias=0.0),
 "W(награда) + сходство=0": dict(kuramoto=False, hebb_bias=0.0, sim_w=0.0),
}
out = {}
for name, cfg in V.items():
    rows = []
    for sd in range(SEEDS):
        s, R, Ms, ks = S.run_swarm(T, sd, **base, **cfg)
        cov = np.array(s.coverage) if s.coverage else np.zeros(T)
        rows.append([R[T//2-1000:T//2].mean(), R[-1000:].mean(), cov[:1000].mean(), cov[-2000:].mean()])
    a = np.array(rows)
    out[name] = dict(pre=round(a[:,0].mean(),3), post=round(a[:,1].mean(),3), post_sd=round(a[:,1].std(),3),
                     cover_start=round(a[:,2].mean(),3), cover_end=round(a[:,3].mean(),3))
    print(name, out[name], flush=True)
# оракул: идеальное покрытие ролей + лучшая модель
q = 0.95**3
for tag, acc in (("pre", S.ACC0), ("post", S.ACC1)):
    val = acc*q - S.LAM*S.COST[:,None]
    out.setdefault("оракул", {})[tag] = round(float(sum(S.MIX[k]*val[:,k].max() for k in range(S.K))),3)
print("оракул", out["оракул"])
json.dump(out, open(f"roles_{cfgname}.json","w"), ensure_ascii=False, indent=1)
