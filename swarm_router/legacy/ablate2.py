import numpy as np, json, sys
import swarm as S
T, SEEDS = 8000, 5
base = dict(sleep=False, fatigue=False)
V = {
 "W + Курамото":        dict(),
 "только W":            dict(kuramoto=False),
 "только Курамото":     dict(plastic_W=False),
 "ни того ни другого":  dict(kuramoto=False, plastic_W=False),
 "соло (без команд)":   dict(coupling=False, kuramoto=False, plastic_W=False),
}
def spec_metrics(s, lo, hi):
    led = np.array(s.led_type); sel = (led[:,0]>=lo)&(led[:,0]<hi)
    L = led[sel]
    # доминантный тип каждого примитива по задачам, где он лидер
    dom = {}; pur_num = pur_den = 0
    for i in np.unique(L[:,1]):
        c = np.bincount(L[L[:,1]==i][:,2], minlength=S.K)
        dom[i] = c.argmax(); pur_num += c.max(); pur_den += c.sum()
    # однородность команд: доля членов команды, чей доминантный тип = тип задачи
    hom_num = hom_den = 0
    for t in np.where(sel)[0]:
        k = led[t,2]
        for j in s.teams[t][1:]:
            if j in dom: hom_den += 1; hom_num += dom[j]==k
    return pur_num/pur_den, (hom_num/hom_den if hom_den else float('nan')), np.mean([len(x) for x in np.array(s.teams,dtype=object)[sel]])
out = {}
for name, cfg in V.items():
    rows=[]
    for sd in range(SEEDS):
        s,R,Ms,ks = S.run_swarm(T, sd, **base, **cfg)
        pur, hom, tsz = spec_metrics(s, T-2000, T)
        # переключение: доля выбора C для типа 0 в последних 1500
        led=np.array(s.led_type); w=led[(led[:,0]>=T-1500)&(led[:,2]==0)]
        swC = (w[:,3]==2).mean()
        rows.append([R[T//2-1000:T//2].mean(), R[-1000:].mean(), pur, hom, tsz, swC])
    a=np.array(rows); out[name]=dict(zip(["pre","post","purity","team_hom","team_size","type0_to_C"], a.mean(0).round(3).tolist()))
    out[name]["post_sd"]=round(float(a[:,1].std()),3)
    print(name, out[name], flush=True)
json.dump(out, open("ablate2.json","w"), ensure_ascii=False, indent=1)
