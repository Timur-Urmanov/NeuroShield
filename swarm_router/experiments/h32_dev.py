"""H32 stage 1: tuning grid and viability of the ES baseline on dev seeds (criteria in notes/hypotheses.md)."""
import os, sys

import numpy as np

from common import ROOT, save

D = os.path.join(ROOT, ".cache", "h32dev"); SEEDS = (100, 101, 102)


def stats(d, tag):
    F = [np.load(os.path.join(D, d, f"{tag}_{s}.npz")) for s in SEEDS if os.path.exists(os.path.join(D, d, f"{tag}_{s}.npz"))]
    if len(F) < len(SEEDS):
        return None
    o = dict(n=len(F))
    o["late"] = float(np.mean([f["R"][120:].mean() for f in F])); o["late_sd"] = float(np.std([f["R"][120:].mean() for f in F], ddof=1)); o["total"] = float(np.mean([f["R"].mean() for f in F]))
    ev = np.array([f["ev"].sum(0) for f in F]); ep = [np.sort(f["epoch"][-1])[::-1] for f in F]
    o["top32"] = float(np.mean([e[:32].sum() / max(1, e.sum()) for e in ep])); o["zero_use"] = float((ev[:, 2] / np.maximum(ev[:, 0], 1)).mean())
    o["uses_evicted"] = float((ev[:, 3] / np.maximum(ev[:, 0], 1)).mean()); o["perf_born"] = float(ev[:, 1].mean())
    o["perf_long"] = float((ev[:, 5] / np.maximum(ev[:, 4], 1)).mean())             # of the perfect ones that died: share that had lived >= 5000 tasks
    cnt = []
    for f in F:
        g, tg = f["genes"][-1], f["targets"]; cnt.append(np.array([[(g == tg[k, j]).all(1).sum() for j in range(3)] for k in range(3)]))
    cnt = np.array(cnt); o["niches"] = float((cnt > 0).sum((1, 2)).mean()); o["perf_alive"] = float(cnt.sum((1, 2)).mean()); o["niches16"] = float((cnt >= 16).sum((1, 2)).mean())
    o["subs_per_task"] = float(np.mean([f["subs"].sum() / (160 * int(f["blk"])) for f in F]))
    o["losses"] = [float(np.mean([(0.95 ** 3 - f["q_pop"])[120:].mean() for f in F])), float(np.mean([(f["q_pop"] - f["q_avail"])[120:].mean() for f in F])),
                   float(np.mean([(f["q_avail"] - f["q_team"])[120:].mean() for f in F]))]
    return o


if __name__ == "__main__":
    out = {}
    print(f"{'config':<16} late   sd    total  top32 zero  uses  pborn  plong niches n16 palive subs  comp  busy  route")
    for d in sorted(os.listdir(D)):
        for tag in ("NI", "A", "ES", "EC", "NT"):
            o = stats(d, tag)
            if o:
                out[f"{d}/{tag}"] = o
                print(f"{d + '/' + tag:<16} {o['late']:.3f} {o['late_sd']:.3f} {o['total']:.3f}  {o['top32']:.2f}  {o['zero_use']:.2f} {o['uses_evicted']:5.1f} {o['perf_born']:6.0f} {o['perf_long']:.2f}  {o['niches']:.1f}  {o['niches16']:.1f} {o['perf_alive']:5.0f}  {o['subs_per_task']:.2f}  " + " ".join(f"{x:.3f}" for x in o["losses"]))
    es = {k: v for k, v in out.items() if k.endswith("/ES")}
    if es:
        best = max(es, key=lambda k: es[k]["late"]); out["best"] = best; print("best ES config by late reward:", best, round(es[best]["late"], 4))
    if "save" in sys.argv:
        save("h32_dev", out)
