"""H33: tuning of the two controllers on dev seeds (T = 1.2M, targets of 3 niches redrawn at 700k)."""
import os, sys

import numpy as np

from common import ROOT, save

D = os.path.join(ROOT, ".cache", "h33dev"); SEEDS = (100, 101, 102)

if __name__ == "__main__":
    out = {}
    print(f"{'variant':<22} 0.4-1.2M  pre(0.4-0.7) rec(0.7-0.95) late(0.95-1.2) | births/gen pre  rec  late | alarms | q_pop late | routing pre late")
    for d in sorted(os.listdir(D)):
        for tag in ("NI", "F500", "N4"):
            fs = [os.path.join(D, d, f"{tag}_{s}.npz") for s in SEEDS]
            if not all(os.path.exists(f) for f in fs):
                continue
            F = [np.load(f) for f in fs]; blk = int(F[0]["blk"]); b = lambda t: int(round(t / blk))
            R = np.array([f["R"] for f in F]); ev = np.array([f["ev"][:, 0] for f in F]); gen = blk / 100
            seg = lambda y, a, c: float(y[:, b(a):b(c)].mean())
            o = dict(main=seg(R, 4e5, 1.2e6), main_sd=float(R[:, b(4e5):].mean(1).std(ddof=1)), pre=seg(R, 4e5, 7e5), rec=seg(R, 7e5, 9.5e5), late=seg(R, 9.5e5, 1.2e6),
                     births=[seg(ev, 4e5, 7e5) / gen, seg(ev, 7e5, 9.5e5) / gen, seg(ev, 9.5e5, 1.2e6) / gen], alarms=float(np.mean([f["n_alarm"] for f in F])),
                     q_pop_late=float(np.mean([f["q_pop"][b(9.5e5):].mean() for f in F])),
                     routing=[float(np.mean([(f["q_avail"] - f["q_team"])[b(4e5):b(7e5)].mean() for f in F])), float(np.mean([(f["q_avail"] - f["q_team"])[b(9.5e5):].mean() for f in F]))],
                     total_births=float(ev.sum(1).mean()))
            out[f"{d}/{tag}"] = o
            print(f"{d + '/' + tag:<22} {o['main']:.4f}±{o['main_sd']:.3f}  {o['pre']:.4f}  {o['rec']:.4f}  {o['late']:.4f} | {o['births'][0]:5.1f} {o['births'][1]:5.1f} {o['births'][2]:5.1f} | {o['alarms']:6.0f} | {o['q_pop_late']:.3f} | {o['routing'][0]:.3f} {o['routing'][1]:.3f}")
    for k in ("g_", "z_"):
        c = {n: v for n, v in out.items() if n.startswith(k)}
        if c:
            best = max(c, key=lambda n: c[n]["main"]); out["best_" + k] = best; print("best", k, best, round(c[best]["main"], 4))
    if "save" in sys.argv:
        save("h33_dev", out)
