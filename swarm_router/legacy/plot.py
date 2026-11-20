import numpy as np, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import swarm as S
r = json.load(open("results.json")); T = r["T"]
C = np.load("curves.npy", allow_pickle=True).item()
W = np.load("W_full.npy")
fig, ax = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw=dict(width_ratios=[1.7, 1]))
def roll(x, w=300): return np.convolve(x, np.ones(w) / w, mode="valid")
lab = {"full": "рой (полный)", "no_sleep": "рой без усталости и сна", "no_coupling": "рой без связей (соло)",
       "no_sleep_fat": "рой с усталостью, без сна", "linucb": "LinUCB (базовая линия)"}
col = {"full": "#2563eb", "no_sleep": "#16a34a", "no_coupling": "#9333ea", "no_sleep_fat": "#dc2626", "linucb": "#111827"}
for k in ["full", "no_sleep", "no_coupling", "no_sleep_fat", "linucb"]:
    y = roll(C[k].mean(0)); ax[0].plot(np.arange(len(y)) + 300, y, label=lab[k], color=col[k],
                                       lw=2.2 if k in ("full", "linucb") else 1.4, ls="--" if k == "linucb" else "-")
o = r["baselines"]["oracle"]
ax[0].hlines([o["pre"]], 0, T // 2, colors="gray", ls=":"); ax[0].hlines([o["post"]], T // 2, T, colors="gray", ls=":", label="оракул")
ax[0].axvline(T // 2, color="gray", alpha=.4); ax[0].text(T // 2 + 60, 0.47, "модель A деградировала", color="gray", fontsize=9)
ax[0].set_xlabel("задача №"); ax[0].set_ylabel("награда (скользящее среднее, 5 сидов)")
ax[0].set_ylim(0.45, 0.8); ax[0].legend(fontsize=8.5, loc="lower right"); ax[0].set_title("Качество маршрутизации")
cl = S.clusters(W); order = [i for c in sorted(cl, key=len, reverse=True) for i in c]
order += [i for i in range(len(W)) if i not in order]
im = ax[1].imshow(W[np.ix_(order, order)], cmap="Blues", vmin=0, vmax=1)
ax[1].set_title(f"Связи W после обучения: {len(cl)} кластера(ов)"); ax[1].set_xticks([]); ax[1].set_yticks([])
plt.colorbar(im, ax=ax[1], fraction=.046)
plt.tight_layout(); plt.savefig("swarm_results.png", dpi=140)
