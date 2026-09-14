# -*- coding: utf-8 -*-
import json, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "eval_results" / "archived"
MOVING_THRESHOLD = 0.02
CMD_VX_LEVELS = [0.0375, 0.0750, 0.1125, 0.1500]

RUN_META = {
    "p2a_siglin_0p01":  ("Phase2a", "σ_lin=0.010"),
    "p2a_siglin_0p05":  ("Phase2a", "σ_lin=0.050"),
    "p2b_sigang_0p25":  ("Phase2b", "σ_ang=0.25"),
    "p2b_sigang_0p5":   ("Phase2b", "σ_ang=0.50"),
    "p2b_sigang_1p0":   ("Phase2b", "σ_ang=1.00"),
    "p2b_sigang_2p0":   ("Phase2b", "σ_ang=2.00"),
    "p3_alive_0p1":     ("Phase3",  "alive=0.1"),
    "p3_alive_0p2":     ("Phase3",  "alive=0.2"),
    "p3_alive_0p3":     ("Phase3",  "alive=0.3"),
    "p3_alive_0p5":     ("Phase3",  "alive=0.5"),
    "p3_alive_1p0":     ("Phase3",  "alive=1.0"),
}

# ── データ読み込み ──────────────────────────────────────────────────────────────
records = []
for p in sorted(RESULTS_DIR.glob("*.json")):
    if p.stem not in RUN_META:
        continue
    data = json.loads(p.read_text())
    vx_by_cmd = {}
    for ph in data.get("phases", []):
        cv, mv = ph.get("cmd_vx", 0), ph.get("meas_vx", 0)
        cy = ph.get("cmd_yaw", 0)
        if abs(cv) > 1e-4 and abs(cy) < 1e-4:
            vx_by_cmd[round(cv, 4)] = mv
    meas = [vx_by_cmd.get(c, float("nan")) for c in CMD_VX_LEVELS]
    valid = [v for v in meas if not math.isnan(v)]
    mean_vx = float(np.mean(valid)) if valid else 0.0
    records.append({
        "name":    p.stem,
        "label":   RUN_META[p.stem][1],
        "phase":   RUN_META[p.stem][0],
        "meas":    meas,
        "mean_vx": mean_vx,
        "walking": mean_vx > MOVING_THRESHOLD,
    })

walking     = [r for r in records if     r["walking"]]
not_walking = [r for r in records if not r["walking"]]
print(f"歩いている: {len(walking)}件  歩いていない: {len(not_walking)}件")

# ── プロット ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

cmds = CMD_VX_LEVELS

# 歩いていない run（薄いグレー）
for r in not_walking:
    ax.plot(cmds, r["meas"], color="#aaaaaa", lw=1.5, ls="--",
            alpha=0.7, zorder=2)
    ax.annotate(r["label"], (cmds[-1], r["meas"][-1]),
                xytext=(4, 0), textcoords="offset points",
                fontsize=7, color="#888888", va="center")

# 歩いている run（オレンジ系）
cmap = plt.cm.get_cmap("YlOrRd")
n = max(len(walking) - 1, 1)
for i, r in enumerate(walking):
    color = cmap(0.4 + 0.55 * i / n)
    ax.plot(cmds, r["meas"], color=color, lw=2.0, alpha=0.85, zorder=3)
    ax.annotate(r["label"], (cmds[-1], r["meas"][-1]),
                xytext=(4, 0), textcoords="offset points",
                fontsize=7, color=color, va="center")

# 理想追従（対角線）
ax.plot(cmds, cmds, "k--", lw=1.5, alpha=0.35, label="Ideal tracking (meas=cmd)", zorder=1)

# 速度上限バンド
max_vxs_walk = [max(r["meas"]) for r in walking if not any(math.isnan(v) for v in r["meas"])]
if max_vxs_walk:
    lo, hi = min(max_vxs_walk) * 0.9, max(max_vxs_walk) * 1.1
    ax.axhspan(lo, hi, color="orange", alpha=0.10, zorder=0,
               label=f"Physical ceiling ({lo:.3f}–{hi:.3f} m/s)")

ax.set_xlabel("Command vx  [m/s]", fontsize=12)
ax.set_ylabel("Measured vx  [m/s]", fontsize=12)
ax.set_title(
    "Locomotion outcome is binary:\nwalk at fixed speed or stay still — regardless of command or parameter",
    fontsize=11, fontweight="bold"
)
ax.set_xlim(0, 0.175)
ax.set_ylim(-0.005, 0.175)
ax.set_xticks(cmds)
ax.grid(alpha=0.3)

walk_patch  = mpatches.Patch(color=cmap(0.65), label=f"Walking runs (n={len(walking)})  — flat ≈0.06–0.08 m/s")
still_patch = mpatches.Patch(color="#aaaaaa",  label=f"Stationary runs (n={len(not_walking)}) — flat ≈0 m/s")
ideal_line  = plt.Line2D([0], [0], color="black", ls="--", alpha=0.4, label="Ideal tracking")
ax.legend(handles=[walk_patch, still_patch, ideal_line], fontsize=9, loc="upper left")

plt.tight_layout()
out = Path(__file__).resolve().parent.parent / "binary_outcome_figure.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
