#!/usr/bin/env python3
import argparse, json, sys
import numpy as np
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--edit", action="store_true")
args = parser.parse_args()

if args.edit:
    import matplotlib; matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
import japanize_matplotlib
plt.rcParams.update({"xtick.labelsize": 11, "ytick.labelsize": 11})

PROJECT_DIR  = Path(__file__).resolve().parent.parent
OFFSETS_JSON = PROJECT_DIR / "prove_yaw_offsets.json"
OUT_PNG      = PROJECT_DIR / "prove_yaw_tracking.png"
SIGMA_ANG    = 0.25

CONDITIONS = [
    dict(eval_name="p2a_siglin_0p01",
         ccr_cache="ccr_p2a_siglin_0p01_yaw.json",
         label="sigma_lin = 0.01",  color="#2980b9"),
    dict(eval_name=None,
         ccr_cache="command_tracking_results.json",
         label="sigma_lin = 0.025", color="#e67e22"),
    dict(eval_name="p2a_siglin_0p05",
         ccr_cache="ccr_p2a_siglin_0p05_yaw.json",
         label="sigma_lin = 0.05",  color="#27ae60"),
]

def load_yaw(cond):
    p = PROJECT_DIR / cond["ccr_cache"]
    d = json.loads(p.read_text())
    sweep = d.get("yaw_sweep", [])
    if not sweep:
        raise KeyError(f"yaw_sweep not found in {p}")
    r0 = sweep[0]
    if "cmd_yaw" in r0:
        return [(r["cmd_yaw"], r["meas_yaw"]) for r in sweep]
    if "cmd" in r0:
        return [(r["cmd"], r["meas"]) for r in sweep]
    raise KeyError(f"Unknown keys: {list(r0.keys())}")

def load_offsets():
    return json.loads(OFFSETS_JSON.read_text()) if OFFSETS_JSON.exists() else {}

offsets = load_offsets()
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("yaw 追従性能", fontsize=18, fontweight="bold")
ann_map = {}
leg_map = {}

for col, cond in enumerate(CONDITIONS):
    clr, lbl = cond["color"], cond["label"]
    rows = load_yaw(cond)
    cmds = np.array([r[0] for r in rows])
    meas = np.array([r[1] for r in rows])

    # 崩壊点判定: |cmd|>0.05 かつ cmd と meas が逆符号
    collapse = np.array([abs(c) > 0.05 and abs(m) > 0.05 and c * m < 0 for c, m in zip(cmds, meas)])
    valid    = ~collapse

    ax = axes[0][col]
    ax.scatter(cmds[valid],    meas[valid],    color=clr, s=80, zorder=5, label="正常")
    ax.scatter(cmds[collapse], meas[collapse], color=clr, s=80, zorder=5,
               marker="x", linewidths=2, label="崩壊")

    if valid.sum() >= 2:
        m_ols, b_ols = np.polyfit(cmds[valid], meas[valid], 1)
    else:
        m_ols, b_ols = 0.0, 0.0
    xf = np.linspace(-1.1, 1.1, 300)
    ax.plot(xf, m_ols * xf + b_ols, color=clr, lw=2, label=f"OLS slope={m_ols:.3f}")
    ax.plot([-1.1, 1.1], [-1.1, 1.1], "k--", lw=1, alpha=0.25, label="完全追従")
    ax.axhline(0, color="gray", lw=0.6, alpha=0.4)
    ax.axvline(0, color="gray", lw=0.6, alpha=0.4)
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("Command yaw [rad/s]", fontsize=13)
    ax.set_ylabel("Measured yaw [rad/s]", fontsize=13)
    ax.set_title(f"{lbl}\nOLS slope = {m_ols:.3f}  (理想 = 1.00)",
                 fontweight="bold", fontsize=13)
    lp = offsets.get("legend_pos", {}).get(str(col))
    if lp:
        leg = ax.legend(fontsize=10, bbox_to_anchor=(lp["x"], lp["y"]), loc="lower left", borderaxespad=0)
    else:
        leg = ax.legend(fontsize=10, loc="upper left")
    leg_map[col] = (ax, leg)
    ax.grid(alpha=0.3)

    for i, (c, mv) in enumerate(zip(cmds, meas)):
        key = f"top_{col}_{i}"
        off = offsets.get(key, {"dx": 5.0, "dy": 5.0})
        ann = ax.annotate(f"{mv:+.3f}", xy=(c, mv),
                          xytext=(off["dx"], off["dy"]),
                          textcoords="offset points", fontsize=10, color=clr,
                          arrowprops=dict(arrowstyle="-", color=clr, lw=0.5, alpha=0.5))
        ann_map[key] = ann

    ax2 = axes[1][col]
    e_range = np.linspace(0, 2.5, 600)
    r_curve = np.exp(-e_range**2 / SIGMA_ANG)
    ax2.plot(e_range, r_curve, color=clr, lw=2.5)
    ax2.set_facecolor("#f9fff9")

    obs_e = float(np.sqrt(np.mean((cmds - meas)**2)))
    r_obs = float(np.exp(-obs_e**2 / SIGMA_ANG))
    grad  = -2.0 * obs_e / SIGMA_ANG * r_obs

    ax2.scatter([obs_e], [r_obs], color="red", s=130, zorder=6)
    e_tan = np.array([max(0, obs_e - 0.15), min(2.5, obs_e + 0.15)])
    ax2.plot(e_tan, r_obs + grad * (e_tan - obs_e), "r-", lw=1.5, alpha=0.8)

    key_obs = f"bot_{col}_obs"
    off_obs = offsets.get(key_obs, {"dx": 12.0, "dy": 10.0})
    ann_obs = ax2.annotate(
        f"RMS 誤差 = {obs_e:.3f} rad/s\n|dr/de| = {abs(grad):.2f}",
        xy=(obs_e, r_obs), xytext=(off_obs["dx"], off_obs["dy"]),
        textcoords="offset points", fontsize=11, color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
    ann_map[key_obs] = ann_obs

    ax2.set_xlim(0, 2.5); ax2.set_ylim(-0.05, 1.15)
    ax2.set_xlabel("速度誤差 e = |cmd - meas| [rad/s]", fontsize=13)
    ax2.set_ylabel("追従報酬  r(e)", fontsize=13)
    ax2.set_title(f"|dr/de| = {abs(grad):.2f}  at  e = {obs_e:.3f}", fontsize=13)
    ax2.grid(alpha=0.3)
    ax2.text(0.97, 0.95, f"r(e) = exp(-e^2/{SIGMA_ANG})",
             transform=ax2.transAxes, ha="right", va="top", fontsize=11, color=clr)

plt.tight_layout(rect=[0, 0.01, 1, 0.97])

if args.edit:
    _drags = [ann.draggable(True) for ann in ann_map.values()]
    for _, lr in leg_map.values(): lr.set_draggable(True)

    def on_close(event):
        data = {k: {"dx": a.xyann[0], "dy": a.xyann[1]} for k, a in ann_map.items()}
        try:
            renderer = fig.canvas.get_renderer()
            lpos = {}
            for c, (ax_r, lr) in leg_map.items():
                lw = lr.get_window_extent(renderer)
                aw = ax_r.get_window_extent(renderer)
                lpos[str(c)] = {"x": float((lw.x0-aw.x0)/aw.width), "y": float((lw.y0-aw.y0)/aw.height)}
            data["legend_pos"] = lpos
        except Exception: pass
        OFFSETS_JSON.write_text(json.dumps(data, indent=2))
        print(f"オフセット保存: {OFFSETS_JSON}")

    fig.canvas.mpl_connect("close_event", on_close)
    print("GUI 編集モード: ラベルをドラッグ -> ウィンドウを閉じると保存")
    plt.show()

    fig.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"PNG 保存: {OUT_PNG}")
else:
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"PNG 保存: {OUT_PNG}")
