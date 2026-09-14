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
OFFSETS_JSON = PROJECT_DIR / "prove_vx_offsets.json"
OUT_PNG      = PROJECT_DIR / "prove_vx_failure.png"

CONDITIONS = [
    dict(sigma=0.01,  eval_name="p2a_siglin_0p01",
         ccr_cache="ccr_p2a_siglin_0p01.json",
         label="sigma_lin = 0.01",  color="#2980b9"),
    dict(sigma=0.025, eval_name=None,
         ccr_cache="command_tracking_results.json",
         label="sigma_lin = 0.025", color="#e67e22"),
    dict(sigma=0.05,  eval_name="p2a_siglin_0p05",
         ccr_cache="ccr_p2a_siglin_0p05.json",
         label="sigma_lin = 0.05",  color="#27ae60"),
]

def load_vx(cond):
    p = PROJECT_DIR / cond["ccr_cache"]
    if p.exists():
        d = json.loads(p.read_text())
        if "vx_sweep" in d:
            return [(r["cmd_vx"], r["meas_vx"]) for r in d["vx_sweep"]]
    if cond["eval_name"]:
        p = PROJECT_DIR / "eval_results" / "archived" / f"{cond['eval_name']}.json"
        d = json.loads(p.read_text())
        return [(ph["cmd_vx"], ph["meas_vx"]) for ph in d.get("phases", [])
                if abs(ph.get("cmd_vy", 0)) < 0.005 and abs(ph.get("cmd_yaw", 0)) < 0.005]
    raise FileNotFoundError(cond)

def load_offsets():
    return json.loads(OFFSETS_JSON.read_text()) if OFFSETS_JSON.exists() else {}

offsets = load_offsets()
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("vx 追従失敗", fontsize=18, fontweight="bold")
ann_map = {}
leg_map = {}

for col, cond in enumerate(CONDITIONS):
    clr, lbl, sigma = cond["color"], cond["label"], cond["sigma"]
    rows = load_vx(cond)
    cmds = np.array([r[0] for r in rows])
    meas = np.array([r[1] for r in rows])

    ax = axes[0][col]
    ax.scatter(cmds, meas, color=clr, s=90, zorder=5)
    m_ols, b_ols = np.polyfit(cmds, meas, 1)
    xf = np.linspace(-0.17, 0.17, 300)
    ax.plot(xf, m_ols * xf + b_ols, color=clr, lw=2, label=f"OLS slope={m_ols:.3f}")
    ax.plot([-0.17, 0.17], [-0.17, 0.17], "k--", lw=1, alpha=0.25, label="完全追従")
    ax.axhline(0, color="gray", lw=0.6, alpha=0.4)
    ax.axvline(0, color="gray", lw=0.6, alpha=0.4)
    ax.set_xlim(-0.18, 0.18); ax.set_ylim(-0.10, 0.18)
    ax.set_xlabel("Command vx [m/s]", fontsize=13)
    ax.set_ylabel("Measured vx [m/s]", fontsize=13)
    ax.set_title(f"{lbl}\nOLS slope = {m_ols:.3f}   (理想 = 1.00)",
                 fontweight="bold", fontsize=13)
    lp = offsets.get("legend_pos", {}).get(str(col))
    if lp:
        leg = ax.legend(fontsize=11, bbox_to_anchor=(lp["x"], lp["y"]), loc="lower left", borderaxespad=0)
    else:
        leg = ax.legend(fontsize=11, loc="upper left")
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
    e_range = np.linspace(0, 0.30, 600)
    r_curve = np.exp(-e_range**2 / sigma)
    ax2.plot(e_range, r_curve, color=clr, lw=2.5)
    ax2.set_facecolor("#fff9f9")
    obs_e = float(np.sqrt(np.mean((cmds - meas)**2)))
    r_obs = float(np.exp(-obs_e**2 / sigma))
    grad  = -2.0 * obs_e / sigma * r_obs
    ax2.scatter([obs_e], [r_obs], color="red", s=130, zorder=6)
    e_tan = np.array([max(0, obs_e - 0.05), min(0.30, obs_e + 0.05)])
    ax2.plot(e_tan, r_obs + grad * (e_tan - obs_e), "r-", lw=1.5, alpha=0.8)
    key_obs = f"bot_{col}_obs"
    off_obs = offsets.get(key_obs, {"dx": 12.0, "dy": 10.0})
    ann_obs = ax2.annotate(
        f"RMS 誤差 = {obs_e:.3f} m/s\n|dr/de| = {abs(grad):.2f}",
        xy=(obs_e, r_obs), xytext=(off_obs["dx"], off_obs["dy"]),
        textcoords="offset points", fontsize=11, color="red",
        arrowprops=dict(arrowstyle="->", color="red", lw=1.2))
    ann_map[key_obs] = ann_obs
    ax2.set_xlim(0, 0.30); ax2.set_ylim(-0.05, 1.15)
    ax2.set_xlabel("速度誤差 e = |cmd - meas| [m/s]", fontsize=13)
    ax2.set_ylabel("追従報酬  r(e)", fontsize=13)
    ax2.set_title(f"{lbl}\n|dr/de| = {abs(grad):.2f}  at  e = {obs_e:.3f}", fontsize=13)
    ax2.grid(alpha=0.3)
    ax2.text(0.97, 0.95, f"r(e) = exp(-e^2/{sigma})",
             transform=ax2.transAxes, ha="right", va="top", fontsize=11, color=clr)

plt.tight_layout(rect=[0, 0.01, 1, 0.97])

if args.edit:
    _drags = [ann.draggable(True) for ann in ann_map.values()]  # noqa
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
else:
    plt.savefig(OUT_PNG, dpi=150, bbox_inches="tight")
    print(f"PNG 保存: {OUT_PNG}")
