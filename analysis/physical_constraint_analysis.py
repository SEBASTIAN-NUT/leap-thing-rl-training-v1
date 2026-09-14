import json, math, sys
import numpy as np
from pathlib import Path

RESULTS_DIR = Path("eval_results/archived")
RUN_META = {
    "p2a_siglin_0p01":  ("Phase2a", "sigma_lin", 0.010),
    "p2a_siglin_0p05":  ("Phase2a", "sigma_lin", 0.050),
    "p2b_sigang_0p25":  ("Phase2b", "sigma_ang", 0.250),
    "p2b_sigang_0p5":   ("Phase2b", "sigma_ang", 0.500),
    "p2b_sigang_1p0":   ("Phase2b", "sigma_ang", 1.000),
    "p2b_sigang_2p0":   ("Phase2b", "sigma_ang", 2.000),
    "p3_alive_0p1":     ("Phase3",  "alive",     0.100),
    "p3_alive_0p2":     ("Phase3",  "alive",     0.200),
    "p3_alive_0p3":     ("Phase3",  "alive",     0.300),
    "p3_alive_0p5":     ("Phase3",  "alive",     0.500),
    "p3_alive_1p0":     ("Phase3",  "alive",     1.000),
}
PHASE_COLORS = {"Phase2a": "#2980b9", "Phase2b": "#e67e22", "Phase3": "#27ae60"}
MOVING_THRESHOLD = 0.02
CMD_VX_LEVELS = [0.0375, 0.0750, 0.1125, 0.1500]

runs = {}
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
    runs[p.stem] = {
        "vx": vx_by_cmd,
        "rmse_vx": data.get("rmse_vx", float("nan")),
        **dict(zip(("phase","param_name","param_value"), RUN_META[p.stem])),
    }

rows = []
for name, r in runs.items():
    vx_vals = [r["vx"].get(c, float("nan")) for c in CMD_VX_LEVELS]
    valid = [v for v in vx_vals if not math.isnan(v)]
    max_vx  = max(valid) if valid else float("nan")
    mean_vx = float(np.mean(valid)) if valid else float("nan")
    gain_cv = float(np.std(valid)/mean_vx) if mean_vx > 1e-4 else float("nan")
    rows.append({
        "name": name, "phase": r["phase"],
        "param_name": r["param_name"], "param_value": r["param_value"],
        "max_vx": max_vx, "mean_vx": mean_vx, "gain_cv": gain_cv,
        "moving": mean_vx > MOVING_THRESHOLD,
        "vx_vals": vx_vals, "rmse_vx": r["rmse_vx"],
    })

moving = [r for r in rows if r["moving"]]
print("="*70)
print("【証拠1】パラメータ値 vs 最大達成速度")
print(f"  {'run':<22} {'param':>10} {'max_vx':>8} {'moving':>7}")
print("  "+"-"*52)
for r in rows:
    print(f"  {r['name']:<22} {r['param_value']:>10.3f} {r['max_vx']:>8.4f} {'YES' if r['moving'] else ' NO':>7}")

if moving:
    mvs = [r["max_vx"] for r in moving]
    print(f"\n  動作 run ({len(moving)}件) max_vx: 平均={np.mean(mvs):.4f}  std={np.std(mvs):.4f}")
    print(f"  コマンド上限 0.15 m/s に対する達成率: {np.mean(mvs)/0.15*100:.1f}%")

print("\n【証拠2】パラメータ vs 速度の相関係数")
for phase in ["Phase2a","Phase2b","Phase3"]:
    pr = [r for r in rows if r["phase"]==phase]
    xs = [r["param_value"] for r in pr]
    ys = [r["max_vx"] for r in pr]
    corr = float(np.corrcoef(xs,ys)[0,1]) if np.std(ys)>1e-6 else float("nan")
    print(f"  {phase}: r = {corr:+.3f}")

print("\n【証拠3】gain 変動係数 (0に近い = コマンドを無視した固定ゲイト)")
for r in rows:
    if r["moving"]:
        print(f"  {r['name']:<22} CV={r['gain_cv']:.3f}  vx@各cmd={[f'{v:.3f}' for v in r['vx_vals']]}")
    else:
        print(f"  {r['name']:<22} (静止)")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Physical Constraint Analysis: Parameter Sweeps Had No Effect on Locomotion Capability", fontsize=11, fontweight="bold")

ax = axes[0]
for r in rows:
    c = PHASE_COLORS.get(r["phase"], "gray")
    ls = "-" if r["moving"] else ":"
    ax.plot(CMD_VX_LEVELS, r["vx_vals"], ls=ls, color=c, lw=1.8, alpha=0.8, label=r["name"])
ax.plot(CMD_VX_LEVELS, CMD_VX_LEVELS, "k--", lw=1.5, alpha=0.4, label="Ideal (gain=1)")
if moving:
    mvs = [max(r["vx_vals"]) for r in moving]
    ax.axhspan(min(mvs)*0.85, max(mvs)*1.15, color="orange", alpha=0.15, label="Physical ceiling")
ax.set_xlabel("Command vx [m/s]"); ax.set_ylabel("Measured vx [m/s]")
ax.set_title("(1) Gain Curve: All Runs\n(Flat = No Command Conditioning)")
ax.legend(fontsize=6); ax.grid(alpha=0.3)

ax = axes[1]
for phase, color in PHASE_COLORS.items():
    pr = [r for r in rows if r["phase"]==phase]
    xs = [r["param_value"] for r in pr]
    ys = [r["max_vx"] for r in pr]
    ax.scatter(xs, ys, color=color, s=80, label=phase, zorder=3)
    for r in pr:
        ax.annotate(r["name"].split("_")[-1], (r["param_value"], r["max_vx"]),
                    textcoords="offset points", xytext=(4,4), fontsize=7)
ax.axhline(0.15, color="red", ls="--", lw=1, alpha=0.5, label="cmd_max=0.15")
if moving:
    ax.axhspan(min(mvs)*0.85, max(mvs)*1.15, color="orange", alpha=0.15, label="Ceiling")
ax.set_xlabel("Hyperparameter Value"); ax.set_ylabel("Max Achieved vx [m/s]")
ax.set_title("(2) Hyperparameter vs Velocity\n(r≈0 → No Correlation)")
ax.legend(fontsize=8); ax.grid(alpha=0.3); ax.set_ylim(-0.01, 0.18)

ax = axes[2]
names = [r["name"] for r in rows]
max_vxs = [r["max_vx"] for r in rows]
colors = [PHASE_COLORS.get(r["phase"],"gray") for r in rows]
ax.bar(range(len(names)), max_vxs, color=colors, alpha=0.8)
ax.axhline(0.15, color="red", ls="--", lw=1.5, alpha=0.6, label="cmd_max=0.15")
if moving:
    ax.axhline(np.mean(mvs), color="orange", ls="-", lw=2,
               label=f"Moving avg={np.mean(mvs):.3f}")
ax.set_xticks(range(len(names)))
ax.set_xticklabels([n.replace("p2a_siglin_","siglin\n").replace("p2b_sigang_","sigang\n").replace("p3_alive_","alive\n") for n in names],
                   rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Max Achieved vx [m/s]")
ax.set_title("(3) Max Velocity per Run\n(All Bounded by Physical Ceiling)")
ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3); ax.set_ylim(0, 0.18)

from matplotlib.patches import Patch
fig.legend([Patch(color=c,label=p) for p,c in PHASE_COLORS.items()],
           [p for p in PHASE_COLORS], loc="lower center", ncol=3,
           fontsize=9, bbox_to_anchor=(0.5,-0.05))
plt.tight_layout(rect=[0,0.05,1,1])
plt.savefig("physical_constraint_analysis.png", dpi=150, bbox_inches="tight")
print("\nSaved: physical_constraint_analysis.png")
