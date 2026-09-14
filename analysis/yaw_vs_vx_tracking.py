# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── データ（check_command_response の実測値）────────────────────────────────
yaw_cmd  = [-1.000, -0.750, -0.500, -0.250,  0.000, +0.250, +0.500, +0.750, +1.000]
yaw_meas = [+0.647, -0.662, -0.594, -0.407, -0.117, +0.163, +0.484, +0.730, -0.476]

vx_cmd   = [-0.150, -0.112, -0.075, -0.037,  0.000, +0.037, +0.075, +0.112, +0.150]
vx_meas  = [+0.073, +0.067, +0.050, +0.067, +0.072, +0.060, +0.061, +0.069, +0.089]

# ── 色分け: ±100%は崩壊点、それ以外は正常 ─────────────────────────────────
yaw_ok   = [abs(c) < 1.0 for c in yaw_cmd]
vx_ok    = [True] * len(vx_cmd)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
fig.suptitle("Command Tracking: Yaw vs Linear Velocity", fontsize=13, fontweight="bold")

# ── 左: yaw ────────────────────────────────────────────────────────────────
ax = axes[0]
lim = 1.15
ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.5, alpha=0.4, label="Ideal (meas=cmd)")
ax.axhline(0, color="gray", lw=0.8, alpha=0.4)
ax.axvline(0, color="gray", lw=0.8, alpha=0.4)

for c, m, ok in zip(yaw_cmd, yaw_meas, yaw_ok):
    color = "#e67e22" if ok else "#e74c3c"
    marker = "o" if ok else "x"
    ms = 10 if ok else 12
    ax.scatter(c, m, color=color, marker=marker, s=ms**2, zorder=5)
    ax.annotate(f"{m:+.3f}", (c, m), textcoords="offset points",
                xytext=(6, 3), fontsize=8, color=color)

ok_patch  = mpatches.Patch(color="#e67e22", label="Normal (|cmd|<1.0)")
bad_patch = mpatches.Patch(color="#e74c3c", label="Collapsed (|cmd|=1.0)")
ax.legend(handles=[plt.Line2D([0],[0],ls="--",color="k",alpha=0.4,label="Ideal"),
                   ok_patch, bad_patch], fontsize=9)
ax.set_xlabel("Command yaw  [rad/s]", fontsize=11)
ax.set_ylabel("Measured yaw  [rad/s]", fontsize=11)
ax.set_title("Yaw Rate  ✓ TRACKED\n(sign and magnitude follow command)", fontsize=10, color="#27ae60")
ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
ax.grid(alpha=0.3); ax.set_aspect("equal")

# ── 右: vx ─────────────────────────────────────────────────────────────────
ax = axes[1]
lim_c = 0.18; lim_m = 0.18
ax.plot([-lim_c, lim_c], [-lim_c, lim_c], "k--", lw=1.5, alpha=0.4, label="Ideal (meas=cmd)")
ax.axhline(0, color="gray", lw=0.8, alpha=0.4)
ax.axvline(0, color="gray", lw=0.8, alpha=0.4)

mean_vx = float(np.mean(vx_meas))
ax.axhline(mean_vx, color="#2980b9", ls="-", lw=2, alpha=0.6,
           label=f"Actual mean = {mean_vx:.3f} m/s (constant)")

for c, m in zip(vx_cmd, vx_meas):
    ax.scatter(c, m, color="#2980b9", marker="o", s=100, zorder=5)
    ax.annotate(f"{m:+.3f}", (c, m), textcoords="offset points",
                xytext=(4, 4), fontsize=8, color="#2980b9")

ax.legend(fontsize=9)
ax.set_xlabel("Command vx  [m/s]", fontsize=11)
ax.set_ylabel("Measured vx  [m/s]", fontsize=11)
ax.set_title("Linear Velocity  ✗ NOT TRACKED\n(always ~+0.065 m/s regardless of command)", fontsize=10, color="#e74c3c")
ax.set_xlim(-lim_c, lim_c); ax.set_ylim(-0.05, lim_m)
ax.grid(alpha=0.3)

plt.tight_layout()
out = "yaw_vs_vx_tracking.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
