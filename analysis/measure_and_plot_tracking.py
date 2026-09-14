# -*- coding: utf-8 -*-
"""
measure_and_plot_tracking.py

使い方:
  python3 measure_and_plot_tracking.py               # 測定 → PNG保存
  python3 measure_and_plot_tracking.py --use-cache   # 再測定なし → PNG保存
  python3 measure_and_plot_tracking.py --edit        # GUIでラベル位置を手動調整
                                                     # ウィンドウを閉じると位置保存+PNG出力
"""
import argparse, json, re, subprocess
from pathlib import Path
import numpy as np

PROJECT_DIR   = Path(__file__).resolve().parent.parent
PYTHON        = PROJECT_DIR / "Open_Duck_Playground/.venv/bin/python"
RESULT_JSON   = PROJECT_DIR / "command_tracking_results.json"
OFFSETS_JSON  = PROJECT_DIR / "annotation_offsets.json"
OUTPUT_PNG    = PROJECT_DIR / "command_tracking_figure.png"


def get_onnx(run_json_rel: str) -> Path:
    p = PROJECT_DIR / run_json_rel
    d = json.loads(p.read_text())
    return Path(d["onnx"])


def run_sweep(onnx: Path, axes: str, fractions: str) -> str:
    cmd = [
        str(PYTHON), "-m", "thing_test.check_command_response",
        "-o", str(onnx),
        "--vx_max", "0.15", "--vy_max", "0.2", "--yaw_max", "1.0",
        f"--fractions={fractions}",
        "--axes", axes,
        "--phase_duration", "4.0",
    ]
    print(f"\n[RUN] axes={axes}  fractions={fractions}")
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout + "\n" + result.stderr


def parse_summary(output: str) -> list:
    records = []
    in_sum = False
    for line in output.splitlines():
        if "=== Summary:" in line:
            in_sum = True
            continue
        if not in_sum:
            continue
        m = re.search(
            r"cmd_vx=([+-][\d.]+).*meas_vx=([+-][\d.]+)"
            r".*cmd_vy=([+-][\d.]+).*meas_vy=([+-][\d.]+)"
            r".*cmd_yaw=([+-][\d.]+).*meas_yaw=([+-][\d.]+)",
            line,
        )
        if m:
            records.append({
                "cmd_vx":  float(m.group(1)), "meas_vx": float(m.group(2)),
                "cmd_vy":  float(m.group(3)), "meas_vy": float(m.group(4)),
                "cmd_yaw": float(m.group(5)), "meas_yaw":float(m.group(6)),
            })
    return records


def measure(onnx: Path) -> dict:
    FRACS = "-1.0,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1.0"
    return {
        "onnx":      str(onnx),
        "vx_sweep":  parse_summary(run_sweep(onnx, "vx",  FRACS)),
        "yaw_sweep": parse_summary(run_sweep(onnx, "yaw", FRACS)),
    }


def _default_offset(axis, c, m, my=None, by=None, mean_vx=None, below_idx=None):
    """保存済み位置がない場合のデフォルトオフセット"""
    if axis == "yaw":
        if abs(c) >= 0.999:
            return (8, 8) if c < 0 else (-50, -14)
        above = m > my * c + by
        if c < -0.4:
            return (8, 8) if above else (8, -14)
        return (-50, 8) if above else (8, -14)
    else:  # vx
        if m >= mean_vx:
            return (-50 if c > 0.12 else 4, 14)
        else:
            return (4 if below_idx % 2 == 0 else -50, -16)


def build_figure(data: dict, edit_mode: bool):
    import matplotlib
    matplotlib.use("TkAgg" if edit_mode else "Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    saved_offsets = {}
    if OFFSETS_JSON.exists():
        saved_offsets = json.loads(OFFSETS_JSON.read_text())

    vx_rows  = data["vx_sweep"]
    yaw_rows = data["yaw_sweep"]
    vx_cmd   = [r["cmd_vx"]   for r in vx_rows]
    vx_meas  = [r["meas_vx"]  for r in vx_rows]
    yaw_cmd  = [r["cmd_yaw"]  for r in yaw_rows]
    yaw_meas = [r["meas_yaw"] for r in yaw_rows]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Command Tracking: Yaw vs Linear Velocity (p3_alive_0p5)",
                 fontsize=13, fontweight="bold")

    annotations = []  # (key, annotation_obj)

    # ── yaw ────────────────────────────────────────────────────────────────
    ax = axes[0]
    lim = 1.15
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=1.5, alpha=0.4)
    ax.axhline(0, color="gray", lw=0.8, alpha=0.3)
    ax.axvline(0, color="gray", lw=0.8, alpha=0.3)

    yc = np.array([c for c, m in zip(yaw_cmd, yaw_meas) if abs(c) < 0.999])
    ym = np.array([m for c, m in zip(yaw_cmd, yaw_meas) if abs(c) < 0.999])
    my, by = np.polyfit(yc, ym, 1)
    xs = np.linspace(-lim, lim, 200)
    ax.plot(xs, my * xs + by, color="#27ae60", lw=2.5, zorder=4)

    for c, m in zip(yaw_cmd, yaw_meas):
        ok = abs(c) < 0.999
        ax.scatter(c, m, color="#e67e22" if ok else "#e74c3c",
                   marker="o" if ok else "x", s=100, zorder=5)
        key = f"yaw_{c:+.3f}"
        dx, dy = saved_offsets.get(key, _default_offset("yaw", c, m, my=my, by=by))
        ann = ax.annotate(f"{m:+.3f}", (c, m), textcoords="offset points",
                          xytext=(dx, dy), fontsize=8)
        if edit_mode:
            ann.draggable(True)
        annotations.append((key, ann))

    ax.legend(handles=[
        plt.Line2D([0],[0], ls="--", color="k",       alpha=0.4, label="Ideal (slope=1)"),
        plt.Line2D([0],[0], ls="-",  color="#27ae60", lw=2.5,
                   label=f"OLS: slope={my:.2f}, intercept={by:+.3f}"),
        mpatches.Patch(color="#e67e22", label="Tracked (|cmd|<1.0)"),
        mpatches.Patch(color="#e74c3c", label="Collapsed (|cmd|=1.0)"),
    ], fontsize=8, loc="lower right")
    ax.set_xlabel("Command yaw  [rad/s]", fontsize=11)
    ax.set_ylabel("Measured yaw  [rad/s]", fontsize=11)
    ax.set_title("Yaw  ✓ Tracked", fontsize=11, color="#27ae60", fontweight="bold")
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect("equal"); ax.grid(alpha=0.3)

    # ── vx ─────────────────────────────────────────────────────────────────
    ax = axes[1]
    lim_c = 0.18
    ax.plot([-lim_c, lim_c], [-lim_c, lim_c], "k--", lw=1.5, alpha=0.4, label="Ideal (slope=1)")
    ax.axhline(0, color="gray", lw=0.8, alpha=0.3)
    ax.axvline(0, color="gray", lw=0.8, alpha=0.3)
    mean_vx = float(np.mean(vx_meas))
    ax.axhline(mean_vx, color="#2980b9", ls="--", lw=1.5, alpha=0.6,
               label=f"Mean = {mean_vx:.3f} m/s")

    vc = np.array(vx_cmd); vm = np.array(vx_meas)
    mv, bv = np.polyfit(vc, vm, 1)
    xs = np.linspace(-lim_c, lim_c, 200)
    ax.plot(xs, mv * xs + bv, color="#e74c3c", lw=2.5, zorder=4,
            label=f"OLS: slope={mv:.2f}, intercept={bv:+.3f}")

    below_idx = 0
    for c, m in zip(vx_cmd, vx_meas):
        ax.scatter(c, m, color="#2980b9", s=100, zorder=5)
        key = f"vx_{c:+.3f}"
        dx, dy = saved_offsets.get(key, _default_offset(
            "vx", c, m, mean_vx=mean_vx,
            below_idx=(below_idx if m < mean_vx else None)
        ))
        if m < mean_vx:
            below_idx += 1
        ann = ax.annotate(f"{m:+.3f}", (c, m), textcoords="offset points",
                          xytext=(dx, dy), fontsize=8)
        if edit_mode:
            ann.draggable(True)
        annotations.append((key, ann))

    ax.legend(fontsize=8, loc="upper left")
    ax.set_xlabel("Command vx  [m/s]", fontsize=11)
    ax.set_ylabel("Measured vx  [m/s]", fontsize=11)
    ax.set_title("Linear Velocity  ✗ Not Tracked", fontsize=11,
                 color="#e74c3c", fontweight="bold")
    ax.set_xlim(-lim_c, lim_c); ax.set_ylim(-0.05, lim_c)
    ax.grid(alpha=0.3)

    plt.tight_layout()

    if edit_mode:
        def on_close(event):
            offsets = {key: list(ann.xyann) for key, ann in annotations}
            OFFSETS_JSON.write_text(json.dumps(offsets, indent=2))
            print(f"Saved offsets: {OFFSETS_JSON}")
            fig.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
            print(f"Saved: {OUTPUT_PNG}")

        fig.canvas.mpl_connect("close_event", on_close)
        if edit_mode:
            print("\n[EDIT] ラベルをドラッグして位置を調整 → ウィンドウを閉じると保存")
        plt.show()
    else:
        plt.savefig(OUTPUT_PNG, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {OUTPUT_PNG}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-json", default="eval_results/archived/p3_alive_0p5.json")
    ap.add_argument("--use-cache", action="store_true", help="再測定なし")
    ap.add_argument("--edit", action="store_true",    help="GUIでラベル位置を手動調整")
    args = ap.parse_args()

    if (args.use_cache or args.edit) and RESULT_JSON.exists():
        print(f"[CACHE] {RESULT_JSON} を使用")
        data = json.loads(RESULT_JSON.read_text())
    else:
        onnx = get_onnx(args.run_json)
        print(f"ONNX: {onnx}")
        data = measure(onnx)
        RESULT_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False))
        print(f"Saved: {RESULT_JSON}")

    build_figure(data, edit_mode=args.edit)


if __name__ == "__main__":
    main()
