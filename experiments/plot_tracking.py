#!/usr/bin/env python3
"""
ポスター発表用 速度追従性能 解析・可視化スクリプト

使い方:
  python experiments/plot_tracking.py
  python experiments/plot_tracking.py --csv eval_results/sigma_lin_0p025.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("[ERROR] matplotlib が見つかりません")
    sys.exit(1)

_JP_FONTS = ["IPAexGothic", "Noto Sans CJK JP", "Hiragino Sans", "Yu Gothic"]
_USE_JP = False

def _setup_font():
    global _USE_JP
    import matplotlib.font_manager as fm
    available = {f.name for f in fm.fontManager.ttflist}
    for font in _JP_FONTS:
        if font in available:
            matplotlib.rcParams["font.family"] = font
            _USE_JP = True
            break
    matplotlib.rcParams["axes.unicode_minus"] = False

_setup_font()

def _t(ja, en):
    return ja if _USE_JP else en

def filename_to_label(stem):
    s = stem.replace("p5_", "")
    s = re.sub(r"(?<=[a-z_])m(\d)", r"-\1", s)
    s = re.sub(r"(\d)p(\d)", r"\1.\2", s)
    return s.replace("_", " ").strip()

def load_json_results(results_dir):
    results = []
    for p in sorted(results_dir.glob("*.json")):
        try:
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
            if "rmse_vx" not in data or "phases" not in data:
                continue
            data["_label"] = filename_to_label(p.stem)
            data["_stem"] = p.stem
            results.append(data)
        except Exception as e:
            print(f"[WARN] {p.name}: {e}")
    return results

def plot_rmse_comparison(results, out_path, dpi):
    labels   = [r["_label"]   for r in results]
    rmse_vx  = [r["rmse_vx"]  for r in results]
    rmse_yaw = [r["rmse_yaw"] for r in results]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.6 + 1), 5))
    b1 = ax.bar(x - w/2, rmse_vx,  w, label="RMSE_vx [m/s]",    color="#1E88E5", alpha=0.88)
    b2 = ax.bar(x + w/2, rmse_yaw, w, label="RMSE_yaw [rad/s]", color="#E53935", alpha=0.88)
    for bar in (*b1, *b2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.002, f"{h:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xlabel(_t("実験条件", "Experiment"), fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.set_title(_t("コマンド追従性能 RMSE 比較", "Command Tracking RMSE"), fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path.name}")

def plot_cmd_vs_meas_scatter(results, out_path, dpi):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(results), 1)))
    for res, color in zip(results, colors):
        phases = res["phases"]
        kw = dict(s=70, alpha=0.85, edgecolors="white", linewidths=0.5, zorder=3)
        axes[0].scatter([p["cmd_vx"]  for p in phases], [p["meas_vx"]  for p in phases],
                        color=color, label=res["_label"], **kw)
        axes[1].scatter([p["cmd_yaw"] for p in phases], [p["meas_yaw"] for p in phases],
                        color=color, label=res["_label"], **kw)
    for ax in axes:
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0]) - 0.01
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1]) + 0.01
        ax.plot([lo, hi], [lo, hi], "k--", lw=1.2, alpha=0.4,
                label=_t("理想 (cmd=meas)", "Ideal"), zorder=1)
        ax.axhline(0, color="gray", lw=0.5, alpha=0.4)
        ax.axvline(0, color="gray", lw=0.5, alpha=0.4)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper left")
    axes[0].set_xlabel(_t("指令 vx [m/s]",    "Command vx [m/s]"),   fontsize=12)
    axes[0].set_ylabel(_t("計測 vx [m/s]",    "Measured vx [m/s]"),  fontsize=12)
    axes[0].set_title( _t("前進速度 追従性能", "Forward Vel. Tracking"), fontsize=13, fontweight="bold")
    axes[1].set_xlabel(_t("指令 yaw [rad/s]",  "Command yaw [rad/s]"),  fontsize=12)
    axes[1].set_ylabel(_t("計測 yaw [rad/s]",  "Measured yaw [rad/s]"), fontsize=12)
    axes[1].set_title( _t("旋回速度 追従性能", "Yaw Rate Tracking"),      fontsize=13, fontweight="bold")
    fig.suptitle(_t("速度コマンド追従性能 (理想は対角線)", "Velocity Tracking (ideal: diagonal)"),
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path.name}")

def plot_phase_comparison(result, out_path, dpi):
    phases  = result["phases"]
    xlabels = [p["phase"] for p in phases]
    x = np.arange(len(xlabels))
    w = 0.35
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    fig.suptitle(
        _t(f"フェーズ別追従: {result['_label']}\nRMSE_vx={result['rmse_vx']:.4f} m/s   RMSE_yaw={result['rmse_yaw']:.4f} rad/s",
           f"Phase Tracking: {result['_label']}\nRMSE_vx={result['rmse_vx']:.4f} m/s   RMSE_yaw={result['rmse_yaw']:.4f} rad/s"),
        fontsize=12, fontweight="bold")
    ax1.bar(x - w/2, [p["cmd_vx"]  for p in phases], w, label=_t("指令値","Command"), color="#90CAF9", alpha=0.9)
    ax1.bar(x + w/2, [p["meas_vx"] for p in phases], w, label=_t("計測値","Measured"), color="#1565C0", alpha=0.9)
    ax1.set_ylabel(_t("前進速度 vx [m/s]", "vx [m/s]"), fontsize=11)
    ax1.legend(fontsize=10); ax1.axhline(0, color="black", lw=0.5); ax1.grid(axis="y", alpha=0.3)
    ax2.bar(x - w/2, [p["cmd_yaw"]  for p in phases], w, label=_t("指令値","Command"), color="#FFCCBC", alpha=0.9)
    ax2.bar(x + w/2, [p["meas_yaw"] for p in phases], w, label=_t("計測値","Measured"), color="#BF360C", alpha=0.9)
    ax2.set_ylabel(_t("旋回速度 yaw [rad/s]", "yaw [rad/s]"), fontsize=11)
    ax2.set_xlabel(_t("フェーズ", "Phase"), fontsize=11)
    ax2.set_xticks(x); ax2.set_xticklabels(xlabels, rotation=30, ha="right", fontsize=9)
    ax2.legend(fontsize=10); ax2.axhline(0, color="black", lw=0.5); ax2.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path.name}")

def plot_overlay_phase(results, out_path, dpi):
    if not results:
        return
    ref_phases = [p["phase"] for p in results[0]["phases"]]
    x = np.arange(len(ref_phases))
    n = len(results)
    w = 0.8 / (n + 1)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(10, len(ref_phases) * 0.9), 7), sharex=True)
    ax1.bar(x, [p["cmd_vx"]  for p in results[0]["phases"]], 0.8,
            label=_t("指令値","Command"), color="#BDBDBD", alpha=0.7, zorder=1)
    ax2.bar(x, [p["cmd_yaw"] for p in results[0]["phases"]], 0.8,
            label=_t("指令値","Command"), color="#BDBDBD", alpha=0.7, zorder=1)
    colors = plt.cm.tab10(np.linspace(0, 1, n))
    for i, (res, color) in enumerate(zip(results, colors)):
        offset = (i - (n-1)/2) * w
        ax1.bar(x + offset, [p["meas_vx"]  for p in res["phases"]], w*0.9,
                color=color, alpha=0.85, label=res["_label"], zorder=2)
        ax2.bar(x + offset, [p["meas_yaw"] for p in res["phases"]], w*0.9,
                color=color, alpha=0.85, label=res["_label"], zorder=2)
    for ax, ylabel in [(ax1, _t("vx [m/s]","vx [m/s]")), (ax2, _t("yaw [rad/s]","yaw [rad/s]"))]:
        ax.axhline(0, color="black", lw=0.5); ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel(ylabel, fontsize=11); ax.legend(fontsize=8, loc="upper left")
    ax2.set_xticks(x); ax2.set_xticklabels(ref_phases, rotation=30, ha="right", fontsize=9)
    ax2.set_xlabel(_t("フェーズ","Phase"), fontsize=11)
    fig.suptitle(_t("全実験 フェーズ別比較 (グレー=指令値)", "All Experiments (gray=command)"),
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path.name}")

def plot_timeseries(csv_path, out_path, dpi):
    rows = []
    with open(csv_path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append({k: (float(v) if k != "phase" else v) for k, v in row.items()})
    if not rows:
        print(f"  [SKIP] empty: {csv_path.name}"); return
    times    = [r["time"]          for r in rows]
    cmd_vx   = [r["cmd_vx"]        for r in rows]
    inst_vx  = [r["inst_vx"]       for r in rows]
    cmd_yaw  = [r["cmd_yaw"]       for r in rows]
    inst_yaw = [r["inst_yaw_rate"] for r in rows]
    phase_labels = [r["phase"]     for r in rows]
    boundaries = [0]
    for j in range(1, len(phase_labels)):
        if phase_labels[j] != phase_labels[j-1]:
            boundaries.append(j)
    boundaries.append(len(rows))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    fig.suptitle(_t(f"速度追従 時系列 — {csv_path.stem}", f"Velocity Tracking — {csv_path.stem}"),
                 fontsize=12, fontweight="bold")
    ax1.plot(times, cmd_vx,  "b--", lw=1.8, label=_t("指令値","Command"), alpha=0.9, zorder=3)
    ax1.plot(times, inst_vx, "b-",  lw=1.0, label=_t("計測値","Measured"), alpha=0.7, zorder=2)
    ax1.fill_between(times, cmd_vx, inst_vx, alpha=0.12, color="blue")
    ax1.set_ylabel(_t("前進速度 vx [m/s]","vx [m/s]"), fontsize=11)
    ax1.axhline(0, color="black", lw=0.5); ax1.legend(fontsize=10); ax1.grid(alpha=0.3)
    ax2.plot(times, cmd_yaw,  "r--", lw=1.8, label=_t("指令値","Command"), alpha=0.9, zorder=3)
    ax2.plot(times, inst_yaw, "r-",  lw=1.0, label=_t("計測値","Measured"), alpha=0.7, zorder=2)
    ax2.fill_between(times, cmd_yaw, inst_yaw, alpha=0.12, color="red")
    ax2.set_ylabel(_t("旋回速度 yaw [rad/s]","yaw [rad/s]"), fontsize=11)
    ax2.set_xlabel(_t("時間 [s]","Time [s]"), fontsize=11)
    ax2.axhline(0, color="black", lw=0.5); ax2.legend(fontsize=10); ax2.grid(alpha=0.3)
    unique_phases = [phase_labels[i] for i in boundaries[:-1]]
    for j, (si, label) in enumerate(zip(boundaries[:-1], unique_phases)):
        ei = boundaries[j+1]
        t_mid = (times[si] + times[ei-1]) / 2
        for ax in [ax1, ax2]:
            ax.axvline(times[si], color="gray", lw=0.6, alpha=0.4, ls=":")
        ax1.text(t_mid, ax1.get_ylim()[1], label, ha="center", va="top",
                 fontsize=7, alpha=0.65, clip_on=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out_path.name}")

def save_summary_csv(results, out_path):
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stem", "label", "rmse_vx", "rmse_yaw"])
        for r in results:
            w.writerow([r["_stem"], r["_label"], f"{r['rmse_vx']:.6f}", f"{r['rmse_yaw']:.6f}"])
    print(f"  [OK] {out_path.name}")

def main():
    parser = argparse.ArgumentParser(description="ポスター用 速度追従解析スクリプト")
    parser.add_argument("--results_dir", default="eval_results")
    parser.add_argument("--out_dir",     default="poster_figures")
    parser.add_argument("--dpi",  type=int, default=300)
    parser.add_argument("--csv",  type=str, default=None)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== plot_tracking.py ===")
    print(f"  入力: {results_dir}/   出力: {out_dir}/  DPI={args.dpi}")
    print(f"  日本語フォント: {'あり' if _USE_JP else 'なし (英語ラベルで出力)'}")

    results = load_json_results(results_dir)
    if results:
        print(f"\n{len(results)} 件読み込み:")
        for r in results:
            print(f"  {r['_stem']:35s}  RMSE_vx={r['rmse_vx']:.4f}  RMSE_yaw={r['rmse_yaw']:.4f}")
        print("\n--- 図を生成 ---")
        plot_rmse_comparison(    results, out_dir / "rmse_comparison.png",     args.dpi)
        plot_cmd_vs_meas_scatter(results, out_dir / "cmd_vs_meas_scatter.png", args.dpi)
        plot_overlay_phase(      results, out_dir / "overlay_phase.png",       args.dpi)
        save_summary_csv(        results, out_dir / "summary_table.csv")
        for res in results:
            plot_phase_comparison(res, out_dir / f"phase_{res['_stem']}.png", args.dpi)
    else:
        print(f"[WARN] {results_dir}/ に JSON が見つかりません")

    csv_files = [Path(args.csv)] if args.csv else [p for p in sorted(results_dir.glob("*.csv")) if p.stem != "tracking_comparison"]
    if csv_files:
        print(f"\n--- 時系列プロット ({len(csv_files)} 件) ---")
        for cp in csv_files:
            if cp.exists():
                plot_timeseries(cp, out_dir / f"timeseries_{cp.stem}.png", args.dpi)

    print(f"\n完了: {out_dir}/")
    for fp in sorted(out_dir.glob("*.png")):
        print(f"  {fp.name:45s} {fp.stat().st_size // 1024:5d} KB")

if __name__ == "__main__":
    main()
