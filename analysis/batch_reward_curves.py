#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX がある全チェックポイントに extract_reward_curve.py を実行し、
フェーズ別ディレクトリに CSV と PNG を出力する。

出力先:
  analysis_output/reward_curves/
    phase2a_sigma_lin/
      sigma_lin_0.01/  curve.csv, curve.png
      sigma_lin_0.05/  ...
    phase2b_sigma_ang/
      sigma_ang_0.1/   ...
      sigma_ang_2.0/   ...
    phase3_alive/
      alive_0.1/       ...
      ...
"""
import csv, re, subprocess, sys
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CKPT         = ROOT / "results/checkpoints"
OUT  = ROOT / "analysis/output" / "reward_curves"

# フォルダ名 → (フェーズグループ, 条件ラベル) または (None, None)
# Phase 3 のみ transfer から取得 (p3_alive_* 命名のみ対象)
def classify(name: str):
    m = re.match(r"p2a_sigma_lin_(\S+?)_\d{8}", name)
    if m:
        v = m.group(1).replace("p", ".")
        return "phase2a_sigma_lin", f"sigma_lin_{v}"

    m = re.match(r"p2b_sigma_ang_(\S+?)_\d{8}", name)
    if m:
        v = m.group(1).replace("p", ".")
        return "phase2b_sigma_ang", f"sigma_ang_{v}"

    m = re.match(r"p3_alive_(\S+?)_\d{8}", name)
    if m:
        v = m.group(1).replace("p", ".")
        return "phase3_alive", f"alive_{v}"

    return None, None


def run_curve(ckpt_dir: Path, out_csv: Path):
    cmd = [sys.executable, str(ROOT / "extract_reward_curve.py"),
           str(ckpt_dir), "--out", str(out_csv)]
    result = subprocess.run(cmd, cwd=str(ROOT))
    return result.returncode == 0 and out_csv.exists()


def make_plot(csv_path: Path, title: str, out_png: Path):
    rows = list(csv.DictReader(open(csv_path)))
    if not rows:
        print(f"  [SKIP] データなし: {csv_path.name}")
        return

    steps = [int(r["step"]) / 1e6 for r in rows]
    total = [float(r["r_total"])   for r in rows]
    lin   = [float(r["r_lin_vel"]) for r in rows]
    ang   = [float(r["r_ang_vel"]) for r in rows]
    alive = [float(r["r_alive"])   for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    kw = dict(lw=1.8, ms=5)
    ax.plot(steps, total, "k-o",  **kw, label="Total")
    ax.plot(steps, lin,   color="#2980b9", marker="s", **kw, label="lin_vel")
    ax.plot(steps, ang,   color="#e67e22", marker="^", **kw, label="ang_vel")
    ax.plot(steps, alive, color="#27ae60", marker="v", **kw, label="alive")
    ax.set_xlabel("Timestep [Million steps]", fontsize=12)
    ax.set_ylabel("Reward (normalized)",      fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_png.relative_to(ROOT)}")


# ── main ─────────────────────────────────────────────────────────────────────
# ローカルと transfer の両方を収集し、同じ cond_label が重複したら後の日付を優先
candidates: dict[tuple, tuple] = {}  # (phase_group, cond_label) → (dir, date_str, onnx_count)

def collect(base: Path):
    if not base.exists():
        return
    for d in sorted(base.iterdir()):
        if not d.is_dir():
            continue
        onnx_count = len(list(d.glob("*.onnx")))
        if onnx_count == 0:
            continue
        pg, cl = classify(d.name)
        if pg is None:
            continue
        date_str = re.search(r"_(\d{8}_\d{6})$", d.name)
        date_str = date_str.group(1) if date_str else "00000000_000000"
        key = (pg, cl)
        if key not in candidates or date_str > candidates[key][1]:
            candidates[key] = (d, date_str, onnx_count)

collect(CKPT, from_transfer=False)

targets = [(d, pg, cl, n) for (pg, cl), (d, _, n) in sorted(candidates.items())]

print(f"\n対象: {len(targets)} チェックポイント")
for d, pg, cl, n in targets:
    print(f"  {pg}/{cl}  ({n} ONNX)  <- {d.name}")

print()
for ckpt_dir, phase_group, cond_label, _ in targets:
    out_dir = OUT / phase_group / cond_label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "curve.csv"
    out_png = out_dir / "curve.png"

    print(f"\n=== {phase_group}/{cond_label} ===")

    if out_csv.exists():
        print(f"  [CACHED] {out_csv.relative_to(ROOT)} — スキップ")
    else:
        ok = run_curve(ckpt_dir, out_csv)
        if not ok:
            print(f"  [FAILED] CCR 失敗: {ckpt_dir.name}")
            continue

    title = f"{phase_group.replace('_', ' ')} — {cond_label.replace('_', '=')}"
    make_plot(out_csv, title, out_png)

print("\n\n=== 完了 ===")
print(f"出力先: {OUT.relative_to(ROOT)}")
print("\nディレクトリ構成:")
for pg_dir in sorted(OUT.iterdir()):
    if not pg_dir.is_dir(): continue
    print(f"  {pg_dir.name}/")
    for cond_dir in sorted(pg_dir.iterdir()):
        if not cond_dir.is_dir(): continue
        files = [f.name for f in cond_dir.iterdir()]
        print(f"    {cond_dir.name}/  {files}")
