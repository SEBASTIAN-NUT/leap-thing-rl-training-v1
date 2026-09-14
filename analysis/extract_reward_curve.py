#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
チェックポイントの各 ONNX に CCR を実行し、ステップごとの報酬内訳を CSV 出力する。
使い方:
    python3 extract_reward_curve.py <checkpoint_dir> [--out <csv_path>]
例:
    python3 extract_reward_curve.py checkpoints/p2a_sigma_lin_0p01_20260731_014025
    python3 extract_reward_curve.py checkpoints/p2a_sigma_lin_0p01_20260731_014025 --out analysis_output/p2a_curve.csv
"""
import argparse, csv, json, math, os, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYTHON = ROOT / "Open_Duck_Playground" / ".venv" / "bin" / "python"
ENV = {**os.environ, "PYTHONPATH": str(ROOT) + ":" + os.environ.get("PYTHONPATH", "")}


def get_step(onnx_path: Path) -> int:
    try:
        return int(onnx_path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def load_scales(checkpoint_dir: Path):
    cfg_path = checkpoint_dir / "run_config.json"
    if not cfg_path.exists():
        print("[WARN] run_config.json が見つかりません。デフォルト値を使用します。")
        return {"sigma_lin": 0.025, "sigma_ang": 0.25,
                "scale_lin": 1.5, "scale_ang": 0.5, "scale_alive": 1.0}

    raw = cfg_path.read_text()
    cfg = json.loads(raw)
    overrides = json.loads(cfg.get("cli_args", {}).get("config_overrides", "{}"))
    env_rc = cfg.get("env_config", {}).get("reward_config", {})
    env_scales = env_rc.get("scales", {})

    def get(key_override, key_env, default):
        return overrides.get(key_override, env_scales.get(key_env, default))

    sigma_lin  = overrides.get("reward_config.tracking_sigma_lin",
                               env_rc.get("tracking_sigma_lin", 0.025))
    sigma_ang  = overrides.get("reward_config.tracking_sigma_ang",
                               env_rc.get("tracking_sigma_ang", 0.25))
    scale_lin  = get("reward_config.scales.tracking_lin_vel", "tracking_lin_vel", 1.5)
    scale_ang  = get("reward_config.scales.tracking_ang_vel", "tracking_ang_vel", 0.5)
    scale_alive = get("reward_config.scales.alive", "alive", 1.0)

    return {"sigma_lin": sigma_lin, "sigma_ang": sigma_ang,
            "scale_lin": scale_lin, "scale_ang": scale_ang, "scale_alive": scale_alive}


def run_ccr(onnx: Path, scales: dict, tmp_json: Path) -> dict | None:
    cmd = [
        str(PYTHON), "-m", "thing_test.check_command_response",
        "-o", str(onnx),
        "--no_viewer",
        "--axes", "vx,yaw",
        "--fractions=0.25,0.5,0.75,1.0",
        f"--json_out={tmp_json}",
    ]
    result = subprocess.run(cmd, env=ENV, cwd=str(ROOT),
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if result.returncode != 0 or not tmp_json.exists():
        return None

    data = json.loads(tmp_json.read_text())
    tmp_json.unlink(missing_ok=True)

    phases = data.get("phases", [])
    if not phases:
        return None

    sl, sa = scales["sigma_lin"], scales["sigma_ang"]
    r_lins, r_angs = [], []
    for p in phases:
        e_lin = p.get("cmd_vx", 0) - p.get("meas_vx", 0)
        e_ang = p.get("cmd_yaw", 0) - p.get("meas_yaw", 0)
        r_lins.append(math.exp(-e_lin ** 2 / sl))
        r_angs.append(math.exp(-e_ang ** 2 / sa))

    r_lin   = (sum(r_lins) / len(r_lins)) * scales["scale_lin"]
    r_ang   = (sum(r_angs) / len(r_angs)) * scales["scale_ang"]
    r_alive = scales["scale_alive"]

    return {"r_lin_vel": r_lin, "r_ang_vel": r_ang,
            "r_alive": r_alive, "r_total": r_lin + r_ang + r_alive}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    ckpt = Path(args.checkpoint_dir)
    if not ckpt.exists():
        print(f"[ERROR] {ckpt} が見つかりません"); sys.exit(1)

    out_csv = Path(args.out) if args.out else ROOT / "analysis/output" / f"{ckpt.name}_curve.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    scales = load_scales(ckpt)
    print(f"スケール: {scales}")

    onnx_files = sorted(ckpt.glob("*.onnx"), key=get_step)
    if not onnx_files:
        print("[ERROR] ONNX ファイルが見つかりません"); sys.exit(1)

    print(f"{len(onnx_files)} 個の ONNX を処理します...\n")
    tmp_json = ROOT / "_tmp_rcurve.json"

    rows = []
    for i, onnx in enumerate(onnx_files, 1):
        step = get_step(onnx)
        print(f"[{i}/{len(onnx_files)}] step={step:>12,}  {onnx.name}", end=" ... ", flush=True)
        r = run_ccr(onnx, scales, tmp_json)
        if r:
            rows.append({"step": step, **r})
            print(f"total={r['r_total']:.4f}  lin={r['r_lin_vel']:.4f}  ang={r['r_ang_vel']:.4f}  alive={r['r_alive']:.4f}")
        else:
            print("FAILED")

    if not rows:
        print("[ERROR] データが取得できませんでした"); sys.exit(1)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", "r_total", "r_lin_vel", "r_ang_vel", "r_alive"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] {out_csv}  ({len(rows)} 行)")


if __name__ == "__main__":
    main()
