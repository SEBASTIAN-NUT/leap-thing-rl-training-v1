import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
# -*- coding: utf-8 -*-
#!/usr/bin/env python3
"""
eval_comparison.py
全フェーズ 速度追従率 一括評価 + 比較CSV生成

使い方:
  Open_Duck_Playground/.venv/bin/python experiments/eval_comparison.py
  Open_Duck_Playground/.venv/bin/python experiments/eval_comparison.py --skip_eval
"""
import argparse
import csv
import json
import subprocess
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON = SCRIPT_DIR / "Open_Duck_Playground" / ".venv" / "bin" / "python"
EVAL_DIR = SCRIPT_DIR / "eval_results" / "comparison"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# (name, glob_patterns, sigma_lin, sigma_ang, alive, sweep_phase)
TARGETS = [
    ("p2a_siglin_0p0025", ["checkpoints/p2a_sigma_lin_0p0025_*"],                         0.0025, 0.25, None, "p2a"),
    ("p2a_siglin_0p005",  ["checkpoints/p2a_sigma_lin_0p005_*"],                          0.005,  0.25, None, "p2a"),
    ("p2a_siglin_0p01",   ["checkpoints/p2a_sigma_lin_0p01_*"],                           0.01,   0.25, None, "p2a"),
    ("p2a_siglin_0p025",  ["checkpoints/p2a_sigma_lin_0p025_*"],                          0.025,  0.25, None, "p2a"),
    ("p2a_siglin_0p05",   ["checkpoints/p2a_sigma_lin_0p05_*"],                           0.05,   0.25, None, "p2a"),
    ("p2b_sigang_0p25",   ["../transfer/checkpoints/p2_siggang_0p25_*",
                           "checkpoints/p2b_sigma_ang_0p25_*"],                           0.025,  0.25, None, "p2b"),
    ("p2b_sigang_0p5",    ["../transfer/checkpoints/p2_siggang_rerun_0p5_*",
                           "../transfer/checkpoints/p2_siggang_0p5_*",
                           "checkpoints/p2b_sigma_ang_0p5_*"],                            0.025,  0.5,  None, "p2b"),
    ("p2b_sigang_1p0",    ["../transfer/checkpoints/p2_siggang_1p0_*",
                           "checkpoints/p2b_sigma_ang_1p0_*"],                            0.025,  1.0,  None, "p2b"),
    ("p2b_sigang_2p0",    ["../transfer/checkpoints/p2_siggang_rerun_2p0_*",
                           "../transfer/checkpoints/p2_siggang_2p0_*",
                           "checkpoints/p2b_sigma_ang_2p0_*"],                            0.025,  2.0,  None, "p2b"),
    ("p3_alive_0p1",      ["../transfer/checkpoints/p3_alive_0p1_*",
                           "checkpoints/p3_alive_0p1_*"],                                 0.025,  0.25, 0.1,  "p3"),
    ("p3_alive_0p2",      ["../transfer/checkpoints/p3_alive_0p2_*",
                           "checkpoints/p3_alive_0p2_*"],                                 0.025,  0.25, 0.2,  "p3"),
    ("p3_alive_0p3",      ["../transfer/checkpoints/p3_alive_0p3_*",
                           "checkpoints/p3_alive_0p3_*"],                                 0.025,  0.25, 0.3,  "p3"),
    ("p3_alive_0p5",      ["../transfer/checkpoints/p3_alive_0p5_*",
                           "checkpoints/p3_alive_0p5_*"],                                 0.025,  0.25, 0.5,  "p3"),
    ("p3_alive_1p0",      ["../transfer/checkpoints/p3_alive_1p0_*",
                           "checkpoints/p3_alive_1p0_*"],                                 0.025,  0.25, 1.0,  "p3"),
]


def step_of(onnx: Path) -> int:
    try:
        return int(onnx.stem.rsplit("_", 1)[-1])
    except ValueError:
        return -1


def find_best_onnx(patterns):
    best_onnx, best_step = None, -1
    for pat in patterns:
        for ckpt_dir in sorted(SCRIPT_DIR.glob(pat)):
            if not ckpt_dir.is_dir():
                continue
            for onnx in ckpt_dir.glob("*.onnx"):
                s = step_of(onnx)
                if s > best_step:
                    best_step, best_onnx = s, onnx
    return best_onnx


def run_eval(name, onnx):
    json_out = EVAL_DIR / f"{name}.json"
    csv_out  = EVAL_DIR / f"{name}.csv"
    cmd = [
        str(PYTHON), "-m", "thing_test.check_command_response",
        "-o", str(onnx), "--no_viewer",
        "--axes", "vx,yaw",
        "--fractions", "0.25,0.5,0.75,1.0",
        "--phase_duration", "4.0",
        "--json_out", str(json_out),
        "--csv_out",  str(csv_out),
    ]
    print(f"  eval: {onnx.name}")
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    output = result.stdout.decode("utf-8", errors="replace")
    print(output[-2000:] if len(output) > 2000 else output)
    if result.returncode != 0 or not json_out.exists():
        print(f"  [FAILED] {name}")
        return None
    with open(json_out, encoding="utf-8") as f:
        data = json.load(f)
    data["_name"] = name
    return data


def tracking_rate(meas, cmd):
    if abs(cmd) < 1e-6:
        return "-"
    return f"{meas / cmd * 100:.1f}"


def save_comparison_csv(results, meta, out_path):
    fields = [
        "name", "sweep_phase", "sigma_lin", "sigma_ang", "alive",
        "fraction", "onnx_steps",
        "cmd_vx", "meas_vx", "tracking_rate_vx_pct",
        "cmd_yaw", "meas_yaw", "tracking_rate_yaw_pct",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for res in results:
            name = res["_name"]
            m = meta.get(name, {})
            for phase in res.get("phases", []):
                cmd_vx   = phase.get("cmd_vx",  0.0)
                meas_vx  = phase.get("meas_vx", 0.0)
                cmd_yaw  = phase.get("cmd_yaw",  0.0)
                meas_yaw = phase.get("meas_yaw", 0.0)
                w.writerow({
                    "name":                  name,
                    "sweep_phase":           m.get("sweep_phase", ""),
                    "sigma_lin":             m.get("sigma_lin", ""),
                    "sigma_ang":             m.get("sigma_ang", ""),
                    "alive":                 m.get("alive", ""),
                    "fraction":              phase.get("phase", ""),
                    "onnx_steps":            m.get("onnx_steps", ""),
                    "cmd_vx":                f"{cmd_vx:.4f}",
                    "meas_vx":               f"{meas_vx:.4f}",
                    "tracking_rate_vx_pct":  tracking_rate(meas_vx, cmd_vx),
                    "cmd_yaw":               f"{cmd_yaw:.4f}",
                    "meas_yaw":              f"{meas_yaw:.4f}",
                    "tracking_rate_yaw_pct": tracking_rate(meas_yaw, cmd_yaw),
                })
    print(f"\n比較CSV -> {out_path}")


def print_summary(results, meta):
    print("\n" + "=" * 80)
    print(f"{'name':<22} {'phase':<5} {'sig_lin':<8} {'sig_ang':<8} {'alive':<6} {'steps':>10}  {'vx@50%':>7} {'vx@100%':>8}")
    print("=" * 80)
    for res in results:
        name = res["_name"]
        m = meta.get(name, {})
        phases = {p["phase"]: p for p in res.get("phases", [])}
        vx50   = phases.get("vx 50%",  {}).get("meas_vx", float("nan"))
        vx100  = phases.get("vx 100%", {}).get("meas_vx", float("nan"))
        cmd50  = phases.get("vx 50%",  {}).get("cmd_vx",  0.075)
        cmd100 = phases.get("vx 100%", {}).get("cmd_vx",  0.150)
        r50  = f"{vx50/cmd50*100:5.1f}%"   if cmd50  > 0 else "  N/A"
        r100 = f"{vx100/cmd100*100:5.1f}%" if cmd100 > 0 else "  N/A"
        print(f"{name:<22} {m.get('sweep_phase',''):<5} "
              f"{str(m.get('sigma_lin','')):<8} {str(m.get('sigma_ang','')):<8} "
              f"{str(m.get('alive','')):<6} {str(m.get('onnx_steps',''))!s:>10}  "
              f"{r50:>7} {r100:>8}")
    print("=" * 80)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_eval", action="store_true")
    args = parser.parse_args()
    results, meta = [], {}
    for (name, patterns, sigma_lin, sigma_ang, alive, sweep_phase) in TARGETS:
        json_path = EVAL_DIR / f"{name}.json"
        if args.skip_eval and json_path.exists():
            print(f"[SKIP] {name}")
            with open(json_path, encoding="utf-8") as f:
                res = json.load(f)
            res["_name"] = name
        else:
            onnx = find_best_onnx(patterns)
            if onnx is None:
                print(f"[SKIP] {name}: ONNXなし")
                continue
            if step_of(onnx) <= 0:
                print(f"[SKIP] {name}: step=0 (未学習)")
                continue
            print(f"\n[{name}]  {onnx.name}  steps={step_of(onnx):,}")
            res = run_eval(name, onnx)
            if res is None:
                continue
        onnx = find_best_onnx(patterns)
        meta[name] = {
            "sweep_phase": sweep_phase,
            "sigma_lin":   sigma_lin,
            "sigma_ang":   sigma_ang,
            "alive":       alive if alive is not None else "-",
            "onnx_steps":  step_of(onnx) if onnx else "?",
        }
        results.append(res)
    if not results:
        print("結果なし。")
        return
    save_comparison_csv(results, meta, EVAL_DIR / "tracking_comparison.csv")
    print_summary(results, meta)


if __name__ == "__main__":
    main()
