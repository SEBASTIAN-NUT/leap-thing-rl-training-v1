#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path.cwd()
EVAL_DIR = SCRIPT_DIR / "eval_results"
SUMMARY_DIR = SCRIPT_DIR / "poster_summary"

RECOMMENDED = [
    {
        "name": "p3_alive_0p5",
        "title": "Phase 3: alive=0.5",
        "params": {
            "phase": "Phase 3 (alive sweep)",
            "reward_config.scales.alive": 0.5,
            "reward_config.scales.tracking_lin_vel": 3.0,
            "reward_config.scales.tracking_ang_vel": 1.0,
            "reward_config.tracking_sigma_lin": 0.025,
            "reward_config.tracking_sigma_ang": 2.0,
        },
        "notes": "生存報酬と追従のバランスが最適な alive スケール",
    },
    {
        "name": "p5_stand_still_m1p0",
        "title": "Phase 5: stand_still=-1.0",
        "params": {
            "phase": "Phase 5 (Go1 terms - stand_still)",
            "reward_config.scales.stand_still": -1.0,
            "from_phase_3": "alive=0.5, trk_lin=3.0, trk_ang=1.0",
        },
        "notes": "停止時のドリフト抑制を追加。yaw 追従性能向上（RMSE_yaw=0.0021）",
    },
    {
        "name": "p5_feet_air_time_0p1",
        "title": "Phase 5: feet_air_time=0.1",
        "params": {
            "phase": "Phase 5 (Go1 terms - feet_air_time)",
            "reward_config.scales.feet_air_time": 0.1,
            "from_phase_3": "alive=0.5, trk_lin=3.0, trk_ang=1.0",
        },
        "notes": "指先の空中時間報酬を追加。歩行リズム改善",
    },
]

def load_eval_json(name: str) -> dict | None:
    path = EVAL_DIR / f"{name}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return None

def compute_csv_metrics(name: str) -> dict:
    csv_path = EVAL_DIR / f"{name}.csv"
    if not csv_path.exists():
        return {}

    import csv as csv_module
    metrics = {
        "vx_100_rmse": None,
        "vx_75_rmse": None,
        "vx_50_rmse": None,
        "vx_25_rmse": None,
        "yaw_100_rmse": None,
    }

    phases = {}
    with open(csv_path, encoding="utf-8") as f:
        reader = csv_module.DictReader(f)
        for row in reader:
            phase = row["phase"]
            if phase not in phases:
                phases[phase] = []
            cmd_vx = float(row["cmd_vx"])
            inst_vx = float(row["inst_vx"])
            cmd_yaw = float(row["cmd_yaw"])
            inst_yaw = float(row["inst_yaw_rate"])
            phases[phase].append({
                "cmd_vx": cmd_vx,
                "inst_vx": inst_vx,
                "cmd_yaw": cmd_yaw,
                "inst_yaw": inst_yaw,
                "err_vx": cmd_vx - inst_vx,
                "err_yaw": cmd_yaw - inst_yaw,
            })

    for phase_label, intensity in [("vx 25%", "vx_25"), ("vx 50%", "vx_50"),
                                     ("vx 75%", "vx_75"), ("vx 100%", "vx_100")]:
        if phase_label in phases:
            errs = [abs(p["err_vx"]) for p in phases[phase_label]]
            rmse = (sum(e*e for e in errs) / len(errs)) ** 0.5 if errs else 0
            metrics[f"{intensity}_rmse"] = round(rmse, 4)

    if "yaw 100%" in phases:
        errs = [abs(p["err_yaw"]) for p in phases["yaw 100%"]]
        rmse = (sum(e*e for e in errs) / len(errs)) ** 0.5 if errs else 0
        metrics["yaw_100_rmse"] = round(rmse, 4)

    return metrics

SUMMARY_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("Creating poster summary folders...")
print("=" * 70)

summaries = []

for rec in RECOMMENDED:
    name = rec["name"]
    folder = SUMMARY_DIR / name
    folder.mkdir(exist_ok=True)

    eval_json = load_eval_json(name) or {}
    csv_metrics = compute_csv_metrics(name)

    metadata = {
        "name": name,
        "title": rec["title"],
        "created_at": datetime.now().isoformat(),
        "parameters": rec["params"],
        "notes": rec["notes"],
        "evaluation": {
            "eval_json": eval_json,
            "csv_metrics": csv_metrics,
        },
    }

    meta_path = folder / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"✓ {meta_path}")

    csv_src = EVAL_DIR / f"{name}.csv"
    if csv_src.exists():
        csv_dst = folder / "time_series.csv"
        csv_dst.write_bytes(csv_src.read_bytes())
        print(f"✓ {csv_dst}")

    summaries.append({
        "name": name,
        "title": rec["title"],
        "rmse_vx": csv_metrics.get("vx_100_rmse"),
        "rmse_yaw": csv_metrics.get("yaw_100_rmse"),
    })

comparison_md = SUMMARY_DIR / "comparison_summary.md"
with open(comparison_md, "w", encoding="utf-8") as f:
    f.write("# 推奨パラメータセット比較\n\n")
    f.write("| パラメータセット | RMSE_vx (100%) | RMSE_yaw (100%) | 特徴 |\n")
    f.write("|---|---|---|---|\n")

    for i, s in enumerate(summaries):
        vx_str = f"{s['rmse_vx']:.4f}" if s['rmse_vx'] else "N/A"
        yaw_str = f"{s['rmse_yaw']:.4f}" if s['rmse_yaw'] else "N/A"
        rec_note = RECOMMENDED[i]["notes"]
        f.write(f"| {s['title']} | {vx_str} | {yaw_str} | {rec_note} |\n")

    f.write("\n## 詳細\n\n")
    for i, rec in enumerate(RECOMMENDED):
        f.write(f"### {rec['title']}\n")
        f.write(f"**説明**: {rec['notes']}\n\n")
        f.write(f"**パラメータ**:\n")
        for key, val in rec['params'].items():
            f.write(f"- `{key}`: {val}\n")
        f.write(f"\n**ファイル**: `{rec['name']}/`\n")
        f.write(f"- `metadata.json`: パラメータと評価指標\n")
        f.write(f"- `time_series.csv`: 推論時系列データ (cmd vs inst)\n")
        f.write("\n")

print(f"✓ {comparison_md}")

print("\n" + "=" * 70)
print("Summary folders created in poster_summary/")
print("=" * 70)
