# -*- coding: utf-8 -*-
"""
TFEvents から生の報酬値を読み込んで、スケール係数を適用して正規化する
(scalars.csv は存在しないため TFEvents を直接読む)
"""
import csv
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT_DIR = Path(__file__).parent
CHECKPOINTS_DIR = PROJECT_DIR / "checkpoints"
OUTPUT_CSV = PROJECT_DIR / "analysis/output" / "03_learning_curves.csv"

# フェーズごとの設定（run_config.json の config_overrides から取得した実際の値）
RUNS = {
    "phase2a_sigma_lin": {
        "checkpoint": "p2a_sigma_lin_0p025_20260731_172703",
        "scales": {
            "tracking_lin_vel": 1.5,
            "tracking_ang_vel": 0.5,
            "alive": 1.0,
        },
    },
    "phase2b_sigma_ang": {
        "checkpoint": "p2b_sigma_ang_2p0_20260731_072750",
        "scales": {
            "tracking_lin_vel": 1.5,
            "tracking_ang_vel": 0.5,
            "alive": 1.0,
        },
    },
    "phase3_alive": {
        "checkpoint": "p3_alive_0p1_20260803_023938",
        "scales": {
            "tracking_lin_vel": 3.0,
            "tracking_ang_vel": 1.0,
            "alive": 0.1,
        },
    },
}

TAGS = {
    "lin_vel_raw": "eval/episode_reward/tracking_lin_vel",
    "ang_vel_raw": "eval/episode_reward/tracking_ang_vel",
    "alive_raw":   "eval/episode_reward/alive",
    "total":       "eval/episode_reward",
}


def load_tfevents(checkpoint_dir: Path) -> dict:
    ea = EventAccumulator(str(checkpoint_dir))
    ea.Reload()
    data = {}
    for key, tag in TAGS.items():
        try:
            for e in ea.Scalars(tag):
                if e.step not in data:
                    data[e.step] = {}
                data[e.step][key] = e.value
        except KeyError:
            print(f"  [WARN] tag not found: {tag}")
    return data


print("\nチェックポイント フォルダ:")
for run_name, cfg in RUNS.items():
    p = CHECKPOINTS_DIR / cfg["checkpoint"]
    print(f"  {run_name}: {cfg['checkpoint']} {'[OK]' if p.exists() else '[NOT FOUND]'}")

all_data = {}

for run_name, cfg in RUNS.items():
    checkpoint_path = CHECKPOINTS_DIR / cfg["checkpoint"]
    if not checkpoint_path.exists():
        print(f"\n[SKIP] {run_name}: not found")
        continue

    print(f"\n読み込み中: {run_name}")
    scales = cfg["scales"]
    raw = load_tfevents(checkpoint_path)

    for step, row in raw.items():
        lin_vel_scaled = row.get("lin_vel_raw", 0.0) * scales["tracking_lin_vel"]
        ang_vel_scaled = row.get("ang_vel_raw", 0.0) * scales["tracking_ang_vel"]
        alive_scaled   = row.get("alive_raw",   0.0) * scales["alive"]
        reward_normalized = lin_vel_scaled + ang_vel_scaled + alive_scaled

        if step not in all_data:
            all_data[step] = {"timestep": step}

        all_data[step][f"{run_name}_reward"]  = reward_normalized
        all_data[step][f"{run_name}_lin_vel"] = lin_vel_scaled
        all_data[step][f"{run_name}_ang_vel"] = ang_vel_scaled
        all_data[step][f"{run_name}_alive"]   = alive_scaled

print(f"\nステップ数:")
for run_name in RUNS:
    n = sum(1 for v in all_data.values() if f"{run_name}_reward" in v)
    print(f"  {run_name}: {n}")

print(f"\n書き込み中: {OUTPUT_CSV}")
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

fieldnames = ["timestep"]
for rn in RUNS:
    fieldnames += [f"{rn}_reward", f"{rn}_lin_vel", f"{rn}_ang_vel", f"{rn}_alive"]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for step in sorted(all_data.keys()):
        writer.writerow(all_data[step])

print(f"[OK] 正規化されたデータを書き込みました: {OUTPUT_CSV}")
print(f"\n次のコマンドで PNG グラフを生成してください:")
print(f"  python generate_learning_curves_png.py")
