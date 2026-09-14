import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
#!/usr/bin/env python3
"""
学習結果サマリー CSV 生成
既存の eval/episode_reward/* タグから末尾平均を抽出し、
Phase 毎のパラメータ比較表を CSV で出力する。
"""
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

CHECKPOINTS = Path("checkpoints")
OUTPUT_DIR  = Path("analysis_plots")

CATEGORIES = [
    ("p1",    "Phase 1: sigma_lin sweep",        r"p1_siglin_(.+?)_\d{8}_\d{6}"),
    ("p2ang", "Phase 2: sigma_ang sweep",         r"p2_siggang_(.+?)_\d{8}_\d{6}"),
    ("p2a",   "Phase 2a: sigma_lin refined",      r"p2a_sigma_lin_(.+?)_\d{8}_\d{6}"),
    ("p3",    "Phase 3: alive sweep",             r"p3_alive_(.+?)_\d{8}_\d{6}"),
    ("p5ss",  "Phase 5: stand_still",             r"p5_stand_still_(.+?)_\d{8}_\d{6}"),
    ("p5ori", "Phase 5: orientation",             r"p5_orientation_(.+?)_\d{8}_\d{6}"),
    ("p5fat", "Phase 5: feet_air_time",           r"p5_feet_air_time_(.+?)_\d{8}_\d{6}"),
]

def read_scalars(event_path: Path) -> dict:
    ea = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    result = {}
    for tag in available:
        evs = ea.Scalars(tag)
        result[tag] = (
            np.array([e.step  for e in evs], dtype=float),
            np.array([e.value for e in evs], dtype=float),
        )
    return result

def best_event_file(d: Path):
    files = list(d.glob("events.out.tfevents.*"))
    return max(files, key=lambda p: p.stat().st_size) if files else None

def tail_mean(data: dict, tag: str, frac: float = 0.85) -> float:
    if tag not in data:
        return float("nan")
    v = data[tag][1]
    if len(v) == 0:
        return float("nan")
    return float(np.mean(v[max(0, int(len(v) * frac)):]))

def collect_all() -> dict:
    groups = defaultdict(dict)
    for ckpt in sorted(CHECKPOINTS.iterdir()):
        if not ckpt.is_dir():
            continue
        evf = best_event_file(ckpt)
        if evf is None:
            continue
        matched = False
        for cat_key, _, pattern in CATEGORIES:
            m = re.fullmatch(pattern, ckpt.name)
            if m:
                label = m.group(1)
                try:
                    data = read_scalars(evf)
                except Exception as e:
                    break
                if not data:
                    break
                existing = groups[cat_key].get(label)
                def max_step(d):
                    return max((v[0][-1] for v in d.values() if len(v[0])), default=0)
                if existing is None or max_step(data) > max_step(existing):
                    groups[cat_key][label] = data
                matched = True
                break
        if not matched:
            pass
    return groups

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    groups = collect_all()

    cat_map = {k: title for k, title, _ in CATEGORIES}
    
    rows = []
    for cat_key in [k for k, _, _ in CATEGORIES]:
        runs = groups.get(cat_key)
        if not runs:
            continue
        title = cat_map[cat_key]
        for label in sorted(runs.keys()):
            data = runs[label]
            ms = max((v[0][-1] for v in data.values() if len(v[0])), default=0)
            
            total = tail_mean(data, "eval/episode_reward")
            lin = tail_mean(data, "eval/episode_reward/tracking_lin_vel")
            ang = tail_mean(data, "eval/episode_reward/tracking_ang_vel")
            alive = tail_mean(data, "eval/episode_reward/alive")
            eplen = tail_mean(data, "eval/avg_episode_length")
            
            if not np.isnan(total):
                lin_pct = (lin / total * 100.0) if total > 0 else 0
                ang_pct = (ang / total * 100.0) if total > 0 else 0
                alive_pct = (alive / total * 100.0) if total > 0 else 0
            else:
                lin_pct = ang_pct = alive_pct = float("nan")
            
            rows.append({
                "Category": title,
                "Label": label,
                "Total Reward": round(total, 3),
                "Lin Tracking": round(lin, 3),
                "Ang Tracking": round(ang, 3),
                "Alive": round(alive, 3),
                "Lin %": round(lin_pct, 1) if not np.isnan(lin_pct) else "",
                "Ang %": round(ang_pct, 1) if not np.isnan(ang_pct) else "",
                "Alive %": round(alive_pct, 1) if not np.isnan(alive_pct) else "",
                "Episode Length": round(eplen, 1),
                "Total Steps": int(ms),
            })
    
    import csv
    out_path = OUTPUT_DIR / "learning_results_summary.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    
    print(f"CSV saved: {out_path}")
    print(f"\n{len(rows)} runs exported")

if __name__ == "__main__":
    main()
