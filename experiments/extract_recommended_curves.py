#!/usr/bin/env python3
import csv
import json
import re
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

CHECKPOINTS = Path("checkpoints")
OUTPUT_DIR = Path("analysis_plots")
EVAL_DIR = Path("eval_results")

RECOMMENDED = [
    ("p3_alive_0p5_20260711_105838", "Phase 3: alive=0.5"),
    ("p5_stand_still_m1p0_20260717_161824", "Phase 5: stand_still=-1.0"),
    ("p5_feet_air_time_0p1_20260720_151328", "Phase 5: feet_air_time=0.1"),
]

def best_event_file(ckpt_dir):
    files = list(ckpt_dir.glob("events.out.tfevents.*"))
    return max(files, key=lambda p: p.stat().st_size) if files else None

def read_scalars(event_path):
    ea = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    result = {}
    for tag in ["eval/episode_reward", "eval/episode_reward/tracking_lin_vel",
                "eval/episode_reward/tracking_ang_vel", "eval/episode_reward/alive",
                "eval/avg_episode_length"]:
        if tag in available:
            evs = ea.Scalars(tag)
            result[tag] = (
                np.array([e.step for e in evs], dtype=float),
                np.array([e.value for e in evs], dtype=float),
            )
    return result

def extract_curve(ckpt_name, description):
    ckpt_dir = CHECKPOINTS / ckpt_name
    if not ckpt_dir.exists():
        print(f"[SKIP] {ckpt_name} not found")
        return None
    
    evf = best_event_file(ckpt_dir)
    if evf is None:
        print(f"[SKIP] {ckpt_name} has no event file")
        return None
    
    try:
        data = read_scalars(evf)
    except Exception as e:
        print(f"[ERROR] {ckpt_name}: {e}")
        return None
    
    if not data:
        print(f"[SKIP] {ckpt_name} has no scalar tags")
        return None
    
    tag_total = "eval/episode_reward"
    if tag_total not in data:
        print(f"[SKIP] {ckpt_name} has no total reward")
        return None
    
    steps, total = data[tag_total]
    rows = []
    for i, step in enumerate(steps):
        row = {"Timesteps": int(step), "Total": round(float(total[i]), 3)}
        for tag, key in [("eval/episode_reward/tracking_lin_vel", "Lin"),
                         ("eval/episode_reward/tracking_ang_vel", "Ang"),
                         ("eval/episode_reward/alive", "Alive"),
                         ("eval/avg_episode_length", "EpisodeLen")]:
            if tag in data:
                _, vals = data[tag]
                if i < len(vals):
                    row[key] = round(float(vals[i]), 3)
        rows.append(row)
    
    return rows

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    summary_rows = []
    
    for ckpt_name, description in RECOMMENDED:
        print(f"[extract] {description}...")
        rows = extract_curve(ckpt_name, description)
        if not rows:
            continue
        
        safe_name = re.sub(r"[^a-z0-9]+", "_", description.lower()).strip("_")
        out_csv = OUTPUT_DIR / f"{safe_name}_curve.csv"
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        print(f"  OK: {out_csv} ({len(rows)} points)")
        
        if rows:
            last = rows[-1]
            row_data = {
                "Experiment": description,
                "Final Total": last.get("Total", ""),
                "Final Lin": last.get("Lin", ""),
                "Final Ang": last.get("Ang", ""),
                "Final Alive": last.get("Alive", ""),
                "Final EpisodeLen": last.get("EpisodeLen", ""),
                "Timesteps": last.get("Timesteps", ""),
                "RMSE_vx": "",
                "RMSE_yaw": "",
            }
            
            exp = description
            if "alive" in exp.lower():
                json_stem = "p3_alive_0p5"
            elif "stand" in exp.lower():
                json_stem = "p5_stand_still_m1p0"
            else:
                json_stem = "p5_feet_air_time_0p1"
            
            json_file = EVAL_DIR / f"{json_stem}.json"
            if json_file.exists():
                try:
                    with open(json_file, encoding="utf-8") as f:
                        data = json.load(f)
                        row_data["RMSE_vx"] = round(data.get("rmse_vx", 0), 4)
                        row_data["RMSE_yaw"] = round(data.get("rmse_yaw", 0), 4)
                except:
                    pass
            
            summary_rows.append(row_data)
    
    comparison_csv = OUTPUT_DIR / "comparison_summary.csv"
    with open(comparison_csv, "w", newline="", encoding="utf-8") as f:
        if summary_rows:
            fieldnames = ["Experiment", "Final Total", "Final Lin", "Final Ang", "Final Alive", "Final EpisodeLen", "Timesteps", "RMSE_vx", "RMSE_yaw"]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"\nOK: {comparison_csv}")
    
    print(f"Done. Output: {OUTPUT_DIR}/")

if __name__ == "__main__":
    main()
