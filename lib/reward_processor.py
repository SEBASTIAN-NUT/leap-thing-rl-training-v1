# -*- coding: utf-8 -*-
from pathlib import Path
from typing import Dict


class RewardProcessor:

    TAGS = {
        "eval/episode_reward":                  "total",
        "eval/episode_reward/tracking_lin_vel": "lin_vel_raw",
        "eval/episode_reward/tracking_ang_vel": "ang_vel_raw",
        "eval/episode_reward/alive":            "alive_raw",
    }

    def __init__(self, checkpoints_dir: Path, scale_factors: Dict):
        self.checkpoints_dir = checkpoints_dir
        self.scale_factors = scale_factors

    def load_scalars(self, checkpoint_path: Path) -> Dict[int, Dict]:
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        except ImportError:
            print("  [WARN] tensorboard not found")
            return {}
        ckpt_dir = self.checkpoints_dir / checkpoint_path
        event_files = list(ckpt_dir.glob("events.out.tfevents.*"))
        if not event_files:
            print(f"  [WARN] no TFEvents in {ckpt_dir}")
            return {}
        event_file = max(event_files, key=lambda p: p.stat().st_size)
        ea = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        ea.Reload()
        available = set(ea.Tags().get("scalars", []))
        data: Dict[int, Dict] = {}
        for tag, key in self.TAGS.items():
            if tag not in available:
                continue
            for ev in ea.Scalars(tag):
                step = int(ev.step)
                if step not in data:
                    data[step] = {}
                data[step][key] = ev.value
        return data

    def normalize_data(self, data: Dict[int, Dict], phase_name: str) -> Dict[int, Dict]:
        scales = self.scale_factors.get(phase_name, {})
        normalized = {}
        for step in sorted(data.keys()):
            row = data[step]
            lin = row.get("lin_vel_raw", 0) * scales.get("tracking_lin_vel", 1.0)
            ang = row.get("ang_vel_raw", 0) * scales.get("tracking_ang_vel", 1.0)
            alv = row.get("alive_raw",   0) * scales.get("alive",             1.0)
            normalized[step] = {"timestep": step, "total": lin+ang+alv,
                                "lin_vel": lin, "ang_vel": ang, "alive": alv}
        return normalized

    def process_phase(self, phase_name: str, checkpoint_path: str) -> Dict[int, Dict]:
        print(f"  処理中: {phase_name}")
        raw = self.load_scalars(Path(checkpoint_path))
        normalized = self.normalize_data(raw, phase_name)
        print(f"  [OK] {len(normalized)} ステップを処理完了")
        return normalized
