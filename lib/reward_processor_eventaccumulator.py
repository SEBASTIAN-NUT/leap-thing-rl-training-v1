# -*- coding: utf-8 -*-
"""
TensorBoard EventAccumulator を使ってスカラー値を抽出
checkpoints フォルダの events.out.tfevents.* から直接データを読む
"""

from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import warnings

warnings.filterwarnings('ignore')

class RewardProcessor:
    """TensorBoard イベントからスカラーを抽出"""

    def __init__(self, checkpoints_dir: Path, scale_factors: dict):
        self.checkpoints_dir = checkpoints_dir
        self.scale_factors = scale_factors

    def process_phase(self, phase_name: str, checkpoint_dir_name: str) -> dict:
        """
        チェックポイントフォルダから TensorBoard イベントを読み込んで、
        スカラー値を抽出して返す

        Returns:
            {step: {'total': val, 'lin_vel': val, 'ang_vel': val, 'alive': val}, ...}
        """

        checkpoint_path = self.checkpoints_dir / checkpoint_dir_name

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        event_files = list(checkpoint_path.glob("events.out.tfevents.*"))

        if not event_files:
            raise FileNotFoundError(f"No TensorBoard event files in {checkpoint_path}")

        event_file = str(event_files[0])

        ea = EventAccumulator(event_file)
        ea.Reload()

        normalized_data = {}

        scalar_tags = ea.Tags().get('scalars', [])

        scale_factors = self.scale_factors.get(phase_name, {})
        lin_vel_scale = scale_factors.get('tracking_lin_vel', 1.0)
        ang_vel_scale = scale_factors.get('tracking_ang_vel', 1.0)
        alive_scale = scale_factors.get('alive', 1.0)

        lin_vel_raw = []
        ang_vel_raw = []
        alive_raw = []
        total_raw = []

        step_dict = {}

        for tag in scalar_tags:
            events = ea.Scalars(tag)

            if 'tracking_lin_vel' in tag:
                for event in events:
                    step = event.step
                    if step not in step_dict:
                        step_dict[step] = {}
                    step_dict[step]['lin_vel_raw'] = event.value

            elif 'tracking_ang_vel' in tag:
                for event in events:
                    step = event.step
                    if step not in step_dict:
                        step_dict[step] = {}
                    step_dict[step]['ang_vel_raw'] = event.value

            elif tag == 'eval/episode_reward/alive':
                for event in events:
                    step = event.step
                    if step not in step_dict:
                        step_dict[step] = {}
                    step_dict[step]['alive_raw'] = event.value

            elif tag == 'eval/episode_reward':
                for event in events:
                    step = event.step
                    if step not in step_dict:
                        step_dict[step] = {}
                    step_dict[step]['total'] = event.value

        for step, data in step_dict.items():
            lin_vel_raw = data.get('lin_vel_raw', 0)
            ang_vel_raw = data.get('ang_vel_raw', 0)
            alive_raw = data.get('alive_raw', 0)

            lin_vel_scaled = lin_vel_raw * lin_vel_scale
            ang_vel_scaled = ang_vel_raw * ang_vel_scale
            alive_scaled = alive_raw * alive_scale

            reward_normalized = lin_vel_scaled + ang_vel_scaled + alive_scaled

            normalized_data[step] = {
                'total': reward_normalized,
                'lin_vel': lin_vel_scaled,
                'ang_vel': ang_vel_scaled,
                'alive': alive_scaled,
            }

        return normalized_data
