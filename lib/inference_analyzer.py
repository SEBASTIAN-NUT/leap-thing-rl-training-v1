# -*- coding: utf-8 -*-
"""推論結果の分析"""
import json
from pathlib import Path
from typing import Dict, Tuple


class InferenceAnalyzer:
    """推論結果から追従率と最適パラメータを計算"""

    def __init__(self, eval_results_dir: Path):
        self.eval_results_dir = eval_results_dir

    def load_eval_result(self, json_file: str) -> Dict:
        """評価結果JSONを読み込む"""
        path = self.eval_results_dir / f"{json_file}.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} が見つかりません")

        with open(path, encoding='utf-8') as f:
            return json.load(f)

    def calculate_tracking_rates(self, eval_config: Dict[str, str]) -> Dict[str, Dict]:
        """全ての評価結果から追従率を計算"""
        results = {}

        for label, config in eval_config.items():
            json_file = config['file'].replace('.json', '')
            data = self.load_eval_result(json_file)

            phases = data['phases']
            avg_cmd_vx = sum(p['cmd_vx'] for p in phases) / len(phases)
            avg_cmd_yaw = sum(p['cmd_yaw'] for p in phases) / len(phases)
            avg_meas_vx = sum(abs(p['meas_vx']) for p in phases) / len(phases)
            avg_meas_yaw = sum(abs(p['meas_yaw']) for p in phases) / len(phases)

            vx_rate = (avg_meas_vx / avg_cmd_vx * 100) if avg_cmd_vx > 0 else 0
            yaw_rate = (avg_meas_yaw / avg_cmd_yaw * 100) if avg_cmd_yaw > 0 else 0

            results[label] = {
                'vx_tracking_rate': vx_rate,
                'yaw_tracking_rate': yaw_rate,
                'avg_cmd_vx': avg_cmd_vx,
                'avg_cmd_yaw': avg_cmd_yaw,
                'avg_meas_vx': avg_meas_vx,
                'avg_meas_yaw': avg_meas_yaw,
                'json_file': json_file,
            }

        return results

    def find_best_param(self, tracking_rates: Dict[str, Dict], metric: str = 'avg') -> Tuple[str, Dict]:
        """最も良いパラメータを確定"""
        if metric == 'avg':
            best = max(
                tracking_rates.items(),
                key=lambda x: (x[1]['vx_tracking_rate'] + x[1]['yaw_tracking_rate']) / 2
            )
        elif metric == 'vx':
            best = max(tracking_rates.items(), key=lambda x: x[1]['vx_tracking_rate'])
        elif metric == 'yaw':
            best = max(tracking_rates.items(), key=lambda x: x[1]['yaw_tracking_rate'])
        else:
            raise ValueError(f"不明な指標: {metric}")

        return best[0], best[1]

    def get_phase_data(self, json_file: str) -> Dict:
        """フェーズ別の詳細データを取得"""
        data = self.load_eval_result(json_file)
        return data['phases']

    def print_summary(self, tracking_rates: Dict[str, Dict]):
        """追従率の概要を表示"""
        print("\n=== 追従率 ===")
        for label, rates in tracking_rates.items():
            print(f"{label}:")
            print(f"  VX: {rates['vx_tracking_rate']:.1f}%")
            print(f"  YAW: {rates['yaw_tracking_rate']:.1f}%")
            print(f"  平均: {(rates['vx_tracking_rate'] + rates['yaw_tracking_rate']) / 2:.1f}%")
