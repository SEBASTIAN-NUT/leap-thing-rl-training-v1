#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
リモート環境用メインスクリプト（自動検出版）
- 最新のチェックポイントを自動検出して解析
- eval_results も自動検出
- ユーザーは python main_remote.py を実行するだけ
"""

import sys
import json
import csv
from pathlib import Path
from datetime import datetime
import traceback
from typing import Dict, List, Tuple

def log_message(logger, level, message):
    """統一されたログメッセージ出力"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prefix = f"[{level:^6}]" if level in ["OK", "WARN", "NG"] else f"[{level}]"
    output = f"{prefix} {message}"
    print(output)
    if logger:
        logger.write(output + "\n")
        logger.flush()

def extract_timestamp(folder_name: str) -> str:
    """フォルダ名からタイムスタンプを抽出（YYYYMMDD_HHMMSS形式）"""
    parts = folder_name.split('_')
    for i, part in enumerate(parts):
        if len(part) == 8 and part.isdigit():
            if i + 1 < len(parts) and len(parts[i + 1]) == 6 and parts[i + 1].isdigit():
                return f"{part}_{parts[i + 1]}"
    return ""

def find_latest_checkpoints(checkpoints_dir: Path) -> Dict[str, str]:
    """各タイプの最新チェックポイントを自動検出"""
    detected = {}

    if not checkpoints_dir.exists():
        return detected

    for folder in checkpoints_dir.iterdir():
        if not folder.is_dir():
            continue

        folder_name = folder.name

        if 'sigma_lin' in folder_name or 'sigma_lin' in folder.name:
            checkpoint_type = 'phase2a_sigma_lin'
        elif 'sigma_ang' in folder_name or 'siggang' in folder_name:
            checkpoint_type = 'phase2b_sigma_ang'
        elif 'alive' in folder_name:
            checkpoint_type = 'phase3_alive'
        elif 'base' in folder_name or 'baseline' in folder_name:
            checkpoint_type = 'phase1_baseline'
        else:
            continue

        if checkpoint_type not in detected:
            detected[checkpoint_type] = folder_name
        else:
            ts_current = extract_timestamp(detected[checkpoint_type])
            ts_new = extract_timestamp(folder_name)
            if ts_new > ts_current:
                detected[checkpoint_type] = folder_name

    return detected

def find_latest_eval_results(eval_results_dir: Path) -> Dict[str, str]:
    """eval_results から使用可能なファイルを検出"""
    detected = {}

    if not eval_results_dir.exists():
        return detected

    for item in eval_results_dir.iterdir():
        name = item.name

        if 'sigma_lin' in name and name.endswith('.json'):
            detected['lin'] = item.name
        elif ('sigma_ang' in name or 'siggang' in name) and name.endswith('.json'):
            detected['ang'] = item.name
        elif 'alive' in name and name.endswith('.json'):
            detected['alive'] = item.name

    return detected

def generate_dynamic_config(project_dir: Path, checkpoints_detected: Dict, eval_detected: Dict) -> dict:
    """動的に config を生成"""
    base_config_path = project_dir / 'config/phase_config.json'

    with open(base_config_path, encoding='utf-8') as f:
        base_config = json.load(f)

    base_config['checkpoints'] = checkpoints_detected

    if eval_detected:
        base_config['eval_configs'] = {
            label: {'file': filename, 'label': label}
            for label, filename in eval_detected.items()
        }

    return base_config

def load_config(config_path: Path) -> dict:
    """config.json を読み込む"""
    with open(config_path, encoding='utf-8') as f:
        return json.load(f)

def run_analysis(project_dir: Path, transfer_dir: Path, run_id: str):
    """解析メイン処理"""

    run_output_dir = transfer_dir / 'analysis_output' / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    log_file = run_output_dir / 'execution_log.txt'
    logger = open(log_file, 'w', encoding='utf-8')

    try:
        log_message(logger, "START", f"解析開始: {run_id}")
        log_message(logger, "INFO", f"プロジェクトディレクトリ: {project_dir}")
        log_message(logger, "INFO", f"出力ディレクトリ: {run_output_dir}")

        checkpoints_dir = project_dir / 'checkpoints'
        eval_results_dir = project_dir / 'eval_results'

        log_message(logger, "INFO", "=" * 60)
        log_message(logger, "INFO", "[AUTO] チェックポイントと評価結果を自動検出中...")
        log_message(logger, "INFO", "=" * 60)

        checkpoints_detected = find_latest_checkpoints(checkpoints_dir)
        eval_detected = find_latest_eval_results(eval_results_dir)

        if not checkpoints_detected:
            log_message(logger, "WARN", f"チェックポイントが見つかりません: {checkpoints_dir}")
        else:
            for phase, checkpoint in checkpoints_detected.items():
                log_message(logger, "OK", f"検出: {phase} -> {checkpoint}")

        if not eval_detected:
            log_message(logger, "WARN", f"Eval results が見つかりません: {eval_results_dir}")
        else:
            for label, filename in eval_detected.items():
                log_message(logger, "OK", f"検出: {label} -> {filename}")

        config_path = project_dir / 'config/phase_config.json'
        if not config_path.exists():
            log_message(logger, "NG", f"Base config 見つかりません: {config_path}")
            return False

        log_message(logger, "OK", "Base config 読み込み完了")

        config = generate_dynamic_config(project_dir, checkpoints_detected, eval_detected)
        log_message(logger, "OK", "動的 config を生成完了")

        log_message(logger, "INFO", "=" * 60)
        log_message(logger, "INFO", "[STEP 1] Learning curves CSV を生成中...")
        log_message(logger, "INFO", "=" * 60)

        sys.path.insert(0, str(project_dir))
        from lib.reward_processor import RewardProcessor

        processor = RewardProcessor(checkpoints_dir, {name: phase['scale_factors'] for name, phase in config['phases'].items()})

        all_data = {}
        for phase_name, checkpoint_path in config['checkpoints'].items():
            try:
                log_message(logger, "INFO", f"処理中: {phase_name}")
                normalized = processor.process_phase(phase_name, checkpoint_path)
                for step, data in normalized.items():
                    if step not in all_data:
                        all_data[step] = {'timestep': step}
                    all_data[step][f'{phase_name}_reward'] = data['total']
                    all_data[step][f'{phase_name}_lin_vel'] = data['lin_vel']
                    all_data[step][f'{phase_name}_ang_vel'] = data['ang_vel']
                    all_data[step][f'{phase_name}_alive'] = data['alive']
                log_message(logger, "OK", f"  {len(normalized)} ステップを処理完了")
            except Exception as e:
                log_message(logger, "WARN", f"スキップ: {phase_name} - {str(e)}")

        output_csv = run_output_dir / '03_learning_curves.csv'
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['timestep'] + [f'{pn}_{comp}' for pn in config['phases'].keys() for comp in ['reward','lin_vel','ang_vel','alive']]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for step in sorted(all_data.keys()):
                writer.writerow(all_data[step])

        log_message(logger, "OK", f"CSV 生成完了: {output_csv}")

        log_message(logger, "INFO", "=" * 60)
        log_message(logger, "INFO", "[STEP 2] グラフを生成中...")
        log_message(logger, "INFO", "=" * 60)

        from lib.graph_generator import GraphGenerator

        graph_output_dir = run_output_dir / 'graphs'
        graph_output_dir.mkdir(parents=True, exist_ok=True)

        graph_gen = GraphGenerator(project_dir)

        log_message(logger, "INFO", "学習曲線グラフを生成中...")
        if graph_gen.generate_learning_curves(run_output_dir):
            log_message(logger, "OK", "学習曲線グラフ生成完了")
        else:
            log_message(logger, "WARN", "学習曲線グラフ生成に警告あり")

        log_message(logger, "INFO", "推論結果グラフを生成中...")
        if graph_gen.generate_inference_graphs(run_output_dir):
            log_message(logger, "OK", "推論結果グラフ生成完了")
        else:
            log_message(logger, "WARN", "推論結果グラフ生成に警告あり")

        log_message(logger, "INFO", "フェーズ別追従グラフを生成中...")
        if graph_gen.generate_phase_comparison_graphs():
            log_message(logger, "OK", "フェーズ別追従グラフ生成完了")
        else:
            log_message(logger, "WARN", "フェーズ別追従グラフ生成に警告あり")

        log_message(logger, "INFO", "=" * 60)
        log_message(logger, "INFO", "[STEP 3] メタデータを生成中...")
        log_message(logger, "INFO", "=" * 60)

        metadata = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "project_dir": str(project_dir),
            "auto_detected": {
                "checkpoints": checkpoints_detected,
                "eval_results": eval_detected,
            },
            "output_csv": str(output_csv),
            "graphs_dir": str(graph_output_dir),
        }

        metadata_file = run_output_dir / 'metadata.json'
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        log_message(logger, "OK", f"Metadata 生成完了: {metadata_file}")

        log_message(logger, "INFO", "=" * 60)
        log_message(logger, "INFO", "[STEP 4] サマリーを生成中...")
        log_message(logger, "INFO", "=" * 60)

        summary_text = f"""
解析実行サマリー
===============================================
実行ID: {run_id}
実行時刻: {datetime.now().strftime('%Y年%m月%d日 %H:%M:%S')}

【自動検出結果】
チェックポイント:
{chr(10).join(f'  - {phase}: {cp}' for phase, cp in checkpoints_detected.items())}

Eval Results:
{chr(10).join(f'  - {label}: {filename}' for label, filename in eval_detected.items())}

【生成ファイル】
  - 03_learning_curves.csv      (正規化済み学習曲線データ)
  - graphs/total_reward_normalized.png
  - graphs/tracking_lin_vel_normalized.png
  - graphs/tracking_ang_vel_normalized.png
  - graphs/alive_normalized.png
  - graphs/inference_rmse_comparison.png
  - execution_log.txt            (処理ログ)
  - metadata.json                (実行メタデータ)

===============================================
"""

        summary_file = run_output_dir / 'summary.txt'
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write(summary_text)

        log_message(logger, "OK", f"サマリー生成完了: {summary_file}")

        log_message(logger, "INFO", "=" * 60)
        log_message(logger, "INFO", "[STEP 5] 最新実行情報を更新中...")
        log_message(logger, "INFO", "=" * 60)

        latest_file = transfer_dir / 'analysis_output' / 'latest.json'
        latest_info = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "summary_path": str(summary_file),
            "log_path": str(log_file),
        }

        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(latest_info, f, ensure_ascii=False, indent=2)

        log_message(logger, "OK", f"Latest 情報を更新: {latest_file}")

        log_message(logger, "OK", "=" * 60)
        log_message(logger, "OK", "全処理完了！")
        log_message(logger, "OK", f"結果フォルダ: {run_output_dir}")
        log_message(logger, "OK", "=" * 60)

        return True

    except Exception as e:
        log_message(logger, "NG", f"エラーが発生しました: {str(e)}")
        log_message(logger, "NG", traceback.format_exc())
        return False

    finally:
        logger.close()

def main():
    """メイン処理"""

    project_dir = Path(__file__).parent
    transfer_dir = Path.home() / 'Desktop/iizawa/iizawa_workspace/transfer'

    if not transfer_dir.exists():
        print(f"[NG] Transfer ディレクトリが見つかりません: {transfer_dir}")
        sys.exit(1)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"\n{'='*60}")
    print(f"LEAP Hand 学習結果解析システム (リモート版 - 自動検出)")
    print(f"{'='*60}")
    print(f"実行ID: {run_id}")
    print(f"プロジェクト: {project_dir}")
    print(f"出力先: {transfer_dir}/analysis_output/{run_id}/")
    print(f"{'='*60}\n")

    success = run_analysis(project_dir, transfer_dir, run_id)

    if success:
        print(f"\n✓ 解析完了。結果を確認してください。")
        sys.exit(0)
    else:
        print(f"\n✗ 解析中にエラーが発生しました。ログを確認してください。")
        sys.exit(1)

if __name__ == '__main__':
    main()
