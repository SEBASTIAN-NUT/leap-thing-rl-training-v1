# -*- coding: utf-8 -*-
"""グラフ生成の共通機能"""
import subprocess
import sys
from pathlib import Path


class GraphGenerator:
    """既存のグラフ生成スクリプトをラップ"""

    def __init__(self, project_dir: Path = None):
        self.project_dir = project_dir or Path(__file__).parent.parent

    def generate_learning_curves(self, output_dir: Path = None):
        """学習曲線グラフを生成（generate_learning_curves_png.py を呼び出す）"""
        print("\n" + "="*60)
        print("学習曲線グラフを生成中...")
        print("="*60)

        if output_dir is None:
            output_dir = self.project_dir / 'analysis_output'

        csv_file = output_dir / '03_learning_curves.csv'

        if not csv_file.exists():
            print(f"[WARN] CSV ファイルが見つかりません: {csv_file}")
            return False

        script = self.project_dir / 'generate_learning_curves_png.py'
        if not script.exists():
            print(f"[WARN] スクリプトが見つかりません: {script}")
            return False

        try:
            env = {'CSV_FILE': str(csv_file), 'OUTPUT_FOLDER': str(output_dir)}
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                errors='replace',
                timeout=300,
                env={**dict(subprocess.os.environ), **env}
            )

            if result.returncode == 0:
                print(result.stdout)
                print("[OK] 学習曲線グラフを生成完了")
                return True
            else:
                print(f"[NG] エラー: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("[NG] タイムアウト")
            return False
        except Exception as e:
            print(f"[NG] エラー: {e}")
            return False

    def generate_inference_graphs(self, output_dir: Path = None):
        """推論結果グラフを生成（generate_inference_graphs.py を呼び出す）"""
        print("\n" + "="*60)
        print("推論結果グラフを生成中...")
        print("="*60)

        if output_dir is None:
            output_dir = self.project_dir / 'analysis_output'

        script = self.project_dir / 'generate_inference_graphs.py'
        if not script.exists():
            print(f"[WARN] スクリプトが見つかりません: {script}")
            return False

        try:
            env = {'OUTPUT_FOLDER': str(output_dir)}
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                errors='replace',
                timeout=300,
                env={**dict(subprocess.os.environ), **env}
            )

            if result.returncode == 0:
                print(result.stdout)
                print("[OK] 推論結果グラフを生成完了")
                return True
            else:
                print(f"[NG] エラー: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("[NG] タイムアウト")
            return False
        except Exception as e:
            print(f"[NG] エラー: {e}")
            return False

    def normalize_rewards(self, output_dir: Path = None):
        """報酬を正規化（normalize_reward_scalars.py を呼び出す）"""
        print("\n" + "="*60)
        print("報酬を正規化中...")
        print("="*60)

        if output_dir is None:
            output_dir = self.project_dir / 'analysis_output'

        script = self.project_dir / 'normalize_reward_scalars.py'
        if not script.exists():
            print(f"[WARN] スクリプトが見つかりません: {script}")
            return False

        try:
            env = {'OUTPUT_FOLDER': str(output_dir)}
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=self.project_dir,
                capture_output=True,
                text=True,
                errors='replace',
                timeout=300,
                env={**dict(subprocess.os.environ), **env}
            )

            if result.returncode == 0:
                print(result.stdout)
                print("[OK] 報酬を正規化完了")
                return True
            else:
                print(f"[NG] エラー: {result.stderr}")
                return False

        except subprocess.TimeoutExpired:
            print("[NG] タイムアウト")
            return False
        except Exception as e:
            print(f"[NG] エラー: {e}")
            return False
