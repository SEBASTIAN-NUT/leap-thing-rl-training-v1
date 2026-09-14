# -*- coding: utf-8 -*-
import os
import subprocess
import sys
from pathlib import Path


class GraphGenerator:

    def __init__(self, project_dir=None):
        self.project_dir = project_dir or Path(__file__).parent.parent

    def generate_learning_curves(self, run_output_dir=None):
        print("\n" + "="*60)
        print("学習曲線グラフを��成中...")
        print("="*60)
        script = self.project_dir / "generate_learning_curves_png.py"
        if not script.exists():
            print(f"[WARN] スクリプトが見つかりません: {script}")
            return False
        env = dict(os.environ)
        if run_output_dir:
            csv_path = Path(run_output_dir) / "03_learning_curves.csv"
            if csv_path.exists():
                env["CSV_FILE"] = str(csv_path)
                env["OUTPUT_FOLDER"] = str(run_output_dir)
            else:
                print(f"[WARN] CSVが見つかりません: {csv_path}")
                return False
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=self.project_dir, capture_output=True,
                text=True, errors="replace", timeout=300, env=env,
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

    def generate_phase_comparison_graphs(self, eval_results_dir=None, out_dir=None):
        print("\n" + "="*60)
        print("フェーズ別追従グラフを生成中...")
        print("="*60)
        script = self.project_dir / "experiments" / "plot_tracking.py"
        if not script.exists():
            print(f"[WARN] スクリプトが見つかりません: {script}")
            return False
        results_dir = eval_results_dir or str(self.project_dir / "eval_results" / "archived")
        output_dir  = out_dir         or str(self.project_dir / "poster_figures" / "comparison")
        try:
            result = subprocess.run(
                [sys.executable, str(script),
                 "--results_dir", results_dir,
                 "--out_dir", output_dir,
                 "--dpi", "150"],
                cwd=str(self.project_dir), capture_output=True,
                text=True, errors="replace", timeout=300,
            )
            if result.returncode == 0:
                print(result.stdout)
                print("[OK] フェーズ別追従グラフ生成完了")
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

    def normalize_rewards(self):
        print("\n" + "="*60)
        print("報酬を正規化中...")
        print("="*60)
        script = self.project_dir / "normalize_reward_scalars.py"
        if not script.exists():
            print(f"[WARN] スクリプトが見つかりません: {script}")
            return False
        try:
            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=self.project_dir, capture_output=True,
                text=True, errors="replace", timeout=300
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
