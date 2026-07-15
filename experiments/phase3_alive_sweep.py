#!/usr/bin/env python3
"""
Phase 3: Alive scale sweep

Phase 2 で決定した sigma を固定し、alive スケールを変化させて
学習への影響を比較する。

実行方法 (プロジェクトルートから):
    Open_Duck_Playground/.venv/bin/python experiments/phase3_alive_sweep.py
    Open_Duck_Playground/.venv/bin/python experiments/phase3_alive_sweep.py --values 0.1 0.2 0.5
    Open_Duck_Playground/.venv/bin/python experiments/phase3_alive_sweep.py --timesteps 50000000

注意:
    ・連続2ランを超えるとコンパイル時 ptxas クラッシュ (SIGSEGV) が発生する。
      3本目以降は sudo reboot 後に実行すること。
    ・alive=0.0, 0.05 は訓練フェーズで別のクラッシュ (エピソード即死 → PPO バッチ問題?) が
      発生するため通常は実行しない。
"""
import argparse
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

# ============================================================
# 実験設定
# ============================================================

SCRIPT_DIR = Path(__file__).resolve().parent.parent  # project root
PYTHON = SCRIPT_DIR / "Open_Duck_Playground" / ".venv" / "bin" / "python"

DEFAULT_TIMESTEPS = 150_000_000
SLEEP_BETWEEN_RUNS = 300  # 300s が consecutive crash 回避に有効

# Phase 2 の結果を受けて固定するパラメータ
FIXED_PARAMS = {
    "reward_config.scales.tracking_lin_vel": 3.0,
    "reward_config.scales.tracking_ang_vel": 1.0,
    "reward_config.tracking_sigma_lin":      0.025,
    "reward_config.tracking_sigma_ang":      2.0,
}

# alive sweep 候補
# 注: 0.0, 0.05 はクラッシュのため通常スキップ
DEFAULT_ALIVE_VALUES = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
SAFE_ALIVE_VALUES    = [0.1, 0.2, 0.3, 0.5, 1.0]  # 0.0/0.05 を除外

# ============================================================
# 事前シミュレーション (reward_check.py と同じ計算)
# ============================================================

def _presim(alive_values):
    """alive スケールごとの報酬割合を表示する。"""
    sigma_l, sigma_a = 0.025, 2.0
    trk_l,  trk_a   = 3.0,   1.0
    cmd_vx,  cmd_yaw = 0.15,  1.0

    print("\n[事前シミュレーション] alive が総報酬に占める割合")
    effs = [0.0, 0.13, 0.25, 0.5, 1.0]
    header = f"{'alive':>6} | " + "".join(f" eff={int(e*100):3d}%" for e in effs) + "  | 判定"
    print(header)
    print("-" * len(header))
    for a in alive_values:
        row = f"{a:6.2f} | "
        ratios = []
        for e in effs:
            r_l = trk_l * math.exp(-((cmd_vx  * (1 - e)) ** 2) / sigma_l)
            r_a = trk_a * math.exp(-((cmd_yaw * (1 - e)) ** 2) / sigma_a)
            total = a + r_l + r_a
            ratio = a / total if total > 0 else 0.0
            ratios.append(ratio)
            row += f"  {ratio*100:5.1f}%"
        r0 = ratios[0]
        if a == 0.0:     verdict = "[生存圧なし]"
        elif r0 > 0.5:   verdict = "[alive 支配]"
        elif r0 > 0.3:   verdict = "[やや大きい]"
        elif r0 < 0.05:  verdict = "[生存圧が弱い]"
        else:            verdict = "[良好]"
        print(row + f"  | {verdict}")
    print()

# ============================================================
# 実験結果 (参考)
# ============================================================
# alive=0.0  : CRASH (訓練フェーズ SIGSEGV, 原因: エピソード即死→PPO問題?)
# alive=0.05 : CRASH (同上)
# alive=0.1  : SUCCESS  peak 13.57 @126M  elapsed 3:31h
# alive=0.2  : SUCCESS  peak 14.01 @126M  elapsed 3:31h
# alive=0.3  : CRASH (コンパイル ptxas SIGSEGV, 2連続成功後)  → 未取得
# alive=0.5  : SUCCESS  peak 18.34 @126M  elapsed 3:31h  ← 最良
# alive=1.0  : 実行中 / 未完了

# ============================================================
# ランナー
# ============================================================

def _set_stack_unlimited():
    """ptxas の stack overflow を防ぐ (Linux only)."""
    try:
        resource.setrlimit(resource.RLIMIT_STACK,
                           (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except (ValueError, resource.error):
        pass


def run_single(overrides: dict, output_dir: str, num_timesteps: int) -> bool:
    """1ランを JAX キャッシュ分離で実行。成功なら True を返す。"""
    jax_cache = tempfile.mkdtemp(prefix="jax_cache_")
    env = {
        **os.environ,
        "JAX_COMPILATION_CACHE_DIR": jax_cache,
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    cmd = [
        str(PYTHON), "-m", "thing_test.runner",
        "--env", "joystick",
        "--task", "flat_terrain",
        "--num_timesteps", str(num_timesteps),
        "--output_dir", output_dir,
        "--config_overrides", json.dumps(overrides),
    ]
    log_path = f"/tmp/{Path(output_dir).name}.log"
    print(f"  log: {log_path}")
    try:
        with open(log_path, "wb") as log:
            result = subprocess.run(cmd, env=env, cwd=str(SCRIPT_DIR),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            log.write(result.stdout)
            output = result.stdout.decode("utf-8", errors="replace")
            print(output[-3000:] if len(output) > 3000 else output)
        return result.returncode == 0
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False
    finally:
        shutil.rmtree(jax_cache, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Phase 3: alive scale sweep")
    parser.add_argument("--values", type=float, nargs="+", default=SAFE_ALIVE_VALUES,
                        help=f"alive 候補 (default: {SAFE_ALIVE_VALUES})")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--skip-presim", action="store_true")
    args = parser.parse_args()

    _set_stack_unlimited()

    if not args.skip_presim:
        _presim(args.values)

    print("=" * 60)
    print("Phase 3: alive scale sweep")
    print(f"  values    : {args.values}")
    print(f"  fixed     : {json.dumps(FIXED_PARAMS, indent=4)}")
    print(f"  timesteps : {args.timesteps:,}")
    print(f"  sleep     : {SLEEP_BETWEEN_RUNS}s between runs")
    print("=" * 60)

    succeeded = []
    failed = []
    total = len(args.values)

    for i, alive in enumerate(args.values, 1):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = str(alive).replace(".", "p")
        run_dir = f"checkpoints/p3_alive_{slug}_{ts}"
        overrides = {"reward_config.scales.alive": alive, **FIXED_PARAMS}

        print(f"\n[{i}/{total}] alive={alive}  ->  {run_dir}")
        print(f"  overrides: {json.dumps(overrides)}")

        ok = run_single(overrides, run_dir, args.timesteps)
        if ok:
            succeeded.append(alive)
            print(f"  => SUCCESS")
        else:
            failed.append(alive)
            print(f"  => FAILED  (ptxas/SIGSEGV なら reboot 後に再実行)")

        if i < total:
            print(f"  Sleeping {SLEEP_BETWEEN_RUNS}s...")
            time.sleep(SLEEP_BETWEEN_RUNS)

    print("\n" + "=" * 60)
    print(f"SUCCEEDED: {succeeded}")
    if failed:
        print(f"FAILED   : {failed}")
        print("  => 失敗したものは sudo reboot 後に個別実行:")
        for v in failed:
            slug = str(v).replace(".", "p")
            print(f"     python experiments/phase3_alive_sweep.py --values {v}")
    print(f"\nTensorBoard: tensorboard --logdir {SCRIPT_DIR}/checkpoints --port 6006")
    print("=" * 60)


if __name__ == "__main__":
    main()
