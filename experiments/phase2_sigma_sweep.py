#!/usr/bin/env python3
"""
Phase 2: Tracking sensitivity (sigma) sweep

Phase 2a: sigma_lin を変化、sigma_ang を固定
Phase 2b: sigma_ang を変化、sigma_lin を固定

実行方法 (プロジェクトルートから):
    Open_Duck_Playground/.venv/bin/python experiments/phase2_sigma_sweep.py
    Open_Duck_Playground/.venv/bin/python experiments/phase2_sigma_sweep.py --phase 2a
    Open_Duck_Playground/.venv/bin/python experiments/phase2_sigma_sweep.py --phase 2b
    Open_Duck_Playground/.venv/bin/python experiments/phase2_sigma_sweep.py --timesteps 50000000
"""
import argparse
import json
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
SLEEP_BETWEEN_RUNS = 300   # seconds; 300s が consecutive crash 回避に有効と判明

# Phase 2 共通の固定パラメータ (sigma sweep 時点での設定)
PHASE2_COMMON_FIXED = {
    "reward_config.scales.tracking_lin_vel": 1.5,
    "reward_config.scales.tracking_ang_vel": 0.5,
    "reward_config.scales.alive":            1.0,
}

# ------ Phase 2a: sigma_lin sweep ------
PHASE2A = {
    "description": "sigma_lin sweep (sigma_ang=0.25 fixed)",
    "sweep_key":   "reward_config.tracking_sigma_lin",
    "candidates":  [0.0025, 0.005, 0.01, 0.025, 0.05],
    "fixed": {
        **PHASE2_COMMON_FIXED,
        "reward_config.tracking_sigma_ang": 0.25,
    },
    "prefix": "p2a_sigma_lin",
}

# ------ Phase 2b: sigma_ang sweep ------
PHASE2B = {
    "description": "sigma_ang sweep (sigma_lin=0.01 fixed)",
    "sweep_key":   "reward_config.tracking_sigma_ang",
    "candidates":  [0.1, 0.25, 0.5, 1.0, 2.0],
    "fixed": {
        **PHASE2_COMMON_FIXED,
        "reward_config.tracking_sigma_lin": 0.01,
    },
    "prefix": "p2b_sigma_ang",
}

# ============================================================
# 実験結果 (参考)
# ============================================================
# Phase 2a 結果: sigma_lin=0.025 が最良
#   - 0.0025: 疎すぎ (学習初期に勾配なし)
#   - 0.005 : やや疎
#   - 0.01  : 普通
#   - 0.025 : 良好 ← 採用
#   - 0.05  : やや緩
#
# Phase 2b 結果: sigma_ang=2.0 が最良
#   - 0.1 〜 0.5: 疎すぎ
#   - 1.0       : 良好
#   - 2.0       : 良好 ← 採用 (yaw コマンドのスケールに合致)
#
# Phase 3 へ引き継ぎ: sigma_lin=0.025, sigma_ang=2.0

# ============================================================
# ランナー
# ============================================================

def _set_stack_unlimited():
    """ptxas のスタックオーバーフローを防ぐ (Linux only)."""
    try:
        resource.setrlimit(resource.RLIMIT_STACK,
                           (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except (ValueError, resource.error):
        pass  # 権限がなければ無視


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


def run_phase(phase: dict, num_timesteps: int):
    _set_stack_unlimited()
    print("=" * 60)
    print(f"Phase: {phase['description']}")
    print(f"  sweep : {phase['sweep_key']}")
    print(f"  values: {phase['candidates']}")
    print(f"  fixed : {json.dumps(phase['fixed'], indent=4)}")
    print(f"  timesteps: {num_timesteps:,}")
    print("=" * 60)

    succeeded = []
    failed = []
    total = len(phase["candidates"])

    for i, value in enumerate(phase["candidates"], 1):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = str(value).replace(".", "p")
        run_dir = f"checkpoints/{phase['prefix']}_{slug}_{ts}"
        overrides = {phase["sweep_key"]: value, **phase["fixed"]}

        print(f"\n[{i}/{total}] {phase['sweep_key']}={value}  ->  {run_dir}")
        print(f"  overrides: {json.dumps(overrides)}")

        ok = run_single(overrides, run_dir, num_timesteps)
        if ok:
            succeeded.append(value)
            print(f"  => SUCCESS")
        else:
            failed.append(value)
            print(f"  => FAILED")

        if i < total:
            print(f"  Sleeping {SLEEP_BETWEEN_RUNS}s before next run...")
            time.sleep(SLEEP_BETWEEN_RUNS)

    print("\n" + "=" * 60)
    print(f"Done. SUCCEEDED: {succeeded}")
    if failed:
        print(f"       FAILED  : {failed}")
    print(f"TensorBoard: tensorboard --logdir {SCRIPT_DIR}/checkpoints --port 6006")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["2a", "2b", "all"], default="all",
                        help="実行するフェーズ (default: all = 2a then 2b)")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    args = parser.parse_args()

    if args.phase in ("2a", "all"):
        run_phase(PHASE2A, args.timesteps)
    if args.phase in ("2b", "all"):
        run_phase(PHASE2B, args.timesteps)


if __name__ == "__main__":
    main()
