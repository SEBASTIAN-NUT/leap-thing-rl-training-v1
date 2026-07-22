#!/usr/bin/env python3
"""
Phase 4: Open Duck 正則化パラメータ クイックチェック

Open Duck Mini V2 で元々アクティブだった torques / action_rate が
現在ゼロになっているため、3段階の強度で動きへの影響を確認する。

目的:
    深くチューニングするのではなく「どの強度なら動きに影響が出るか」を確認するだけ。
    結果を見て Phase 5 (Go1 terms) のベース設定を決める。

実行方法 (プロジェクトルートから):
    Open_Duck_Playground/.venv/bin/python experiments/phase4_reward_sweep.py
    Open_Duck_Playground/.venv/bin/python experiments/phase4_reward_sweep.py --configs light original
    Open_Duck_Playground/.venv/bin/python experiments/phase4_reward_sweep.py --also_run_alive03

候補:
    light    : torques=-1e-4, action_rate=-0.1   (弱め、ほぼ影響なし確認用)
    original : torques=-1e-3, action_rate=-0.5   (Open Duck 元値)
    strong   : torques=-5e-3, action_rate=-1.0   (強め)
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

SCRIPT_DIR = Path(__file__).resolve().parent.parent
PYTHON = SCRIPT_DIR / "Open_Duck_Playground" / ".venv" / "bin" / "python"

DEFAULT_TIMESTEPS = 50_000_000
SLEEP_BETWEEN_RUNS = 300

# Phase 3 決定済みパラメータ
PHASE3_BEST = {
    "reward_config.scales.alive":            0.5,
    "reward_config.scales.tracking_lin_vel": 3.0,
    "reward_config.scales.tracking_ang_vel": 1.0,
    "reward_config.tracking_sigma_lin":      0.025,
    "reward_config.tracking_sigma_ang":      2.0,
}

# alive=0.3 リトライ用
ALIVE03_PARAMS = {
    **PHASE3_BEST,
    "reward_config.scales.alive": 0.3,
}

# ============================================================
# Open Duck 正則化 3段階
# ============================================================
# Open Duck Mini V2 元値: torques=-1e-3, action_rate=-0.5
# Go1 元値:              torques=-2e-4, action_rate=-0.01

CONFIGS = {
    "light": {
        "description": "弱め (ほぼ影響なし確認用)",
        "extra": {
            "reward_config.scales.torques":      -1e-4,
            "reward_config.scales.action_rate":  -0.1,
        },
    },
    "original": {
        "description": "Open Duck 元値",
        "extra": {
            "reward_config.scales.torques":      -1e-3,
            "reward_config.scales.action_rate":  -0.5,
        },
    },
    "strong": {
        "description": "強め (どこまで効くか確認)",
        "extra": {
            "reward_config.scales.torques":      -5e-3,
            "reward_config.scales.action_rate":  -1.0,
        },
    },
}

DEFAULT_CONFIGS = ["light", "original", "strong"]

# ============================================================
# 実験結果 (参考)
# ============================================================
# (未実施)

# ============================================================
# ランナー
# ============================================================

def _set_stack_unlimited():
    try:
        resource.setrlimit(resource.RLIMIT_STACK,
                           (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except (ValueError, resource.error):
        pass


def run_single(overrides: dict, output_dir: str, num_timesteps: int) -> bool:
    jax_cache = tempfile.mkdtemp(prefix="jax_cache_")
    env = {
        **os.environ,
        "JAX_COMPILATION_CACHE_DIR": jax_cache,
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
    parser = argparse.ArgumentParser(description="Phase 4: Open Duck 正則化クイックチェック")
    parser.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                        choices=list(CONFIGS.keys()),
                        help="実行する設定 (default: light original strong)")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--also_run_alive03", action="store_true",
                        help="Phase 3 未取得の alive=0.3 を先頭に追加")
    args = parser.parse_args()

    _set_stack_unlimited()

    runs = []
    if args.also_run_alive03:
        runs.append(("p3_alive_0p3", "Phase 3 リトライ: alive=0.3", ALIVE03_PARAMS))
    for key in args.configs:
        c = CONFIGS[key]
        overrides = {**PHASE3_BEST, **c["extra"]}
        runs.append((f"p4_{key}", f"[{key}] {c['description']}", overrides))

    print("\n" + "=" * 60)
    print("Phase 4: Open Duck 正則化クイックチェック")
    print(f"  total runs: {len(runs)}  |  timesteps: {args.timesteps:,}")
    print()
    for slug, label, overrides in runs:
        extra = {k: v for k, v in overrides.items() if k not in PHASE3_BEST}
        print(f"  {label}")
        if extra:
            print(f"    extra: {extra}")
    print("=" * 60)

    succeeded, failed = [], []
    for i, (slug, label, overrides) in enumerate(runs, 1):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = f"checkpoints/{slug}_{ts}"
        print(f"\n[{i}/{len(runs)}] {label}")
        print(f"  overrides: {json.dumps(overrides)}")

        ok = run_single(overrides, run_dir, args.timesteps)
        if ok:
            succeeded.append(label)
            print("  => SUCCESS")
        else:
            failed.append(label)
            print("  => FAILED  (reboot 後に再実行)")

        if i < len(runs):
            print(f"  Sleeping {SLEEP_BETWEEN_RUNS}s...")
            time.sleep(SLEEP_BETWEEN_RUNS)

    print("\n" + "=" * 60)
    print(f"SUCCEEDED: {succeeded}")
    if failed:
        print(f"FAILED   : {failed}")
    print(f"\n次のステップ: phase5_go1_terms.py で Go1 の項目を 1 つずつ追加")
    print(f"TensorBoard: tensorboard --logdir {SCRIPT_DIR}/checkpoints --port 6006")
    print("=" * 60)


if __name__ == "__main__":
    main()
