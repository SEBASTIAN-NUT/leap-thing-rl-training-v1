#!/usr/bin/env python3
"""
Phase 5: Go1 報酬項目の個別追加・チューニング

現在ゼロになっている Go1 由来の報酬項目を 1 つずつ追加し、
各項目のスケールを値のスイープで調整する。

進め方:
    ① 1 つの項目を --term で指定し、複数の値をスイープする
    ② TensorBoard と check_command_response.py で動きを確認
    ③ 最良値を --fixed_extra に追加して次の項目へ

実行例 (プロジェクトルートから):
    # Step 1: stand_still を調整
    Open_Duck_Playground/.venv/bin/python experiments/phase5_go1_terms.py --term stand_still

    # Step 2: stand_still=-1.0 を固定して orientation を調整
    Open_Duck_Playground/.venv/bin/python experiments/phase5_go1_terms.py \\
        --term orientation \\
        --fixed_extra '{"reward_config.scales.stand_still": -1.0}'

    # 値を手動指定したい場合
    Open_Duck_Playground/.venv/bin/python experiments/phase5_go1_terms.py \\
        --term feet_air_time --values 0.05 0.2 0.5

利用可能な項目 (--term の選択肢):
    stand_still   : stop コマンド時のドリフト抑制 (観測: vx=+0.057 @ cmd=0)
    orientation   : 手の傾き抑制
    ang_vel_xy    : ローリング・ピッチング抑制
    feet_air_time : 指先の空中時間報酬 (歩行リズム)
    energy        : エネルギー消費抑制 (|vel|×|torque|)
    pose          : ホームポーズ維持報酬
    dof_pos_limits: 関節可動域違反ペナルティ
    termination   : エピソード終了ペナルティ
    lin_vel_z     : 垂直速度ペナルティ (バウンス抑制)
    ang_vel_xy    : 水平角速度ペナルティ
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

DEFAULT_TIMESTEPS = 150_000_000
SLEEP_BETWEEN_RUNS = 300

# ============================================================
# ベースパラメータ
# ============================================================
# Phase 3 + Phase 4 で確定した固定値。
# Phase 4 の結果を受けて torques/action_rate を更新すること。
# (Phase 4 で "original" が最良なら下記のままでよい)

PHASE_BASE = {
    # Phase 3 固定
    "reward_config.scales.alive":            0.5,
    "reward_config.scales.tracking_lin_vel": 3.0,
    "reward_config.scales.tracking_ang_vel": 1.0,
    "reward_config.tracking_sigma_lin":      0.025,
    "reward_config.tracking_sigma_ang":      2.0,
    # Phase 4 固定 (Phase 4 結果を見て更新)
    # "reward_config.scales.torques":       -1e-3,   # Phase 4 best
    # "reward_config.scales.action_rate":   -0.5,    # Phase 4 best
}

# ============================================================
# Go1 項目の定義
# ============================================================
# values: デフォルトスイープ候補
# go1   : Go1 の参考値
# note  : 期待される効果

TERMS = {
    "stand_still": {
        "key":    "reward_config.scales.stand_still",
        "values": [-0.5, -1.0, -2.0],
        "go1":    -1.0,
        "note":   "stop コマンド時のドリフト抑制 (観測済み vx=+0.057, yaw=-0.191)",
    },
    "orientation": {
        "key":    "reward_config.scales.orientation",
        "values": [-0.5, -1.0, -5.0],
        "go1":    -5.0,
        "note":   "手の傾き抑制",
    },
    "ang_vel_xy": {
        "key":    "reward_config.scales.ang_vel_xy",
        "values": [-0.02, -0.05, -0.2],
        "go1":    -0.05,
        "note":   "ローリング・ピッチング抑制",
    },
    "feet_air_time": {
        "key":    "reward_config.scales.feet_air_time",
        "values": [0.05, 0.1, 0.5],
        "go1":    0.1,
        "note":   "指先の空中時間報酬 (歩行リズム促進)",
    },
    "energy": {
        "key":    "reward_config.scales.energy",
        "values": [-5e-4, -1e-3, -5e-3],
        "go1":    -1e-3,
        "note":   "エネルギー消費抑制 (|vel|×|torque|)",
    },
    "pose": {
        "key":    "reward_config.scales.pose",
        "values": [0.2, 0.5, 1.0],
        "go1":    0.5,
        "note":   "ホームポーズ維持報酬 (exp(-||q-q0||^2))",
    },
    "dof_pos_limits": {
        "key":    "reward_config.scales.dof_pos_limits",
        "values": [-0.5, -1.0, -2.0],
        "go1":    -1.0,
        "note":   "関節可動域違反ペナルティ",
    },
    "termination": {
        "key":    "reward_config.scales.termination",
        "values": [-0.5, -1.0, -2.0],
        "go1":    -1.0,
        "note":   "エピソード終了ペナルティ",
    },
    "lin_vel_z": {
        "key":    "reward_config.scales.lin_vel_z",
        "values": [-0.1, -0.5, -1.0],
        "go1":    -0.5,
        "note":   "垂直速度ペナルティ (上下バウンス抑制)",
    },
}

# ============================================================
# 実験結果 (参考)
# ============================================================
# stand_still  : 未実施
# orientation  : 未実施
# ang_vel_xy   : 未実施
# feet_air_time: 未実施
# energy       : 未実施
# pose         : 未実施

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
        with open(log_path, "w") as log:
            result = subprocess.run(cmd, env=env, cwd=str(SCRIPT_DIR),
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True)
            log.write(result.stdout)
            print(result.stdout[-3000:] if len(result.stdout) > 3000 else result.stdout)
        return result.returncode == 0
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False
    finally:
        shutil.rmtree(jax_cache, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Go1 報酬項目の個別追加",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {k:15s}: {v['note']}  (Go1={v['go1']}, default values={v['values']})"
            for k, v in TERMS.items()
        ),
    )
    parser.add_argument(
        "--term", required=True, choices=list(TERMS.keys()),
        help="スイープする報酬項目",
    )
    parser.add_argument(
        "--values", type=float, nargs="+", default=None,
        help="スイープする値 (省略時は TERMS のデフォルト値を使用)",
    )
    parser.add_argument(
        "--fixed_extra", type=str, default=None,
        help='前ステップで確定した追加パラメータ (JSON形式)\n'
             '例: \'{"reward_config.scales.stand_still": -1.0}\'',
    )
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    args = parser.parse_args()

    _set_stack_unlimited()

    term = TERMS[args.term]
    values = args.values if args.values is not None else term["values"]

    fixed_extra = {}
    if args.fixed_extra:
        try:
            fixed_extra = json.loads(args.fixed_extra)
        except json.JSONDecodeError as e:
            print(f"[ERROR] --fixed_extra のJSON解析失敗: {e}")
            sys.exit(1)

    base = {**PHASE_BASE, **fixed_extra}

    print("\n" + "=" * 60)
    print(f"Phase 5: {args.term} sweep")
    print(f"  {term['note']}")
    print(f"  key   : {term['key']}")
    print(f"  values: {values}  (Go1 参考値: {term['go1']})")
    print(f"  base  : {json.dumps(base, indent=4)}")
    if fixed_extra:
        print(f"  fixed_extra (前ステップ確定): {fixed_extra}")
    print(f"  timesteps: {args.timesteps:,}  |  runs: {len(values)}")
    print("=" * 60)

    succeeded, failed = [], []
    for i, val in enumerate(values, 1):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        val_str = str(val).replace(".", "p").replace("-", "m")
        run_dir = f"checkpoints/p5_{args.term}_{val_str}_{ts}"
        overrides = {**base, term["key"]: val}

        print(f"\n[{i}/{len(values)}] {args.term}={val}")
        print(f"  dir: {run_dir}")
        print(f"  overrides: {json.dumps(overrides)}")

        ok = run_single(overrides, run_dir, args.timesteps)
        if ok:
            succeeded.append(val)
            print("  => SUCCESS")
        else:
            failed.append(val)
            print("  => FAILED  (reboot 後に再実行)")

        if i < len(values):
            print(f"  Sleeping {SLEEP_BETWEEN_RUNS}s...")
            time.sleep(SLEEP_BETWEEN_RUNS)

    print("\n" + "=" * 60)
    print(f"SUCCEEDED: {succeeded}")
    if failed:
        print(f"FAILED   : {failed}")
        for v in failed:
            val_str = str(v).replace(".", "p").replace("-", "m")
            print(f"  再実行: python experiments/phase5_go1_terms.py --term {args.term} --values {v}")
    print()
    print("次ステップ: 最良値を --fixed_extra に追加して次の項目へ")
    print(f"  例) python experiments/phase5_go1_terms.py \\")
    best_guess = succeeded[0] if succeeded else values[0]
    print(f"       --term <次の項目> \\")
    print(f"       --fixed_extra '{{...前の確定値..., \"{term[\"key\"]}\": {best_guess}}}'")
    print(f"\nTensorBoard: tensorboard --logdir {SCRIPT_DIR}/checkpoints --port 6006")
    print("=" * 60)


if __name__ == "__main__":
    main()
