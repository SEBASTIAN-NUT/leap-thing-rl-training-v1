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

DEFAULT_TIMESTEPS = 50_000_000
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
    jax_cache = str(Path(__file__).resolve().parent.parent / ".jax_cache")
    Path(jax_cache).mkdir(exist_ok=True)
    env = {**__import__("os").environ, "JAX_COMPILATION_CACHE_DIR": jax_cache}
    cmd = [
        str(PYTHON), "-m", "thing_test.runner",
        "--env", "joystick", "--task", "flat_terrain",
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
            out = result.stdout.decode("utf-8", errors="replace")
        if result.returncode != 0:
            lines = out.splitlines()
            print(f"  [FAILED] exit_code={result.returncode}  log={log_path}")
            if len(lines) <= 60:
                print(out)
            else:
                print("\n--- first 20 lines ---")
                print("\n".join(lines[:20]))
                print(f"\n... ({len(lines) - 40} lines omitted) ...\n")
                print("--- last 20 lines ---")
                print("\n".join(lines[-20:]))
        else:
            print(out[-3000:] if len(out) > 3000 else out)
        return result.returncode == 0
    except Exception as e:
        print(f"  [ERROR] {e}")
        return False
    finally:
        pass


def find_latest_onnx(output_dir: str):
    def step_num(p):
        try:
            return int(p.stem.rsplit("_", 1)[-1])
        except ValueError:
            return -1
    candidates = list(Path(output_dir).glob("*.onnx"))
    return max(candidates, key=step_num) if candidates else None


def run_eval(onnx_path, json_out: str):
    Path(json_out).parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(PYTHON), "-m", "thing_test.check_command_response",
        "-o", str(onnx_path), "--no_viewer",
        "--axes", "vx,yaw", "--fractions", "0.25,0.5,0.75,1.0",
        "--phase_duration", "4.0", "--json_out", json_out,
    ]
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout.decode("utf-8", errors="replace"))
    if result.returncode != 0 or not Path(json_out).exists():
        return None
    with open(json_out, encoding="utf-8") as f:
        return json.load(f)


def run_term(term_name: str, values: list, base: dict,
             timesteps: int, do_eval: bool):
    """1項目のスイープを実行。(best_val, succeeded, failed) を返す。"""
    term = TERMS[term_name]
    succeeded, failed, eval_results = [], [], {}

    for i, val in enumerate(values, 1):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        val_str = str(val).replace(".", "p").replace("-", "m")
        run_dir = f"checkpoints/p5_{term_name}_{val_str}_{ts}"
        overrides = {**base, term["key"]: val}

        print(f"\n[{i}/{len(values)}] {term_name}={val}")
        print(f"  dir: {run_dir}")
        print(f"  overrides: {json.dumps(overrides)}")

        ok = run_single(overrides, run_dir, timesteps)
        if ok:
            succeeded.append(val)
            print("  => SUCCESS")
            if do_eval:
                onnx = find_latest_onnx(run_dir)
                if onnx:
                    jout = f"eval_results/p5_{term_name}_{val_str}.json"
                    print(f"  [eval] {onnx.name} -> {jout}")
                    res = run_eval(onnx, jout)
                    if res:
                        eval_results[val] = {"rmse_vx": res["rmse_vx"],
                                             "rmse_yaw": res["rmse_yaw"]}
                        print(f"  [eval] RMSE_vx={res['rmse_vx']:.4f}  "
                              f"RMSE_yaw={res['rmse_yaw']:.4f}")
                    else:
                        print("  [eval] 評価失敗")
                else:
                    print("  [eval] ONNX が見つからない")
        else:
            failed.append(val)
            print("  => FAILED")

        if i < len(values):
            print(f"  Sleeping {SLEEP_BETWEEN_RUNS}s...")
            time.sleep(SLEEP_BETWEEN_RUNS)

    best_val = None
    if eval_results:
        best_val = min(eval_results, key=lambda v: eval_results[v]["rmse_vx"])
        print(f"\n=== {term_name} RMSE 比較 ===")
        for v, r in sorted(eval_results.items()):
            mark = " <- best vx" if v == best_val else ""
            print(f"  {v:>8}  RMSE_vx={r['rmse_vx']:.4f}  "
                  f"RMSE_yaw={r['rmse_yaw']:.4f}{mark}")
        print(f"=> 採用: {term_name}={best_val}")
    elif succeeded:
        best_val = succeeded[-1]
        print(f"=> (eval なし) 最後の成功値を採用: {term_name}={best_val}")

    return best_val, succeeded, failed


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5: Go1 報酬項目の個別/全項目連続最適化",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  {k:15s}: {v['note']}  (Go1={v['go1']}, values={v['values']})"
            for k, v in TERMS.items()
        ),
    )
    parser.add_argument("--term", choices=list(TERMS.keys()), default=None,
                        help="実行する項目 (--all_terms と排他)")
    parser.add_argument("--all_terms", action="store_true",
                        help="TERMS に定義された全項目を順番に実行")
    parser.add_argument("--values", type=float, nargs="+", default=None,
                        help="スイープ値を上書き (--term 単体使用時のみ有効)")
    parser.add_argument("--fixed_extra", type=str, default=None,
                        help='開始時に追加する固定パラメータ (JSON)')
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--auto_eval", action="store_true",
                        help="各学習後に check_command_response で RMSE を評価")
    args = parser.parse_args()

    if not args.term and not args.all_terms:
        parser.error("--term か --all_terms のどちらかを指定してください")
    if args.term and args.all_terms:
        parser.error("--term と --all_terms は同時に指定できません")

    fixed_extra = {}
    if args.fixed_extra:
        try:
            fixed_extra = json.loads(args.fixed_extra)
        except json.JSONDecodeError as e:
            print(f"[ERROR] --fixed_extra のJSON解析失敗: {e}")
            sys.exit(1)

    _set_stack_unlimited()

    terms_to_run = list(TERMS.keys()) if args.all_terms else [args.term]
    current_fixed = fixed_extra.copy()
    summary = {}   # term_name -> {"best": val, "succeeded": [...], "failed": [...]}

    for idx, term_name in enumerate(terms_to_run):
        term = TERMS[term_name]
        values = (args.values if (not args.all_terms and args.values)
                  else term["values"])
        base = {**PHASE_BASE, **current_fixed}

        print("\n" + "=" * 60)
        print(f"[{idx+1}/{len(terms_to_run)}] {term_name} sweep")
        print(f"  {term['note']}")
        print(f"  key    : {term['key']}")
        print(f"  values : {values}  (Go1={term['go1']})")
        print(f"  fixed  : {json.dumps(current_fixed)}")
        print(f"  steps  : {args.timesteps:,}")
        print("=" * 60)

        best_val, succeeded, failed = run_term(
            term_name, values, base, args.timesteps, args.auto_eval
        )
        summary[term_name] = {"best": best_val,
                               "succeeded": succeeded, "failed": failed}

        if best_val is not None:
            current_fixed[term["key"]] = best_val

        if idx < len(terms_to_run) - 1:
            print(f"\n次の項目まで {SLEEP_BETWEEN_RUNS}s 待機...")
            time.sleep(SLEEP_BETWEEN_RUNS)

    print("\n" + "=" * 60)
    print("=== 全項目完了 ===")
    print("確定した fixed_extra:")
    print(f"  {json.dumps(current_fixed, indent=4)}")
    print()
    for tname, info in summary.items():
        status = f"best={info['best']}"
        if info["failed"]:
            status += f"  failed={info['failed']}"
        print(f"  {tname:15s}: {status}")
    print(f"\nTensorBoard: tensorboard --logdir {SCRIPT_DIR}/checkpoints --port 6006")
    print("=" * 60)


if __name__ == "__main__":
    main()
