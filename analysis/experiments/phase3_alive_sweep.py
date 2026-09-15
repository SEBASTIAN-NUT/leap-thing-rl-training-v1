import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
#!/usr/bin/env python3
"""
Phase 3: Alive scale sweep

Phase 2 で決定した sigma を固定し、alive スケールを変化させて
学習への影響を比較する。

実行方法 (プロジェクトルートから):
    Open_Duck_Playground/.venv/bin/python experiments/phase3_alive_sweep.py
    Open_Duck_Playground/.venv/bin/python experiments/phase3_alive_sweep.py --values 0.1 0.2 0.5
    Open_Duck_Playground/.venv/bin/python experiments/phase3_alive_sweep.py --auto_eval

選択基準: 速度追従 RMSE_vx の最小値 (報酬最大値ではない)

注意:
    ・連続2ランを超えるとコンパイル時 ptxas クラッシュ (SIGSEGV) が発生する。
      3本目以降は sudo reboot 後に実行すること。
    ・alive=0.0, 0.05 はクラッシュのため通常スキップ。
"""
import argparse
import json
import math
import os
import resource
import shutil
import subprocess
import tempfile
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON = SCRIPT_DIR / "Open_Duck_Playground" / ".venv" / "bin" / "python"

DEFAULT_TIMESTEPS = 150_000_000
SLEEP_BETWEEN_RUNS = 300

# !! Phase 2b 再実行後は tracking_sigma_lin と tracking_sigma_ang を更新すること !!
FIXED_PARAMS = {
    "reward_config.scales.tracking_lin_vel": 3.0,
    "reward_config.scales.tracking_ang_vel": 1.0,
    "reward_config.tracking_sigma_lin":      0.025,
    "reward_config.tracking_sigma_ang":      2.0,  # Phase 2b result
}

SAFE_ALIVE_VALUES = [0.1, 0.2, 0.3, 0.5, 1.0]


def _presim(alive_values):
    sigma_l, sigma_a = FIXED_PARAMS["reward_config.tracking_sigma_lin"], FIXED_PARAMS["reward_config.tracking_sigma_ang"]
    trk_l = FIXED_PARAMS["reward_config.scales.tracking_lin_vel"]
    trk_a = FIXED_PARAMS["reward_config.scales.tracking_ang_vel"]
    cmd_vx, cmd_yaw = 0.15, 1.0

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
        if a == 0.0:    verdict = "[生存圧なし]"
        elif r0 > 0.5:  verdict = "[alive 支配]"
        elif r0 > 0.3:  verdict = "[やや大きい]"
        elif r0 < 0.05: verdict = "[生存圧が弱い]"
        else:           verdict = "[良好]"
        print(row + f"  | {verdict}")
    print()


def find_latest_onnx(output_dir: str):
    def step_num(p):
        try:
            return int(p.stem.rsplit("_", 1)[-1])
        except ValueError:
            return -1
    candidates = list(Path(output_dir).glob("*.onnx"))
    return max(candidates, key=step_num) if candidates else None


def run_eval(onnx_path: Path, json_out: str) -> dict | None:
    Path(json_out).parent.mkdir(parents=True, exist_ok=True)
    csv_out = json_out.replace(".json", ".csv")
    cmd = [
        str(PYTHON), "-m", "thing_test.check_command_response",
        "-o", str(onnx_path),
        "--no_viewer",
        "--axes", "vx,yaw",
        "--fractions", "0.25,0.5,0.75,1.0",
        "--phase_duration", "4.0",
        "--json_out", json_out,
        "--csv_out", csv_out,
    ]
    result = subprocess.run(cmd, cwd=str(SCRIPT_DIR),
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout.decode("utf-8", errors="replace"))
    if result.returncode != 0 or not Path(json_out).exists():
        return None
    with open(json_out, encoding="utf-8") as f:
        return json.load(f)


def _set_stack_unlimited():
    try:
        resource.setrlimit(resource.RLIMIT_STACK,
                           (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except (ValueError, resource.error):
        pass


def run_single(overrides: dict, output_dir: str, num_timesteps: int) -> bool:
    jax_cache = tempfile.mkdtemp(prefix="jax_cache_")
    env = {**os.environ, "JAX_COMPILATION_CACHE_DIR": jax_cache}
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
    parser.add_argument("--values", type=float, nargs="+", default=SAFE_ALIVE_VALUES)
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--skip-presim", action="store_true")
    parser.add_argument("--auto_eval", action="store_true",
                        help="学習後に check_command_response で速度追従 RMSE を自動評価")
    args = parser.parse_args()

    _set_stack_unlimited()

    if not args.skip_presim:
        _presim(args.values)

    print("=" * 60)
    print("Phase 3: alive scale sweep")
    print(f"  values   : {args.values}")
    print(f"  fixed    : {json.dumps(FIXED_PARAMS, indent=4)}")
    print(f"  timesteps: {args.timesteps:,}")
    print(f"  sleep    : {SLEEP_BETWEEN_RUNS}s between runs")
    print(f"  選択基準 : {'RMSE_vx 最小 (速度追従)' if args.auto_eval else '報酬最大 (要手動確認)'}")
    print("=" * 60)

    succeeded = []
    failed = []
    eval_results = {}
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
            print("  => SUCCESS")
            if args.auto_eval:
                onnx = find_latest_onnx(run_dir)
                if onnx:
                    json_out = f"eval_results/p3_alive_{slug}.json"
                    print(f"  [eval] {onnx.name} -> {json_out}")
                    res = run_eval(onnx, json_out)
                    if res:
                        eval_results[alive] = {
                            "rmse_vx":  res["rmse_vx"],
                            "rmse_yaw": res["rmse_yaw"],
                            "json":     json_out,
                        }
                        print(f"  [eval] RMSE_vx={res['rmse_vx']:.4f}  RMSE_yaw={res['rmse_yaw']:.4f}")
                    else:
                        print("  [eval] 評価失敗")
                else:
                    print("  [eval] ONNX が見つからない")
        else:
            failed.append(alive)
            print("  => FAILED  (ptxas/SIGSEGV なら reboot 後に再実行)")

        if i < total:
            print(f"  Sleeping {SLEEP_BETWEEN_RUNS}s...")
            time.sleep(SLEEP_BETWEEN_RUNS)

    print("\n" + "=" * 60)
    print(f"SUCCEEDED: {succeeded}")
    if failed:
        print(f"FAILED   : {failed}")
        for v in failed:
            slug = str(v).replace(".", "p")
            print(f"     python experiments/phase3_alive_sweep.py --values {v} --auto_eval")

    if eval_results:
        print("\n=== 速度追従 RMSE 比較 (alive sweep) ===")
        print(f"  {'alive':>6}  {'RMSE_vx':>10}  {'RMSE_yaw':>10}")
        best = min(eval_results, key=lambda v: eval_results[v]["rmse_vx"])
        for v, r in sorted(eval_results.items()):
            mark = " ← best (RMSE_vx 最小)" if v == best else ""
            print(f"  {v:>6}  {r['rmse_vx']:>10.4f}  {r['rmse_yaw']:>10.4f}{mark}")
        print(f"\n=> 採用推奨: alive={best}")
    else:
        print("  (--auto_eval なし: TensorBoard で報酬曲線を確認)")

    print(f"\nTensorBoard: tensorboard --logdir {SCRIPT_DIR}/checkpoints --port 6006")
    print("=" * 60)


if __name__ == "__main__":
    main()
