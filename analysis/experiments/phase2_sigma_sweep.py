import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent.parent))
#!/usr/bin/env python3
"""
Phase 2: Tracking sensitivity (sigma) sweep

Phase 2a: sigma_lin を変化、sigma_ang を固定
Phase 2b: sigma_ang を変化、sigma_lin を固定 (Phase 1 採用値 0.025 で固定)

実行方法 (プロジェクトルートから):
    Open_Duck_Playground/.venv/bin/python experiments/phase2_sigma_sweep.py
    Open_Duck_Playground/.venv/bin/python experiments/phase2_sigma_sweep.py --phase 2a
    Open_Duck_Playground/.venv/bin/python experiments/phase2_sigma_sweep.py --phase 2b
    Open_Duck_Playground/.venv/bin/python experiments/phase2_sigma_sweep.py --timesteps 50000000
    Open_Duck_Playground/.venv/bin/python experiments/phase2_sigma_sweep.py --auto_eval

選択基準: 速度追従 RMSE_vx の最小値 (報酬最大値ではない)
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

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
PYTHON = SCRIPT_DIR / "Open_Duck_Playground" / ".venv" / "bin" / "python"

DEFAULT_TIMESTEPS = 150_000_000
SLEEP_BETWEEN_RUNS = 300

PHASE2_COMMON_FIXED = {
    "reward_config.scales.tracking_lin_vel": 1.5,
    "reward_config.scales.tracking_ang_vel": 0.5,
    "reward_config.scales.alive":            1.0,
}

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

# sigma_lin は Phase 1 採用値 0.025 で固定 (旧スクリプトの 0.01 は誤り)
PHASE2B = {
    "description": "sigma_ang sweep (sigma_lin=0.025 fixed, Phase 1 result)",
    "sweep_key":   "reward_config.tracking_sigma_ang",
    "candidates":  [0.1, 0.25, 0.5, 1.0, 2.0],
    "fixed": {
        **PHASE2_COMMON_FIXED,
        "reward_config.tracking_sigma_lin": 0.025,
    },
    "prefix": "p2b_sigma_ang",
}


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


def run_phase(phase: dict, num_timesteps: int, auto_eval: bool = False):
    _set_stack_unlimited()
    print("=" * 60)
    print(f"Phase: {phase['description']}")
    print(f"  sweep : {phase['sweep_key']}")
    print(f"  values: {phase['candidates']}")
    print(f"  fixed : {json.dumps(phase['fixed'], indent=4)}")
    print(f"  timesteps: {num_timesteps:,}")
    print(f"  選択基準: {'RMSE_vx 最小 (速度追従)' if auto_eval else '報酬最大 (要手動確認)'}")
    print("=" * 60)

    succeeded = []
    failed = []
    eval_results = {}
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
            if auto_eval:
                onnx = find_latest_onnx(run_dir)
                if onnx:
                    json_out = f"eval_results/{phase['prefix']}_{slug}.json"
                    print(f"  [eval] {onnx.name} -> {json_out}")
                    res = run_eval(onnx, json_out)
                    if res:
                        eval_results[value] = {
                            "rmse_vx": res["rmse_vx"],
                            "rmse_yaw": res["rmse_yaw"],
                            "json": json_out,
                        }
                        print(f"  [eval] RMSE_vx={res['rmse_vx']:.4f}  RMSE_yaw={res['rmse_yaw']:.4f}")
                    else:
                        print("  [eval] 評価失敗")
                else:
                    print("  [eval] ONNX が見つからない")
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

    if eval_results:
        print(f"\n=== 速度追従 RMSE 比較 ({phase['sweep_key']}) ===")
        print(f"  {'value':>8}  {'RMSE_vx':>10}  {'RMSE_yaw':>10}")
        best = min(eval_results, key=lambda v: eval_results[v]["rmse_vx"])
        for v, r in sorted(eval_results.items()):
            mark = " ← best (RMSE_vx 最小)" if v == best else ""
            print(f"  {v:>8}  {r['rmse_vx']:>10.4f}  {r['rmse_yaw']:>10.4f}{mark}")
        print(f"\n=> 採用推奨: {phase['sweep_key']}={best}")
    else:
        print("  (--auto_eval なし: TensorBoard で報酬曲線を確認)")

    print(f"TensorBoard: tensorboard --logdir {SCRIPT_DIR}/checkpoints --port 6006")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["2a", "2b", "all"], default="all")
    parser.add_argument("--timesteps", type=int, default=DEFAULT_TIMESTEPS)
    parser.add_argument("--values", type=float, nargs="+", default=None)
    parser.add_argument("--auto_eval", action="store_true",
                        help="学習後に check_command_response で速度追従 RMSE を自動評価")
    args = parser.parse_args()

    if args.phase in ("2a", "all"):
        phase = dict(PHASE2A)
        if args.values is not None:
            phase["candidates"] = args.values
        run_phase(phase, args.timesteps, auto_eval=args.auto_eval)
    if args.phase in ("2b", "all"):
        phase = dict(PHASE2B)
        if args.values is not None:
            phase["candidates"] = args.values
        run_phase(phase, args.timesteps, auto_eval=args.auto_eval)


if __name__ == "__main__":
    main()
