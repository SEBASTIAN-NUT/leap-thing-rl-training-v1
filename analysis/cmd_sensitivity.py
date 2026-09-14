# -*- coding: utf-8 -*-
"""
cmd_sensitivity.py
コマンド感度解析: obs[0] (cmd_vx) だけ変化させたとき policy の出力がどう変わるか
出力変化なし → policy がコマンドを無視している
"""
import json, math
import numpy as np
from pathlib import Path

try:
    import onnxruntime as ort
except ImportError:
    print("[ERROR] pip install onnxruntime")
    raise

PROJECT_DIR = Path(__file__).resolve().parent.parent
OBS_DIM = 112
CMD_VX_IDX = 0   # obs[0] = cmd_vx

# 使う run (歩けている run を複数試す)
RUNS = [
    "eval_results/archived/p2b_sigang_0p25.json",
    "eval_results/archived/p3_alive_0p5.json",
    "eval_results/archived/p3_alive_1p0.json",
]

CMD_SWEEP = [0.0, 0.0375, 0.0750, 0.1125, 0.1500]

def neutral_obs():
    """デフォルト姿勢・静止状態の観測ベクトル（cmd 以外は全ゼロ）"""
    return np.zeros(OBS_DIM, dtype=np.float32)

def run_sensitivity(onnx_path: Path, label: str):
    sess = ort.InferenceSession(str(onnx_path))
    inp_name = sess.get_inputs()[0].name

    actions = []
    for cmd_vx in CMD_SWEEP:
        obs = neutral_obs()
        obs[CMD_VX_IDX] = cmd_vx
        out = sess.run(None, {inp_name: obs[None]})[0][0]
        actions.append(out)

    actions = np.array(actions)            # (n_cmd, action_dim)
    action_std = actions.std(axis=0)       # per-joint std across cmd values
    mean_std  = float(action_std.mean())   # スカラー指標

    print(f"\n{'='*60}")
    print(f"RUN: {label}")
    print(f"  cmd_vx 変化に対する action の std (per joint):")
    print(f"    mean={mean_std:.5f}  max={action_std.max():.5f}  min={action_std.min():.5f}")
    print(f"  → {'コマンドを無視している (std≈0)' if mean_std < 0.01 else 'コマンドに反応あり (std>0)'}")

    # 各 cmd での action 差分（cmd=0 からの変化）
    print(f"\n  cmd_vx vs action[0] (拇指の代表関節):")
    for i, cmd in enumerate(CMD_SWEEP):
        print(f"    cmd={cmd:.4f}  act[0]={actions[i,0]:+.5f}  "
              f"Δ={actions[i,0]-actions[0,0]:+.5f}")
    return mean_std

print("コマンド感度解析")
print("obs[0]=cmd_vx のみ変化、他は全ゼロ（デフォルト姿勢）\n")

results = []
for run_json in RUNS:
    p = PROJECT_DIR / run_json
    if not p.exists():
        print(f"[SKIP] {run_json}")
        continue
    data = json.loads(p.read_text())
    onnx_rel = data.get("onnx", "")
    onnx_path = PROJECT_DIR / onnx_rel
    if not onnx_path.exists():
        print(f"[SKIP ONNX] {onnx_rel}")
        continue
    std = run_sensitivity(onnx_path, Path(run_json).stem)
    results.append((Path(run_json).stem, std))

print("\n" + "="*60)
print("まとめ")
for name, std in results:
    verdict = "コマンド無視" if std < 0.01 else "コマンド反応あり"
    print(f"  {name:<30} mean_std={std:.5f}  → {verdict}")
