import csv
import numpy as np
import h5py

# CSV読み込み
with open('cpg_trajectory.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

timestamps = np.array([float(r['t']) for r in rows], dtype=np.float32)
if_mcp_rad = np.array([float(r['if_mcp']) for r in rows], dtype=np.float32)
if_pip_rad = np.array([float(r['if_pip']) for r in rows], dtype=np.float32)

# 関節角をaction値に正規化（各関節の可動域で正規化）
# LEAP Hand MCP: -0.4 ～ 0.8 rad, PIP: -0.5 ～ 1.0 rad
mcp_min, mcp_max = -0.4, 0.8
pip_min, pip_max = -0.5, 1.0

action_mcp = 2 * (if_mcp_rad - mcp_min) / (mcp_max - mcp_min) - 1
action_pip = 2 * (if_pip_rad - pip_min) / (pip_max - pip_min) - 1

# クリップして -1.0 ～ 1.0 に
action_mcp = np.clip(action_mcp, -1.0, 1.0)
action_pip = np.clip(action_pip, -1.0, 1.0)

# 16次元アクションに変換
raw_actions = np.zeros((len(rows), 16), dtype=np.float32)
raw_actions[:, 0] = action_mcp
raw_actions[:, 2] = action_pip

# HDF5に保存
with h5py.File('cpg_trajectory.h5', 'w') as f:
    f.create_dataset('timestamps', data=timestamps)
    f.create_dataset('raw_actions', data=raw_actions)

print(f"✅ 変換完了: action値に正規化した cpg_trajectory.h5")
print(f"  MCP action range: {action_mcp.min():.3f} ～ {action_mcp.max():.3f}")
print(f"  PIP action range: {action_pip.min():.3f} ～ {action_pip.max():.3f}")
