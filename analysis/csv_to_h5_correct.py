import csv
import numpy as np
import h5py

with open('cpg_project/cpg_trajectory.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

timestamps = np.array([float(r['t']) for r in rows], dtype=np.float32)
if_mcp_rad = np.array([float(r['if_mcp']) for r in rows], dtype=np.float32)
if_pip_rad = np.array([float(r['if_pip']) for r in rows], dtype=np.float32)

# 実際の可動域に基づいた正規化
MCP_MIN, MCP_MAX = -0.0963, 0.2672
PIP_MIN, PIP_MAX = -0.0195, 0.5747

action_mcp = 2.0 * (if_mcp_rad - MCP_MIN) / (MCP_MAX - MCP_MIN) - 1.0
action_pip = 2.0 * (if_pip_rad - PIP_MIN) / (PIP_MAX - PIP_MIN) - 1.0

# ★ オフセット調整: 平均を 0 に
action_pip_offset = action_pip - action_pip.mean()

action_mcp = np.clip(action_mcp, -1.0, 1.0)
action_pip_offset = np.clip(action_pip_offset, -1.0, 1.0)

# 16次元アクション
raw_actions = np.zeros((len(rows), 16), dtype=np.float32)
raw_actions[:, 1] = action_mcp
raw_actions[:, 2] = action_pip_offset

# HDF5保存
with h5py.File('cpg_trajectory.h5', 'w') as f:
    f.create_dataset('timestamps', data=timestamps)
    f.create_dataset('raw_actions', data=raw_actions)

print(f"✅ 修正完了 (オフセット調整)")
print(f"  MCP action: [{action_mcp.min():.3f}, {action_mcp.max():.3f}]")
print(f"  PIP action: [{action_pip_offset.min():.3f}, {action_pip_offset.max():.3f}] (平均: {action_pip_offset.mean():.3f})")
