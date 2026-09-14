import csv
import numpy as np
import h5py

with open('cpg_trajectory.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

timestamps = np.array([float(r['t']) for r in rows], dtype=np.float32)
if_mcp_rad = np.array([float(r['if_mcp']) for r in rows], dtype=np.float32)
if_pip_rad = np.array([float(r['if_pip']) for r in rows], dtype=np.float32)

# action値に変換
ACTION_SCALE_RAD = 0.5
action_mcp = np.clip(if_mcp_rad / ACTION_SCALE_RAD, -1.0, 1.0)
action_pip = np.clip(if_pip_rad / ACTION_SCALE_RAD, -1.0, 1.0)

# 16次元アクション - 3本指に同じ値を適用
raw_actions = np.zeros((len(rows), 16), dtype=np.float32)
# Index Finger
raw_actions[:, 0] = action_mcp   # if_mcp
raw_actions[:, 2] = action_pip   # if_pip
# Middle Finger
raw_actions[:, 4] = action_mcp   # mf_mcp
raw_actions[:, 6] = action_pip   # mf_pip
# Ring Finger
raw_actions[:, 8] = action_mcp   # rf_mcp
raw_actions[:, 10] = action_pip  # rf_pip

with h5py.File('cpg_trajectory.h5', 'w') as f:
    f.create_dataset('timestamps', data=timestamps)
    f.create_dataset('raw_actions', data=raw_actions)

print(f"✅ 3本指対応に変換完了")
