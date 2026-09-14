import csv
import numpy as np
import h5py

# CSV読み込み
with open('cpg_trajectory.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

timestamps = np.array([float(r['t']) for r in rows], dtype=np.float32)
if_mcp = np.array([float(r['if_mcp']) for r in rows], dtype=np.float32)
if_pip = np.array([float(r['if_pip']) for r in rows], dtype=np.float32)

# 16次元アクションに変換（if_mcp→action[0], if_pip→action[2], 他は0）
raw_actions = np.zeros((len(rows), 16), dtype=np.float32)
raw_actions[:, 0] = if_mcp   # MCP
raw_actions[:, 2] = if_pip   # PIP

# HDF5に保存
with h5py.File('cpg_trajectory.h5', 'w') as f:
    f.create_dataset('timestamps', data=timestamps)
    f.create_dataset('raw_actions', data=raw_actions)

print(f"✅ 変換完了: cpg_trajectory.h5 ({len(rows)} ステップ)")
