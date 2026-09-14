import csv
import numpy as np
import h5py

with open('cpg_trajectory.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

timestamps = np.array([float(r['t']) for r in rows], dtype=np.float32)
if_mcp_rad = np.array([float(r['if_mcp']) for r in rows], dtype=np.float32)
if_pip_rad = np.array([float(r['if_pip']) for r in rows], dtype=np.float32)

ACTION_SCALE_RAD = 0.5
action_mcp = np.clip(if_mcp_rad / ACTION_SCALE_RAD, -1.0, 1.0)
action_pip = np.clip(if_pip_rad / ACTION_SCALE_RAD, -1.0, 1.0)

raw_actions = np.zeros((len(rows), 16), dtype=np.float32)
# MuJoCo joint →実機 motor マッピング
# Index: joint[0]=if_mcp→motor_1, joint[2]=if_pip→motor_2
raw_actions[:, 1] = action_mcp   # motor_1 (if_mcp)
raw_actions[:, 2] = action_pip   # motor_2 (if_pip)
# Middle: joint[4]=mf_mcp→motor_5, joint[6]=mf_pip→motor_6
raw_actions[:, 5] = action_mcp   # motor_5 (mf_mcp)
raw_actions[:, 6] = action_pip   # motor_6 (mf_pip)
# Ring: joint[8]=rf_mcp→motor_9, joint[10]=rf_pip→motor_10
raw_actions[:, 9] = action_mcp   # motor_9 (rf_mcp)
raw_actions[:, 10] = action_pip  # motor_10 (rf_pip)

with h5py.File('cpg_trajectory.h5', 'w') as f:
    f.create_dataset('timestamps', data=timestamps)
    f.create_dataset('raw_actions', data=raw_actions)

print(f"✅ 正しいマッピングで変換完了")
