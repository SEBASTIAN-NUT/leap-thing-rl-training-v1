import h5py
import numpy as np

with h5py.File('cpg_trajectory.h5', 'r') as f:
    timestamps = f['timestamps'][:]
    raw_actions = f['raw_actions'][:]
    
    print(f"Timestamps shape: {timestamps.shape}")
    print(f"Raw actions shape: {raw_actions.shape}")
    print(f"\nFirst 5 samples:")
    print(f"  timestamps: {timestamps[:5]}")
    print(f"  action[0] (if_mcp): {raw_actions[:5, 0]}")
    print(f"  action[2] (if_pip): {raw_actions[:5, 2]}")
    print(f"\nLast 5 samples:")
    print(f"  timestamps: {timestamps[-5:]}")
    print(f"  action[0] (if_mcp): {raw_actions[-5:, 0]}")
    print(f"  action[2] (if_pip): {raw_actions[-5:, 2]}")
