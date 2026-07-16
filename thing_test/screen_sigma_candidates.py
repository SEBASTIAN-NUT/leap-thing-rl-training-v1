"""Screen candidate tracking_sigma_lin / tracking_sigma_ang values offline
(no training) before committing to a reward-parameter sweep.

For each sigma candidate, prints tracking reward at 0% vs 50% actual tracking,
across commands at 25 / 50 / 75 / 100% of max range.

  0%追従 : reward when robot stands still (worst case — sigma too small if ≈0)
  50%追従: reward when robot tracks at half the commanded speed
  差     : 50%追従 - 0%追従 (larger = stronger PPO learning gradient)

If 0%追従 ≈ 0 for all commands  → sigma too small, gradient vanishes, no learning.
If 差    ≈ 0 for all commands   → sigma too large, reward is flat, no learning.

Usage:
    cd ~/Desktop/iizawa/iizawa_workspace/thing_project
    Open_Duck_Playground/.venv/bin/python -m thing_test.screen_sigma_candidates

See also: thing_test/analyze_reward_gradient.py for continuous ∂R/∂err analysis.
"""
import sys
sys.path.insert(0, ".")
import jax.numpy as jp
from playground.common.rewards import reward_tracking_lin_vel, reward_tracking_ang_vel

LIN_CMD_MAX = 0.15
ANG_CMD_MAX = 1.0

SIGMA_LIN_CANDIDATES = [0.005, 0.01, 0.015, 0.025, 0.05, 0.1]
SIGMA_ANG_CANDIDATES = [0.05, 0.1, 0.15, 0.25, 0.4, 0.6]


def screen_lin(sigma):
    print(f"  sigma_lin={sigma}")
    for frac in [0.25, 0.5, 0.75, 1.0]:
        cmd = jp.array([LIN_CMD_MAX * frac, 0.0, 0.0])
        r_zero = float(reward_tracking_lin_vel(cmd, jp.array([0.0, 0.0, 0.0]), sigma))
        r_half = float(reward_tracking_lin_vel(cmd, jp.array([LIN_CMD_MAX * frac * 0.5, 0.0, 0.0]), sigma))
        print(f"    cmd={frac*100:5.0f}%: 0%追従={r_zero:.4f}  50%追従={r_half:.4f}  差={r_half-r_zero:+.4f}")


def screen_ang(sigma):
    print(f"  sigma_ang={sigma}")
    for frac in [0.25, 0.5, 0.75, 1.0]:
        cmd = jp.array([0.0, 0.0, ANG_CMD_MAX * frac])
        r_zero = float(reward_tracking_ang_vel(cmd, jp.array([0.0, 0.0, 0.0]), sigma))
        r_half = float(reward_tracking_ang_vel(cmd, jp.array([0.0, 0.0, ANG_CMD_MAX * frac * 0.5]), sigma))
        print(f"    cmd={frac*100:5.0f}%: 0%追従={r_zero:.4f}  50%追従={r_half:.4f}  差={r_half-r_zero:+.4f}")


print("=== tracking_lin_vel (range +-0.15) ===")
for s in SIGMA_LIN_CANDIDATES:
    screen_lin(s)
    print()

print("=== tracking_ang_vel (range +-1.0) ===")
for s in SIGMA_ANG_CANDIDATES:
    screen_ang(s)
    print()
