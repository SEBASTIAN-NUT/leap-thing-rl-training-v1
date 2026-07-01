"""Compute actual reward-term magnitudes (using the real reward functions,
not hand-derived formulas) for a few representative scenarios.
"""
import sys
sys.path.insert(0, ".")
import jax.numpy as jp
from playground.common.rewards import (
    reward_tracking_lin_vel,
    reward_tracking_ang_vel,
    cost_torques,
    cost_action_rate,
    cost_stand_still,
)

SCALES = {
    "tracking_lin_vel": 2.5,
    "tracking_ang_vel": 6.0,
    "torques": -1.0e-3,
    "action_rate": -0.5,
    "termination": -1.0,
    "stand_still": -0.5,
}
SIGMA = 0.01

scenarios = [
    ("完全停止(cmd=0) かつ 静止",
     [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ("前進指令50%(vx=0.075) だが静止",
     [0.075, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ("前進指令50%(vx=0.075) かつ完璧追従",
     [0.075, 0.0, 0.0], [0.075, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ("前進指令100%(vx=0.15) だが静止",
     [0.15, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ("旋回指令50%(yaw=0.5) だが静止",
     [0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ("旋回指令50%(yaw=0.5) かつ完璧追従",
     [0.0, 0.0, 0.5], [0.0, 0.0, 0.0], [0.0, 0.0, 0.5]),
    ("旋回指令100%(yaw=1.0) だが静止",
     [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
    ("旋回指令わずか25%追従(yaw=1.0実速度0.25)",
     [0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.25]),
]

print(f"{'シナリオ':45s} {'lin_reward(*scale)':>20s} {'ang_reward(*scale)':>20s}")
for name, cmd, linvel, gyro in scenarios:
    cmd_j = jp.array(cmd)
    linvel_j = jp.array(linvel)
    gyro_j = jp.array(gyro)
    r_lin = float(reward_tracking_lin_vel(cmd_j, linvel_j, SIGMA))
    r_ang = float(reward_tracking_ang_vel(cmd_j, gyro_j, SIGMA))
    print(f"{name:45s} {r_lin:8.4f} (*{r_lin*SCALES['tracking_lin_vel']:6.3f}) "
          f"{r_ang:8.4f} (*{r_ang*SCALES['tracking_ang_vel']:6.3f})")

print()
print("=== 行動コストの参考値 ===")
act = jp.ones(16)
last_act = jp.zeros(16)
c_action_rate = float(cost_action_rate(act, last_act))
print(f"cost_action_rate(全関節 0->1 変化): {c_action_rate:.4f}  (*scale -0.5 = {c_action_rate*SCALES['action_rate']:.4f})")

act_small = jp.ones(16) * 0.1
c_action_rate_small = float(cost_action_rate(act_small, last_act))
print(f"cost_action_rate(全関節 0->0.1 変化): {c_action_rate_small:.4f}  (*scale -0.5 = {c_action_rate_small*SCALES['action_rate']:.4f})")
