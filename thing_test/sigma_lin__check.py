"""Offline numerical check for tracking reward sensitivity.

Mirrors thing_walk.py _get_reward() exactly:
  lin_vel_error = sum((cmd[:2] - local_vel[:2])^2)
  tracking_lin  = exp(-lin_vel_error / sigma_lin)

  ang_vel_error = (cmd[2] - gyro_z)^2
  tracking_ang  = exp(-ang_vel_error / sigma_ang)

Run with: python reward_check.py
"""
import math


def reward_lin(cmd_vx, actual_vx, cmd_vy, actual_vy, sigma_lin):
    error = (cmd_vx - actual_vx) ** 2 + (cmd_vy - actual_vy) ** 2
    return math.exp(-error / sigma_lin)


def reward_ang(cmd_yaw, actual_yaw, sigma_ang):
    error = (cmd_yaw - actual_yaw) ** 2
    return math.exp(-error / sigma_ang)


# --- sigma_lin sweep (sigma_ang fixed at 0.25) ---
SIGMA_LIN_CANDIDATES = [0.0025, 0.005, 0.01, 0.025, 0.05]
SIGMA_ANG_FIXED = 0.25

# --- sigma_ang sweep (sigma_lin fixed at 0.01) ---
SIGMA_ANG_CANDIDATES = [0.1, 0.25, 0.5, 1.0, 2.0]
SIGMA_LIN_FIXED = 0.01

fracs = [0.0, 0.25, 0.5, 0.75, 1.0]

def print_table(title, sigmas, is_lin):
    print(f"\n=== {title} ===")
    header = f"{'sigma':>8} | " + " ".join(f"act={int(f*100):3d}%" for f in fracs) + "  gradient"
    print(header)
    print("-" * len(header))
    for sigma in sigmas:
        row = f"{sigma:>8.4f} | "
        if is_lin:
            rewards = [reward_lin(0.15, 0.15 * f, 0.0, 0.0, sigma) for f in fracs]
        else:
            rewards = [reward_ang(1.0, 1.0 * f, sigma) for f in fracs]
        for r in rewards:
            row += f"{r:7.3f} "
        grad = rewards[-1] - rewards[0]
        row += f"  {grad:.3f}"

        # 判定
        r0 = rewards[0]
        if r0 < 0.005:
            verdict = "  [疎すぎ]"
        elif r0 > 0.5:
            verdict = "  [緩すぎ]"
        elif grad > 0.5:
            verdict = "  [良好]"
        else:
            verdict = "  [やや緩]"
        print(row + verdict)


print_table("sigma_lin sweep  (cmd_vx=0.15, sigma_ang=固定0.25)", SIGMA_LIN_CANDIDATES, is_lin=True)
print_table("sigma_ang sweep  (cmd_yaw=1.0,  sigma_lin=固定0.01)", SIGMA_ANG_CANDIDATES, is_lin=False)

print("""
判定基準:
  良好   : 0%追従で reward > 0.05 かつ gradient > 0.5
  疎すぎ : 0%追従で reward < 0.005 → 学習初期に勾配なし
  緩すぎ : 0%追従で reward > 0.5  → 良悪の区別がつかない
""")
