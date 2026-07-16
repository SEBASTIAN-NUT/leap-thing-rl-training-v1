"""
報酬関数の勾配を誤差0→100%にわたって連続的に分析する。
"差"（0%vs50%）ではなく、各誤差点での∂R/∂errを求め、
どのsigma値が学習に有効なグラジェントを持つかを判断する。

「学習していない」の定義:
  |∂R/∂err| が err=cmd_max（最悪ケース: ロボットが静止）で
  実質ゼロに近い場合、PPOは改善の方向を推定できない。

報酬式: R = exp(-err² / sigma)   ← sigma²ではなくsigma
勾配式: ∂R/∂err = -2*err/sigma * exp(-err²/sigma)

Usage:
    cd ~/Desktop/iizawa/iizawa_workspace/thing_project
    Open_Duck_Playground/.venv/bin/python -m thing_test.analyze_reward_gradient

Key outputs per sigma:
  - 各誤差点 (0~100% of cmd_max) での報酬R と 勾配∂R/∂err
  - ★ 勾配最大点: |∂R/∂err| が最大になる誤差値 → 学習信号が最も強い点
  - ★ 初期学習信号: err=cmd_max での勾配 → ロボット静止 = 最悪ケースでの勾配

See also: thing_test/screen_sigma_candidates.py for discrete 0%% vs 50%% comparison.
"""
import sys
sys.path.insert(0, ".")
import numpy as np

CMD_VX_MAX = 0.15   # m/s
CMD_YAW_MAX = 1.0   # rad/s

SIGMA_LIN_CANDIDATES = [0.005, 0.01, 0.015, 0.025, 0.05, 0.1]
SIGMA_ANG_CANDIDATES = [0.1, 0.15, 0.25, 0.4]

def reward(err_sq, sigma):
    return np.exp(-err_sq / sigma)

def gradient(err, sigma):
    """∂R/∂err = -2*err/sigma * exp(-err²/sigma)"""
    return -2.0 * err / sigma * np.exp(-err**2 / sigma)

def analyze(sigma, cmd_max, label):
    print(f"  sigma={sigma}")
    print(f"    {'誤差':>6}  {'報酬R':>8}  {'勾配∂R/∂err':>12}  {'勾配×err (相対)':>14}")
    for frac in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]:
        err = cmd_max * frac
        R   = reward(err**2, sigma)
        G   = gradient(err, sigma)
        rel = G * err  # 相対的なインパクト
        print(f"    err={frac*100:5.1f}%  R={R:8.4f}  ∂R/∂err={G:12.4f}  G×err={rel:14.4f}")
    # 勾配最大点
    err_peak = np.sqrt(sigma / 2.0)
    G_peak = gradient(err_peak, sigma)
    print(f"    ★ 勾配最大点: err={err_peak:.4f} m/s ({err_peak/cmd_max*100:.1f}% of max), |∂R/∂err|={abs(G_peak):.4f}")
    # 初期状態（ロボット静止=最大誤差）での勾配
    G_init = gradient(cmd_max, sigma)
    print(f"    ★ 初期学習信号 (err=cmd_max): ∂R/∂err={G_init:.4f}")
    print()

print("=" * 60)
print("=== tracking_lin_vel (cmd_max=0.15 m/s) ===")
print("=" * 60)
for s in SIGMA_LIN_CANDIDATES:
    analyze(s, CMD_VX_MAX, "lin")

print("=" * 60)
print("=== tracking_ang_vel (cmd_max=1.0 rad/s) ===")
print("=" * 60)
for s in SIGMA_ANG_CANDIDATES:
    analyze(s, CMD_YAW_MAX, "ang")
