"""v5: palm-flat home position search.

手のひら全体を地面に接地させる home position を探索する。
v4 との主な違い:
  - ORIENT_QUAT = [0,1,0,0]: floating base を 180°X 回転させ palm が下向きに
  - 指を伸ばした状態（pip=dip=0）を固定し、palm_z のみを探索
  - 安定性評価: palm_collision geom が床に接触しているかで判定
"""
import sys
sys.path.insert(0, ".")
import mujoco
import numpy as np

from .mujoco_infer_base import MJInferBase
from . import constants

ACTUATOR_ORDER = [
    "if_mcp", "if_rot", "if_pip", "if_dip",
    "mf_mcp", "mf_rot", "mf_pip", "mf_dip",
    "rf_mcp", "rf_rot", "rf_pip", "rf_dip",
    "th_cmc", "th_axl", "th_mcp", "th_ipl",
]

# 手のひら下向き: floating base を 180°X 回転
ORIENT_QUAT = [0.0, 1.0, 0.0, 0.0]

# 指を伸ばした状態（pip=dip=0）。mcp/rot は自然な角度を維持
BASE_QPOS_PALM = {
    "if_mcp":  0.30,  "if_rot": -0.30,  "if_pip": 0.0,  "if_dip": 0.0,
    "mf_mcp":  0.30,  "mf_rot":  0.00,  "mf_pip": 0.0,  "mf_dip": 0.0,
    "rf_mcp":  0.20,  "rf_rot":  0.50,  "rf_pip": 0.0,  "rf_dip": 0.0,
    "th_cmc":  0.50,  "th_axl": -0.30,  "th_mcp": 0.0,  "th_ipl": 0.0,
}

PALM_GEOM = "palm_collision"


def palm_geom_z(base, palm_z, joints):
    qpos = [0.0165, 0.0052, palm_z] + ORIENT_QUAT + [joints[n] for n in ACTUATOR_ORDER]
    base.data.qpos[:] = qpos
    mujoco.mj_forward(base.model, base.data)
    gid = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_GEOM, PALM_GEOM)
    return float(base.data.geom_xpos[gid][2])


def score_pose_palm(base, palm_z, joints, hold_steps=1000):
    qpos = np.array([0.0165, 0.0052, palm_z] + ORIENT_QUAT
                    + [joints[n] for n in ACTUATOR_ORDER])
    base.data.qpos[:] = qpos
    base.data.qvel[:] = 0.0
    base.data.ctrl[:] = qpos[7:]
    mujoco.mj_forward(base.model, base.data)

    palm_gid = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_GEOM, PALM_GEOM)
    max_qacc = 0.0
    nan_hit = False
    palm_contact_steps = 0
    palm_z_start = base.data.qpos[2]
    survived = 0

    for i in range(hold_steps):
        mujoco.mj_step(base.model, base.data)
        if not np.all(np.isfinite(base.data.qacc)):
            nan_hit = True
            break
        max_qacc = max(max_qacc, float(np.max(np.abs(base.data.qacc))))
        # palm が地面に接触しているかチェック
        for c in range(base.data.ncon):
            con = base.data.contact[c]
            if con.geom1 == palm_gid or con.geom2 == palm_gid:
                palm_contact_steps += 1
                break
        survived = i + 1

    palm_drift = abs(base.data.qpos[2] - palm_z_start)
    contact_frac = palm_contact_steps / hold_steps
    return dict(
        max_qacc=max_qacc, palm_drift=palm_drift, nan_hit=nan_hit,
        contact_frac=contact_frac, survived_steps=survived,
    )


def main():
    base = MJInferBase(str(constants.task_to_xml("flat_terrain")))
    joints = dict(BASE_QPOS_PALM)

    print("=== palm-flat home position search ===")
    print(f"ORIENT_QUAT = {ORIENT_QUAT}  (180deg around X → palm faces down)")
    print(f"finger joints: pip=dip=0 (extended)\n")

    best = None
    best_score = float("inf")

    for palm_z in np.linspace(0.03, 0.10, 8):
        pgz = palm_geom_z(base, palm_z, joints)
        info = score_pose_palm(base, palm_z, joints)
        score = info["palm_drift"] + info["max_qacc"] * 1e-4 - info["contact_frac"] * 0.1
        marker = ""
        if info["contact_frac"] > 0.5 and not info["nan_hit"] and score < best_score:
            best_score = score
            best = (palm_z, dict(joints))
            marker = " ← best"
        print(f"palm_z={palm_z:.3f}  palm_geom_z={pgz:.4f}  {info}{marker}")

    print()
    if best is None:
        print("[WARN] 安定した姿勢が見つかりません。BASE_QPOS_PALM を調整してください。")
        return

    palm_z, best_joints = best
    qpos_vals = [0.0165, 0.0052, palm_z] + ORIENT_QUAT + [best_joints[n] for n in ACTUATOR_ORDER]
    qpos_str = " ".join(f"{v:.4f}" for v in qpos_vals)
    print(f"=== 推奨 home position (palm-flat) ===")
    print(f'<key name="home" qpos="{qpos_str}"/>')
    print()
    print(f"palm_z={palm_z:.4f}  joints={[f'{best_joints[n]:.4f}' for n in ACTUATOR_ORDER]}")


if __name__ == "__main__":
    main()
