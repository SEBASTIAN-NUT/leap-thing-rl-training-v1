"""v4: respect each joint's actual range when searching curl scale (mf_dip
and th_mcp sit exactly at their joint limit already, so each finger's two
curl joints are now searched independently and clipped to their real
ranges, instead of one shared multiplier that can push a pinned joint out
of range).
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
FINGER_CURL_JOINTS = {
    "if": ["if_pip", "if_dip"],
    "mf": ["mf_pip", "mf_dip"],
    "rf": ["rf_pip", "rf_dip"],
    "th": ["th_mcp", "th_ipl"],
}
FINGER_TIP_GEOM = {
    "if": "if_tip_collision",
    "mf": "mf_tip_collision",
    "rf": "rf_tip_collision",
    "th": "th_tip_collision",
}
BASE_QPOS = {
    "if_mcp": 0.3220, "if_rot": -0.3350, "if_pip": 1.1318, "if_dip": 0.3684,
    "mf_mcp": 0.3220, "mf_rot": -0.0000, "mf_pip": 1.2992, "mf_dip": -0.3660,
    "rf_mcp": 0.2075, "rf_rot": 0.5235, "rf_pip": 1.0481, "rf_dip": 0.0434,
    "th_cmc": 1.8131, "th_axl": -0.3490, "th_mcp": -0.4700, "th_ipl": 0.5115,
}
ORIENT_QUAT = [0.9999, 0.0052, -0.0147, -0.0004]


def get_joint_ranges(base):
    ranges = {}
    for name in ACTUATOR_ORDER:
        jid = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ranges[name] = tuple(base.model.jnt_range[jid])
    return ranges


def valid_scale_range(joint_name, base_val, jrange):
    lo, hi = jrange
    if base_val == 0:
        return (0.0, 0.0)
    if base_val > 0:
        return (lo / base_val, hi / base_val)
    else:
        return (hi / base_val, lo / base_val)


def fingertip_z(base, palm_z, joints, finger):
    qpos = [0.0165, 0.0052, palm_z] + ORIENT_QUAT + [joints[n] for n in ACTUATOR_ORDER]
    base.data.qpos[:] = qpos
    mujoco.mj_forward(base.model, base.data)
    gid = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_GEOM, FINGER_TIP_GEOM[finger])
    return base.data.geom_xpos[gid][2]


def find_best_joint_pair(base, palm_z, finger, target_z, jranges, n=30):
    j1, j2 = FINGER_CURL_JOINTS[finger]
    s1_lo, s1_hi = valid_scale_range(j1, BASE_QPOS[j1], jranges[j1])
    s2_lo, s2_hi = valid_scale_range(j2, BASE_QPOS[j2], jranges[j2])
    s1_lo, s1_hi = max(s1_lo, 0.0), min(s1_hi, 1.3)
    s2_lo, s2_hi = max(s2_lo, 0.0), min(s2_hi, 1.3)

    best, best_err = None, float("inf")
    for s1 in np.linspace(s1_lo, s1_hi, n):
        for s2 in np.linspace(s2_lo, s2_hi, n):
            joints = dict(BASE_QPOS)
            joints[j1] = BASE_QPOS[j1] * s1
            joints[j2] = BASE_QPOS[j2] * s2
            z = fingertip_z(base, palm_z, joints, finger)
            err = abs(z - target_z)
            if err < best_err:
                best_err, best = err, (s1, s2)
    s1, s2 = best
    joints = dict(BASE_QPOS)
    joints[j1] = BASE_QPOS[j1] * s1
    joints[j2] = BASE_QPOS[j2] * s2
    achieved_z = fingertip_z(base, palm_z, joints, finger)
    return (s1, s2), achieved_z, (joints[j1], joints[j2])


def score_pose(base, palm_z, joints, hold_steps=1000):
    qpos = np.array([0.0165, 0.0052, palm_z] + ORIENT_QUAT + [joints[n] for n in ACTUATOR_ORDER])
    base.data.qpos[:] = qpos
    base.data.qvel[:] = 0.0
    base.data.ctrl[:] = qpos[7:]
    mujoco.mj_forward(base.model, base.data)

    grav_id = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_SENSOR, "upvector")
    sstart = base.model.sensor_adr[grav_id]

    max_qacc = 0.0
    flipped = False
    nan_hit = False
    palm_z_start = base.data.qpos[2]
    contact_ok_steps = 0
    survived_steps = 0

    for i in range(hold_steps):
        mujoco.mj_step(base.model, base.data)
        if not np.all(np.isfinite(base.data.qacc)):
            nan_hit = True
            break
        max_qacc = max(max_qacc, float(np.max(np.abs(base.data.qacc))))
        upvec_z = base.data.sensordata[sstart + 2]
        if upvec_z < 0.0:
            flipped = True
            break
        if base.data.ncon >= 3:
            contact_ok_steps += 1
        survived_steps = i + 1

    palm_drift = abs(base.data.qpos[2] - palm_z_start)
    contact_frac = contact_ok_steps / hold_steps
    return dict(
        max_qacc=max_qacc, palm_drift=palm_drift, flipped=flipped,
        nan_hit=nan_hit, contact_frac=contact_frac, survived_steps=survived_steps,
    )


def main():
    base = MJInferBase(str(constants.task_to_xml("flat_terrain")))
    jranges = get_joint_ranges(base)
    target_z = 0.005

    for palm_z in [0.08, 0.09, 0.10, 0.11, 0.12, 0.13, 0.14]:
        joints = dict(BASE_QPOS)
        joint_pair_vals = {}
        for finger in ["if", "mf", "rf", "th"]:
            (s1, s2), achieved_z, (v1, v2) = find_best_joint_pair(base, palm_z, finger, target_z, jranges)
            j1, j2 = FINGER_CURL_JOINTS[finger]
            joints[j1], joints[j2] = v1, v2
            joint_pair_vals[finger] = (v1, v2)
            print(f"  palm_z={palm_z:.2f} {finger}: {j1}={v1:+.4f} {j2}={v2:+.4f} -> tip_z={achieved_z:.4f}")

        info = score_pose(base, palm_z, joints, hold_steps=1000)
        print(f"palm_z={palm_z:.2f} -> {info}")
        print()


if __name__ == "__main__":
    main()
