"""v3: fix the thumb's non-monotonic curl-scale-vs-tip-height relationship,
fine palm_z scan around 0.10, full-episode-length (1000 step) validation.
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


def fingertip_z(base, palm_z, joints, finger):
    qpos = [0.0165, 0.0052, palm_z] + ORIENT_QUAT + [joints[n] for n in ACTUATOR_ORDER]
    base.data.qpos[:] = qpos
    mujoco.mj_forward(base.model, base.data)
    gid = mujoco.mj_name2id(base.model, mujoco.mjtObj.mjOBJ_GEOM, FINGER_TIP_GEOM[finger])
    return base.data.geom_xpos[gid][2]


def find_best_scale_for_target_z(base, palm_z, finger, target_z, scan_lo=-2.0, scan_hi=3.0, n=200):
    scales = np.linspace(scan_lo, scan_hi, n)
    best_scale, best_err = None, float("inf")
    for s in scales:
        joints = dict(BASE_QPOS)
        for j in FINGER_CURL_JOINTS[finger]:
            joints[j] = BASE_QPOS[j] * s
        z = fingertip_z(base, palm_z, joints, finger)
        err = abs(z - target_z)
        if err < best_err:
            best_err, best_scale = err, s
    lo2, hi2 = best_scale - (scan_hi - scan_lo) / n, best_scale + (scan_hi - scan_lo) / n
    for s in np.linspace(lo2, hi2, 50):
        joints = dict(BASE_QPOS)
        for j in FINGER_CURL_JOINTS[finger]:
            joints[j] = BASE_QPOS[j] * s
        z = fingertip_z(base, palm_z, joints, finger)
        err = abs(z - target_z)
        if err < best_err:
            best_err, best_scale = err, s
    joints = dict(BASE_QPOS)
    for j in FINGER_CURL_JOINTS[finger]:
        joints[j] = BASE_QPOS[j] * best_scale
    achieved_z = fingertip_z(base, palm_z, joints, finger)
    return best_scale, achieved_z


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
    target_z = 0.005

    for palm_z in [0.08, 0.09, 0.10, 0.11, 0.12]:
        joints = dict(BASE_QPOS)
        tuned_scales = {}
        for finger in ["if", "mf", "rf", "th"]:
            scale, achieved_z = find_best_scale_for_target_z(base, palm_z, finger, target_z)
            tuned_scales[finger] = scale
            for j in FINGER_CURL_JOINTS[finger]:
                joints[j] = BASE_QPOS[j] * scale
            print(f"  palm_z={palm_z:.2f} {finger}: curl_scale={scale:.3f} -> tip_z={achieved_z:.4f}")

        info = score_pose(base, palm_z, joints, hold_steps=1000)
        print(f"palm_z={palm_z:.2f} scales={tuned_scales} -> {info}")
        print()


if __name__ == "__main__":
    main()
