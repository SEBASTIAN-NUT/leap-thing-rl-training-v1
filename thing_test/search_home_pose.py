"""Search for a physically stable "stand on fingertips" home pose for the
LEAP hand, by grid-searching palm height + finger curl scale and scoring
each candidate via a short PD-hold simulation (standard MuJoCo, same idea
as diag_home_pose.py).

Stability score (lower is better) combines:
  - max |qacc| over the hold (numerical blow-up signal)
  - palm_z drift from its starting value (falling/sinking signal)
  - whether the hand flips over (upvector z < 0 at any point)
  - whether at least 3 of the 4 fingertips stay in contact most of the time

Usage:
    Open_Duck_Playground/.venv/bin/python -m thing_test.search_home_pose
"""
import itertools
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
CURL_JOINTS = ["if_pip", "if_dip", "mf_pip", "mf_dip", "rf_pip", "rf_dip", "th_mcp", "th_ipl"]

BASE_QPOS = [
    0.3220, -0.3350, 1.1318, 0.3684,
    0.3220, -0.0000, 1.2992, -0.3660,
    0.2075, 0.5235, 1.0481, 0.0434,
    1.8131, -0.3490, -0.4700, 0.5115,
]
ORIENT_QUAT = [0.9999, 0.0052, -0.0147, -0.0004]


def build_qpos(palm_z, curl_scale):
    joints = dict(zip(ACTUATOR_ORDER, BASE_QPOS))
    for name in CURL_JOINTS:
        joints[name] = joints[name] * curl_scale
    joint_vals = [joints[n] for n in ACTUATOR_ORDER]
    return [0.0165, 0.0052, palm_z] + ORIENT_QUAT + joint_vals


def score_pose(base, palm_z, curl_scale, hold_steps=300):
    qpos = np.array(build_qpos(palm_z, curl_scale))
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

    palm_z_end = base.data.qpos[2]
    palm_drift = abs(palm_z_end - palm_z_start)
    contact_frac = contact_ok_steps / hold_steps

    penalty = 0.0
    if nan_hit:
        penalty += 1e6
    if flipped:
        penalty += 1e6
    penalty += max_qacc
    penalty += palm_drift * 1000.0
    penalty += (1.0 - contact_frac) * 500.0
    return penalty, dict(
        max_qacc=max_qacc, palm_drift=palm_drift, flipped=flipped,
        nan_hit=nan_hit, contact_frac=contact_frac, palm_z_end=palm_z_end,
    )


def main():
    base = MJInferBase(str(constants.task_to_xml("flat_terrain")))

    palm_zs = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    curl_scales = [0.7, 0.85, 1.0, 1.15, 1.3]

    results = []
    for palm_z, curl_scale in itertools.product(palm_zs, curl_scales):
        penalty, info = score_pose(base, palm_z, curl_scale)
        results.append((penalty, palm_z, curl_scale, info))
        print(f"palm_z={palm_z:.2f} curl_scale={curl_scale:.2f} "
              f"penalty={penalty:9.2f}  {info}")

    results.sort(key=lambda r: r[0])
    print("\n=== Top 5 candidates ===")
    for penalty, palm_z, curl_scale, info in results[:5]:
        print(f"palm_z={palm_z:.2f} curl_scale={curl_scale:.2f} "
              f"penalty={penalty:9.2f}  {info}")


if __name__ == "__main__":
    main()
