#!/usr/bin/env python3
"""
poster_experiments.py - regenerate the push-force / tilt diagnostic data
as real CSV files (csv.writer), instead of printed tables that would
need to be manually retyped. Writes:
  - push_force_vs_grip.csv          (push depth -> measured contact force)
  - push_vs_tilt_no_thumb.csv       (push depth -> measured palm tilt)
  - push_vs_tilt_thumb_light.csv    (thumb resting, unloaded)
  - push_vs_tilt_thumb_loaded.csv   (thumb pressed to the same push depth)
"""
import csv
import numpy as np
import mujoco

XML = "thing_test/xmls/scene_flat_terrain_cpg.xml"
FINGERS = [
    dict(name="if", mcp="if_mcp", pip="if_pip", mcp_ctrl=0,  pip_ctrl=2,  tip="if_tip_collision"),
    dict(name="mf", mcp="mf_mcp", pip="mf_pip", mcp_ctrl=4,  pip_ctrl=6,  tip="mf_tip_collision"),
    dict(name="rf", mcp="rf_mcp", pip="rf_pip", mcp_ctrl=8,  pip_ctrl=10, tip="rf_tip_collision"),
]
THUMB = dict(name="th", mcp="th_cmc", pip="th_mcp", mcp_ctrl=12, pip_ctrl=14, tip="th_tip_collision")
PUSH_LEVELS = [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
PUSH_LEVELS_TILT = [0.02, 0.03, 0.04, 0.05, 0.06]


def joint_addrs(m, jname):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
    return m.jnt_qposadr[jid], m.jnt_dofadr[jid], m.jnt_range[jid].copy()


def ik_solve(m, d_s, tip_gid, palm_bid, qpos_idx, dof_idx, jrange,
             q_guess, x_target, z_target, iters=30, damping=1e-4):
    n = len(qpos_idx)
    q = np.array(q_guess, dtype=float)
    for k in range(n):
        d_s.qpos[qpos_idx[k]] = q[k]
    jacp = np.zeros((3, m.nv))
    target = np.array([x_target, z_target])
    for _ in range(iters):
        mujoco.mj_forward(m, d_s)
        tip, palm = d_s.geom_xpos[tip_gid], d_s.xpos[palm_bid]
        cur = np.array([tip[0] - palm[0], tip[2] - palm[2]])
        err = target - cur
        if err @ err < 1e-8:
            break
        mujoco.mj_jacGeom(m, d_s, jacp, None, tip_gid)
        J = jacp[np.ix_([0, 2], dof_idx)]
        dq = np.linalg.solve(J.T @ J + damping * np.eye(n), J.T @ err)
        q = np.clip(q + dq, jrange[:, 0], jrange[:, 1])
        for k in range(n):
            d_s.qpos[qpos_idx[k]] = q[k]
    return q


def x_at_z(m, d_s, tip_gid, palm_bid, qpos_mcp, mcp_range, z_query, n=500):
    mcps = np.linspace(mcp_range[0], mcp_range[1], n)
    xr = np.empty(n); zr = np.empty(n)
    for k, a in enumerate(mcps):
        d_s.qpos[qpos_mcp] = a
        mujoco.mj_forward(m, d_s)
        tip, palm = d_s.geom_xpos[tip_gid], d_s.xpos[palm_bid]
        xr[k] = tip[0] - palm[0]
        zr[k] = tip[2] - palm[2]
    i = int(np.argmin(np.abs(zr - z_query)))
    return float(xr[i])


def contact_normal_force(m, d, geom_ids):
    total = 0.0
    result = np.zeros(6)
    for i in range(d.ncon):
        con = d.contact[i]
        if con.geom1 in geom_ids or con.geom2 in geom_ids:
            mujoco.mj_contactForce(m, d, i, result)
            total += abs(result[0])
    return total


def setup_finger(m, f):
    tip_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f["tip"])
    qpos_mcp, dof_mcp, range_mcp = joint_addrs(m, f["mcp"])
    qpos_pip, dof_pip, range_pip = joint_addrs(m, f["pip"])
    return dict(tip_gid=tip_gid, mcp_ctrl=f["mcp_ctrl"], pip_ctrl=f["pip_ctrl"],
                qpos_idx=[qpos_mcp, qpos_pip], dof_idx=[dof_mcp, dof_pip],
                jrange=np.array([range_mcp, range_pip]), qpos_mcp=qpos_mcp,
                range_mcp=range_mcp)


def press_fingers(m, d, kid, palm_bid, finfo, push, ctrl_out=None):
    """Reset to home, press the given fingers to `push` depth, settle 500 steps."""
    mujoco.mj_resetDataKeyframe(m, d, kid)
    d.qpos[2] = 0.035
    mujoco.mj_forward(m, d)
    home_ctrl = d.ctrl.copy()
    z_contact = -d.xpos[palm_bid, 2]
    d_scratch = mujoco.MjData(m)

    ctrl = home_ctrl.copy()
    x_tgt_last = None
    for fi in finfo:
        d_scratch.qpos[:] = d.qpos
        x_tgt = x_at_z(m, d_scratch, fi["tip_gid"], palm_bid, fi["qpos_mcp"], fi["range_mcp"], z_contact - push)
        q = ik_solve(m, d_scratch, fi["tip_gid"], palm_bid, fi["qpos_idx"], fi["dof_idx"],
                     fi["jrange"], [0.0, 0.0], x_tgt, z_contact - push)
        ctrl[fi["mcp_ctrl"]] = q[0]
        ctrl[fi["pip_ctrl"]] = q[1]
        x_tgt_last = x_tgt

    d.ctrl[:] = ctrl
    for _ in range(500):
        mujoco.mj_step(m, d)
    return z_contact, x_tgt_last


def main():
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    palm_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm")
    up_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "upvector")
    up_adr = m.sensor_adr[up_sid]

    finfo = [setup_finger(m, f) for f in FINGERS]
    thinfo = setup_finger(m, THUMB)
    W = float(m.body_mass.sum() * 9.81)

    # 1) push_force_vs_grip.csv
    with open("push_force_vs_grip.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["push_m", "x_target_mm", "z_target_mm", "N_tip_total_N", "N_over_W", "ncon", "palm_z_mm", "total_weight_N"])
        for push in PUSH_LEVELS:
            z_contact, x_tgt = press_fingers(m, d, kid, palm_bid, finfo, push)
            N = contact_normal_force(m, d, [fi["tip_gid"] for fi in finfo])
            w.writerow([push, round(x_tgt * 1000, 2), round((z_contact - push) * 1000, 2),
                        round(N, 3), round(N / W, 3), d.ncon, round(d.xpos[palm_bid, 2] * 1000, 2), round(W, 3)])
    print("Wrote push_force_vs_grip.csv")

    # 2) push_vs_tilt_no_thumb.csv
    with open("push_vs_tilt_no_thumb.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["push_m", "palm_z_mm", "upvec_x", "upvec_y", "upvec_z", "tilt_deg"])
        for push in PUSH_LEVELS_TILT:
            press_fingers(m, d, kid, palm_bid, finfo, push)
            up = d.sensordata[up_adr:up_adr + 3]
            tilt_deg = float(np.degrees(np.arccos(np.clip(abs(up[2]), -1, 1))))
            w.writerow([push, round(d.xpos[palm_bid, 2] * 1000, 2),
                        round(float(up[0]), 4), round(float(up[1]), 4), round(float(up[2]), 4), round(tilt_deg, 2)])
    print("Wrote push_vs_tilt_no_thumb.csv")

    # 3) push_vs_tilt_thumb_light.csv (thumb resting, push=0, unloaded)
    with open("push_vs_tilt_thumb_light.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["push_m", "palm_z_mm", "upvec_x", "upvec_z", "tilt_deg"])
        for push in PUSH_LEVELS_TILT:
            mujoco.mj_resetDataKeyframe(m, d, kid)
            d.qpos[2] = 0.035
            mujoco.mj_forward(m, d)
            home_ctrl = d.ctrl.copy()
            z_contact = -d.xpos[palm_bid, 2]
            d_scratch = mujoco.MjData(m)
            ctrl = home_ctrl.copy()
            for fi in finfo:
                d_scratch.qpos[:] = d.qpos
                x_tgt = x_at_z(m, d_scratch, fi["tip_gid"], palm_bid, fi["qpos_mcp"], fi["range_mcp"], z_contact - push)
                q = ik_solve(m, d_scratch, fi["tip_gid"], palm_bid, fi["qpos_idx"], fi["dof_idx"],
                             fi["jrange"], [0.0, 0.0], x_tgt, z_contact - push)
                ctrl[fi["mcp_ctrl"]] = q[0]
                ctrl[fi["pip_ctrl"]] = q[1]
            d_scratch.qpos[:] = d.qpos
            x_tgt_th = x_at_z(m, d_scratch, thinfo["tip_gid"], palm_bid, thinfo["qpos_mcp"], thinfo["range_mcp"], z_contact)
            q_th = ik_solve(m, d_scratch, thinfo["tip_gid"], palm_bid, thinfo["qpos_idx"], thinfo["dof_idx"],
                            thinfo["jrange"], [0.0, 0.0], x_tgt_th, z_contact)
            ctrl[thinfo["mcp_ctrl"]] = q_th[0]
            ctrl[thinfo["pip_ctrl"]] = q_th[1]
            d.ctrl[:] = ctrl
            for _ in range(500):
                mujoco.mj_step(m, d)
            up = d.sensordata[up_adr:up_adr + 3]
            tilt_deg = float(np.degrees(np.arccos(np.clip(abs(up[2]), -1, 1))))
            w.writerow([push, round(d.xpos[palm_bid, 2] * 1000, 2),
                        round(float(up[0]), 4), round(float(up[2]), 4), round(tilt_deg, 2)])
    print("Wrote push_vs_tilt_thumb_light.csv")

    # 4) push_vs_tilt_thumb_loaded.csv (thumb pressed to same push depth)
    with open("push_vs_tilt_thumb_loaded.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["push_m", "palm_z_mm", "upvec_x", "upvec_z", "tilt_deg", "N_total_N"])
        for push in PUSH_LEVELS_TILT:
            all_finfo = finfo + [thinfo]
            press_fingers(m, d, kid, palm_bid, all_finfo, push)
            up = d.sensordata[up_adr:up_adr + 3]
            tilt_deg = float(np.degrees(np.arccos(np.clip(abs(up[2]), -1, 1))))
            N = contact_normal_force(m, d, [fi["tip_gid"] for fi in all_finfo])
            w.writerow([push, round(d.xpos[palm_bid, 2] * 1000, 2),
                        round(float(up[0]), 4), round(float(up[2]), 4), round(tilt_deg, 2), round(N, 3)])
    print("Wrote push_vs_tilt_thumb_loaded.csv")


if __name__ == "__main__":
    main()
