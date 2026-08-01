#!/usr/bin/env python3
"""
probe_tilt_thumb2.py - retest with the thumb actually loaded (pressed to
the same push depth as if/mf/rf), not just resting at light contact.
An unloaded contact carries ~0 normal force and so contributes ~0
moment resistance -- probe_tilt_thumb.py's null result showed exactly
this. This time the thumb presses down too, so it can actually
contribute a counter-moment against if/mf/rf's forward-lever tipping.
"""
import numpy as np
import mujoco

XML = "thing_test/xmls/scene_flat_terrain_cpg.xml"
FINGERS = [
    dict(name="if", mcp="if_mcp", pip="if_pip", mcp_ctrl=0,  pip_ctrl=2,  tip="if_tip_collision"),
    dict(name="mf", mcp="mf_mcp", pip="mf_pip", mcp_ctrl=4,  pip_ctrl=6,  tip="mf_tip_collision"),
    dict(name="rf", mcp="rf_mcp", pip="rf_pip", mcp_ctrl=8,  pip_ctrl=10, tip="rf_tip_collision"),
]
THUMB = dict(name="th", mcp="th_cmc", pip="th_mcp", mcp_ctrl=12, pip_ctrl=14, tip="th_tip_collision")


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


def main():
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    palm_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm")
    up_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "upvector")
    up_adr = m.sensor_adr[up_sid]

    finfo = [setup_finger(m, f) for f in FINGERS]
    thinfo = setup_finger(m, THUMB)
    all_tip_gids = [fi["tip_gid"] for fi in finfo] + [thinfo["tip_gid"]]

    print("Thumb pressed to the SAME push depth as if/mf/rf (actually loaded):")
    print(f"{'push[m]':>8} {'palm_z[mm]':>10} {'upvec_x':>8} {'upvec_z':>8} {'tilt[deg]':>10} {'N_total[N]':>10}")
    for push in [0.02, 0.03, 0.04, 0.05, 0.06]:
        mujoco.mj_resetDataKeyframe(m, d, kid)
        d.qpos[2] = 0.035
        mujoco.mj_forward(m, d)
        home_ctrl = d.ctrl.copy()
        z_contact = -d.xpos[palm_bid, 2]
        d_scratch = mujoco.MjData(m)

        ctrl = home_ctrl.copy()
        for fi in finfo + [thinfo]:
            d_scratch.qpos[:] = d.qpos
            x_tgt = x_at_z(m, d_scratch, fi["tip_gid"], palm_bid, fi["qpos_mcp"], fi["range_mcp"], z_contact - push)
            q = ik_solve(m, d_scratch, fi["tip_gid"], palm_bid, fi["qpos_idx"], fi["dof_idx"],
                         fi["jrange"], [0.0, 0.0], x_tgt, z_contact - push)
            ctrl[fi["mcp_ctrl"]] = q[0]
            ctrl[fi["pip_ctrl"]] = q[1]

        d.ctrl[:] = ctrl
        for _ in range(500):
            mujoco.mj_step(m, d)

        up = d.sensordata[up_adr:up_adr+3]
        tilt_deg = np.degrees(np.arccos(np.clip(abs(up[2]), -1, 1)))
        N = contact_normal_force(m, d, all_tip_gids)
        print(f"{push:8.3f} {d.xpos[palm_bid,2]*1000:10.1f} {up[0]:8.3f} {up[2]:8.3f} "
              f"{tilt_deg:10.2f} {N:10.2f}")


if __name__ == "__main__":
    main()
