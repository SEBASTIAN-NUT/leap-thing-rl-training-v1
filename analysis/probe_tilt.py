#!/usr/bin/env python3
"""
probe_tilt.py - check whether the palm rise seen in probe_push_force.py
is a pure vertical translation or an actual tilt (rotation), using the
XML's existing "upvector" sensor (framezaxis of the imu site -- reads
(0,0,1) when level, deviates in x/y when the palm tips over). A single
point (sphere) contact per fingertip provides no resistance to rotation
about that point, so tipping (not just lifting) is a real possibility.
"""
import numpy as np
import mujoco

XML = "thing_test/xmls/scene_flat_terrain_cpg.xml"
FINGERS = [
    dict(name="if", mcp="if_mcp", pip="if_pip", mcp_ctrl=0,  pip_ctrl=2,  tip="if_tip_collision"),
    dict(name="mf", mcp="mf_mcp", pip="mf_pip", mcp_ctrl=4,  pip_ctrl=6,  tip="mf_tip_collision"),
    dict(name="rf", mcp="rf_mcp", pip="rf_pip", mcp_ctrl=8,  pip_ctrl=10, tip="rf_tip_collision"),
]


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


def main():
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    palm_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm")
    up_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "upvector")
    up_adr = m.sensor_adr[up_sid]

    finfo = []
    for f in FINGERS:
        tip_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f["tip"])
        qpos_mcp, dof_mcp, range_mcp = joint_addrs(m, f["mcp"])
        qpos_pip, dof_pip, range_pip = joint_addrs(m, f["pip"])
        finfo.append(dict(tip_gid=tip_gid, mcp_ctrl=f["mcp_ctrl"], pip_ctrl=f["pip_ctrl"],
                           qpos_idx=[qpos_mcp, qpos_pip], dof_idx=[dof_mcp, dof_pip],
                           jrange=np.array([range_mcp, range_pip]), qpos_mcp=qpos_mcp,
                           range_mcp=range_mcp))

    print(f"{'push[m]':>8} {'palm_z[mm]':>10} {'upvec_x':>8} {'upvec_y':>8} {'upvec_z':>8} {'tilt[deg]':>10}")
    for push in [0.02, 0.03, 0.04, 0.05, 0.06]:
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

        d.ctrl[:] = ctrl
        for _ in range(500):
            mujoco.mj_step(m, d)

        up = d.sensordata[up_adr:up_adr+3]
        tilt_deg = np.degrees(np.arccos(np.clip(abs(up[2]), -1, 1)))
        print(f"{push:8.3f} {d.xpos[palm_bid,2]*1000:10.1f} "
              f"{up[0]:8.3f} {up[1]:8.3f} {up[2]:8.3f} {tilt_deg:10.2f}")


if __name__ == "__main__":
    main()
