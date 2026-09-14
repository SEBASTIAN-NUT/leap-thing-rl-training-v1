#!/usr/bin/env python3
"""
log_cycle_summary.py - headless re-run of the shared-IK CPG (same design
as cpg_viewer.py) that writes a per-cycle summary CSV instead of only
printing it, so results like the lift=0.06 flip event can be captured
as a real file rather than a transcribed console log. Writes:
  - cycle_summary_liftXXX.csv  (t, cycle, ncon, palm_x, dx, tilt, upvec)
"""
import math, argparse, csv
import numpy as np
import mujoco

XML = "thing_test/xmls/scene_flat_terrain_cpg.xml"
FINGER_CTRL = [(0, 2), (4, 6), (8, 10)]
REF_MCP_JOINT = "if_mcp"
REF_PIP_JOINT = "if_pip"
REF_TIP_GEOM = "if_tip_collision"
PHASES = [0.0, 0.0, 0.0]
WARMUP = 1.0


def joint_addrs(m, jname):
    jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
    return m.jnt_qposadr[jid], m.jnt_dofadr[jid], m.jnt_range[jid].copy()


def ik_solve(m, d_s, tip_gid, palm_bid, qpos_idx, dof_idx, jrange,
             q_guess, x_target, z_target, iters=10, damping=1e-4):
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


def run(args):
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(m, d, kid)
    d.qpos[2] = 0.035
    mujoco.mj_forward(m, d)
    home_ctrl = d.ctrl.copy()
    palm_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm")
    up_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, "upvector")
    up_adr = m.sensor_adr[up_sid]
    d_scratch = mujoco.MjData(m)

    tip_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, REF_TIP_GEOM)
    qpos_mcp, dof_mcp, range_mcp = joint_addrs(m, REF_MCP_JOINT)
    qpos_pip, dof_pip, range_pip = joint_addrs(m, REF_PIP_JOINT)
    qpos_idx = [qpos_mcp, qpos_pip]
    dof_idx = [dof_mcp, dof_pip]
    jrange = np.array([range_mcp, range_pip])

    d_scratch.qpos[:] = d.qpos
    z_contact = -d.xpos[palm_bid, 2]
    x_front = x_at_z(m, d_scratch, tip_gid, palm_bid, qpos_mcp, range_mcp, z_contact)
    x_back = x_at_z(m, d_scratch, tip_gid, palm_bid, qpos_mcp, range_mcp, z_contact - args.push)

    def target_xz(theta):
        s = theta % (2 * math.pi)
        if s <= math.pi:
            frac = (1 - math.cos(s)) / 2.0
            x = x_back + (x_front - x_back) * frac
            z = z_contact + args.lift * math.sin(s)
            return x, z
        frac = (s - math.pi) / math.pi
        x = x_front - (x_front - x_back) * frac
        z = z_contact - args.push
        return x, z

    q = np.array([float(d.qpos[qpos_mcp]), float(d.qpos[qpos_pip])])
    dt = m.opt.timestep
    nsteps = int(args.duration / dt)

    rows = []
    prev_cycle = -1
    prev_x = float(d.xpos[palm_bid, 0])

    for step_i in range(nsteps):
        t = d.time
        ramp = min(1.0, t / WARMUP)
        d_scratch.qpos[3:7] = d.qpos[3:7]
        theta = 2 * math.pi * args.freq * t + PHASES[0]
        x_tgt, z_tgt = target_xz(theta)
        q = ik_solve(m, d_scratch, tip_gid, palm_bid, qpos_idx, dof_idx, jrange,
                     q, x_tgt, z_tgt)

        ctrl = home_ctrl.copy()
        for mcp_idx, pip_idx in FINGER_CTRL:
            ctrl[mcp_idx] = home_ctrl[mcp_idx] + ramp * (q[0] - home_ctrl[mcp_idx])
            ctrl[pip_idx] = home_ctrl[pip_idx] + ramp * (q[1] - home_ctrl[pip_idx])
        d.ctrl[:] = ctrl
        mujoco.mj_step(m, d)

        px = float(d.xpos[palm_bid, 0])
        cycle = int(t * args.freq)
        if cycle != prev_cycle and t > WARMUP:
            dx = px - prev_x
            up = d.sensordata[up_adr:up_adr + 3]
            tilt_deg = float(np.degrees(np.arccos(np.clip(abs(up[2]), -1, 1))))
            rows.append([round(t, 1), cycle, int(d.ncon), round(px * 1000, 2),
                         round(dx * 1000, 2), round(tilt_deg, 2),
                         round(float(up[0]), 4), round(float(up[1]), 4), round(float(up[2]), 4)])
            prev_cycle = cycle
            prev_x = px

    out_name = f"cycle_summary_lift{args.lift:.3f}".replace(".", "p") + ".csv"
    with open(out_name, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["t_s", "cycle", "ncon", "palm_x_mm", "dx_mm_per_cycle",
                    "tilt_deg", "upvec_x", "upvec_y", "upvec_z"])
        w.writerows(rows)
    print(f"Wrote {out_name} ({len(rows)} rows)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freq", type=float, default=1.0)
    p.add_argument("--push", type=float, default=0.02)
    p.add_argument("--lift", type=float, default=0.06)
    p.add_argument("--duration", type=float, default=8.0)
    run(p.parse_args())
