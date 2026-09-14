#!/usr/bin/env python3
"""
CPG viewer: fingertip-trajectory based locomotion for LEAP hand.

Stage 2: explicit (x,z) fingertip trajectory + Jacobian-based inverse
kinematics (damped least squares / Levenberg-Marquardt IK). IK is solved
ONCE (index finger's chain) and applied identically to if/mf/rf (see
prior version's changelog -- avoids independent-IK branch divergence).

This version adds per-cycle tilt logging (via the XML's "upvector"
sensor) to test the hypothesis that sudden flips at larger --lift values
are caused by rocking disturbance accumulating cycle-over-cycle (each
fast, large-angle swing imparts a reaction torque on the palm that does
not fully damp out before the next cycle) until the tilt crosses a
critical angle, rather than a sudden one-off glitch.
"""

import math, argparse
import numpy as np
import mujoco, mujoco.viewer

XML = "../thing_test/xmls/leap_thing_cpg.xml"

FINGER_CTRL = [(0, 2), (4, 6), (8, 10)]   # if, mf, rf
REF_MCP_JOINT = "if_mcp"
REF_PIP_JOINT = "if_pip"
REF_TIP_GEOM  = "if_tip_collision"

PHASES = [0.0, 0.0, 0.0]
WARMUP = 1.0
TIP_RADIUS = 0.009  # m


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
    xr = np.empty(n)
    zr = np.empty(n)
    for k, a in enumerate(mcps):
        d_s.qpos[qpos_mcp] = a
        mujoco.mj_forward(m, d_s)
        tip, palm = d_s.geom_xpos[tip_gid], d_s.xpos[palm_bid]
        xr[k] = tip[0] - palm[0]
        zr[k] = tip[2] - palm[2]
    i = int(np.argmin(np.abs(zr - z_query)))
    return float(xr[i]), float(mcps[i])


def run(args):
    if args.lift <= TIP_RADIUS:
        print(f"WARNING: --lift={args.lift:.4f}m does not clear the fingertip "
              f"sphere radius ({TIP_RADIUS:.3f}m).")

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

    print("Stage 2: single shared IK (MCP+PIP) applied identically to if/mf/rf")
    print("Trajectory: independent-lift arc swing / straight-line stance")
    d_scratch.qpos[:] = d.qpos
    z_contact = -d.xpos[palm_bid, 2]
    x_front, _ = x_at_z(m, d_scratch, tip_gid, palm_bid, qpos_mcp, range_mcp, z_contact)
    x_back, _  = x_at_z(m, d_scratch, tip_gid, palm_bid, qpos_mcp, range_mcp, z_contact - args.push)
    print(f"  shared: x_front={x_front*1000:+.1f}mm x_back={x_back*1000:+.1f}mm "
          f"stride={(x_front-x_back)*1000:.1f}mm  lift={args.lift*1000:.1f}mm")
    print(f"Trajectory params: push={args.push:.3f}m  lift={args.lift:.3f}m  freq={args.freq:.1f}Hz")
    print(f"Initial: ncon={d.ncon}  palm_z={d.xpos[palm_bid,2]:.4f}m\n")

    q = np.array([float(d.qpos[qpos_mcp]), float(d.qpos[qpos_pip])])
    state = {"prev_cycle": -1, "prev_x": d.xpos[palm_bid, 0]}

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

    def step(m, d):
        nonlocal q
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

        px = d.xpos[palm_bid, 0]
        cycle = int(t * args.freq)
        if cycle != state["prev_cycle"] and t > WARMUP:
            dx = px - state["prev_x"]
            up = d.sensordata[up_adr:up_adr+3]
            tilt_deg = np.degrees(np.arccos(np.clip(abs(up[2]), -1, 1)))
            print(f"t={t:5.1f}s  cycle={cycle:3d}  ncon={d.ncon:2d}"
                  f"  palm_X={px*1000:+7.2f}mm  dx={dx*1000:+6.2f}mm/cycle"
                  f"  tilt={tilt_deg:6.2f}deg  upvec=({up[0]:+.3f},{up[1]:+.3f},{up[2]:+.3f})")
            state["prev_cycle"] = cycle
            state["prev_x"] = px

    with mujoco.viewer.launch_passive(m, d) as viewer:
        frame_count = 0
    while viewer.is_running():
        frame_count += 1
        if frame_count > 1: break
            step(m, d)
            viewer.sync()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freq", type=float, default=1.0,  help="CPG freq [Hz]")
    p.add_argument("--push", type=float, default=0.02, help="stance press depth [m]")
    p.add_argument("--lift", type=float, default=0.03, help="swing lift height [m]")
    run(p.parse_args())

# キー入力でカメラ位置を出力する機能を追加
def print_camera_info(viewer):
    """現在のカメラ位置を出力（XMLに埋め込む用）"""
    cam = viewer.cam
    print(f'\n=== Current Camera Position (XMLに埋め込む) ===')
    print(f'  pos="{cam.lookat[0]:.2f} {cam.lookat[1]:.2f} {cam.lookat[2]:.2f}"')
    print(f'  xyaxes="{cam.forward[0]:.1f} {cam.forward[1]:.1f} {cam.forward[2]:.1f} {cam.up[0]:.1f} {cam.up[1]:.1f} {cam.up[2]:.1f}"')
    print('=====================================\n')

# Override to disable motor commands (keep hand at home pose)
import sys
if '--zero-control' in sys.argv:
    # Modify the run function to zero out control
    original_run = run
    def run_zero(args):
        # This would require modifying the function
        pass
