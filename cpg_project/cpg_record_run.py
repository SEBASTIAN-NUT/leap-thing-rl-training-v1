#!/usr/bin/env python3
"""
cpg_record.py - headless CPG run with logging, for reporting.

Final validated design (matches cpg_viewer.py):
  - IK solved ONCE (index finger's chain) and applied identically to
    if/mf/rf (avoids independent-IK branch divergence between fingers)
  - swing: eased arc, lift height INDEPENDENT of stride (must exceed the
    9mm fingertip sphere radius to actually clear the floor)
  - stance: straight line at constant ground-contact depth (push)
  - default push=0.02m, lift=0.03m -- validated stable for 22+ cycles
    with steadily growing propulsion (see poster_data/08_*.csv)

Records, non-interactively for a fixed duration:
  - MP4 video (offscreen renderer; skipped with a warning if imageio is
    not installed -- CSV logging still runs regardless)
  - whole-body center of mass (mass-weighted over all bodies, via
    d.xipos * body_mass) + palm tilt (upvector sensor) -> <out_prefix>_com.csv
  - index finger ("if"): fingertip position (world + palm-relative),
    commanded IK target (x,z), and all 4 joint angles
    (MCP, ROT, PIP, DIP) -> <out_prefix>_if_finger.csv
"""

import math, argparse, csv
import numpy as np
import mujoco

try:
    import imageio
    HAVE_VIDEO = True
except ImportError:
    HAVE_VIDEO = False

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
    """Damped least squares (Levenberg-Marquardt) IK."""
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


def body_com(m, d):
    """Whole-body mass-weighted center of mass (world frame)."""
    return (d.xipos * m.body_mass[:, None]).sum(axis=0) / m.body_mass.sum()


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
    log_joint_names = ["if_mcp", "if_rot", "if_pip", "if_dip"]
    log_qpos_idx = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j)]
                    for j in log_joint_names]

    print("Stage 2 (final): single shared IK (MCP+PIP) applied identically to if/mf/rf")
    d_scratch.qpos[:] = d.qpos
    z_contact = -d.xpos[palm_bid, 2]
    x_front, _ = x_at_z(m, d_scratch, tip_gid, palm_bid, qpos_mcp, range_mcp, z_contact)
    x_back, _  = x_at_z(m, d_scratch, tip_gid, palm_bid, qpos_mcp, range_mcp, z_contact - args.push)
    print(f"  shared: x_front={x_front*1000:+.1f}mm x_back={x_back*1000:+.1f}mm "
          f"stride={(x_front-x_back)*1000:.1f}mm  lift={args.lift*1000:.1f}mm")
    print(f"Trajectory params: push={args.push:.3f}m  lift={args.lift:.3f}m  freq={args.freq:.1f}Hz")

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

    renderer = None
    frames = []
    if HAVE_VIDEO:
        try:
            renderer = mujoco.Renderer(m, height=480, width=640)
        except Exception as e:
            print(f"WARNING: could not create offscreen renderer ({e}); "
                  f"skipping video. Try `export MUJOCO_GL=egl` and rerun if this "
                  f"is a GL-context error. CSV logging continues.")
            renderer = None

    dt = m.opt.timestep
    nsteps = int(args.duration / dt)
    video_every = max(1, int(round(1.0 / (args.video_fps * dt))))

    com_rows = []
    finger_rows = []
    q = np.array([float(d.qpos[qpos_mcp]), float(d.qpos[qpos_pip])])

    print(f"Running {args.duration:.1f}s ({nsteps} steps, dt={dt:.4f}s)...")
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

        com = body_com(m, d)
        palm_w = d.xpos[palm_bid]
        up = d.sensordata[up_adr:up_adr + 3]
        tilt_deg = float(np.degrees(np.arccos(np.clip(abs(up[2]), -1, 1))))
        com_rows.append([t, com[0], com[1], com[2], palm_w[0], palm_w[1], palm_w[2],
                          up[0], up[1], up[2], tilt_deg])

        tip_w = d.geom_xpos[tip_gid]
        tip_rel = tip_w - palm_w
        joint_angles = [d.qpos[qi] for qi in log_qpos_idx]
        finger_rows.append([t, tip_w[0], tip_w[1], tip_w[2],
                             tip_rel[0], tip_rel[1], tip_rel[2],
                             x_tgt, z_tgt, *joint_angles])

        if renderer is not None and step_i % video_every == 0:
            renderer.update_scene(d)
            frames.append(renderer.render().copy())

    with open(args.out_prefix + "_com.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["t", "com_x", "com_y", "com_z", "palm_x", "palm_y", "palm_z",
                    "upvec_x", "upvec_y", "upvec_z", "tilt_deg"])
        w.writerows(com_rows)

    with open(args.out_prefix + "_if_finger.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["t", "tip_x_world", "tip_y_world", "tip_z_world",
                    "tip_x_rel", "tip_y_rel", "tip_z_rel",
                    "x_target_rel", "z_target_rel",
                    "if_mcp", "if_rot", "if_pip", "if_dip"])
        w.writerows(finger_rows)

    print(f"Wrote {args.out_prefix}_com.csv ({len(com_rows)} rows)")
    print(f"Wrote {args.out_prefix}_if_finger.csv ({len(finger_rows)} rows)")

    if renderer is not None and frames:
        video_path = args.out_prefix + ".mp4"
        try:
            imageio.mimwrite(video_path, frames, fps=args.video_fps)
            print(f"Wrote {video_path} ({len(frames)} frames @ {args.video_fps}fps)")
        except Exception as e:
            print(f"WARNING: video write failed ({e}). "
                  f"Try `pip install imageio[ffmpeg]` in this venv.")
    elif not HAVE_VIDEO:
        print("Video skipped: imageio not installed "
              "(pip install imageio[ffmpeg] to enable).")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freq", type=float, default=1.0,  help="CPG freq [Hz]")
    p.add_argument("--push", type=float, default=0.02, help="stance press depth [m]")
    p.add_argument("--lift", type=float, default=0.03, help="swing lift height [m]")
    p.add_argument("--duration", type=float, default=20.0, help="sim seconds to run")
    p.add_argument("--video_fps", type=float, default=30.0)
    p.add_argument("--out_prefix", type=str, default="cpg_run_final")
    run(p.parse_args())
