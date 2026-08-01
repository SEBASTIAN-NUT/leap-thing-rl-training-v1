#!/usr/bin/env python3
"""真横からのスイング軌跡を記録（サイドビュー動画）"""
import math, argparse
import numpy as np
import mujoco

try:
    import imageio
    HAVE_VIDEO = True
except ImportError:
    HAVE_VIDEO = False
    print("WARNING: imageio not installed. Video recording disabled.")

XML = "thing_test/xmls/leap_thing_cpg.xml"
FINGER_CTRL = [(0, 2), (4, 6), (8, 10)]
REF_MCP_JOINT = "if_mcp"
REF_PIP_JOINT = "if_pip"
REF_TIP_GEOM = "if_tip_collision"
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

    renderer = None
    frames = []
    if HAVE_VIDEO:
        try:
            renderer = mujoco.Renderer(m, height=480, width=640)
        except Exception as e:
            print(f"WARNING: Renderer init failed ({e}); skipping video")
            renderer = None

    dt = m.opt.timestep
    nsteps = int(args.duration / dt)
    video_every = max(1, int(round(1.0 / (args.video_fps * dt))))

    q = np.array([float(d.qpos[qpos_mcp]), float(d.qpos[qpos_pip])])

    print(f"Recording swing trajectory (duration={args.duration}s, push={args.push}m, lift={args.lift}m)...")
    for step_i in range(nsteps):
        t = d.time
        ramp = min(1.0, t / WARMUP)
        d_scratch.qpos[3:7] = d.qpos[3:7]
        theta = 2 * math.pi * args.freq * t
        x_tgt, z_tgt = target_xz(theta)
        q = ik_solve(m, d_scratch, tip_gid, palm_bid, qpos_idx, dof_idx, jrange, q, x_tgt, z_tgt)

        ctrl = home_ctrl.copy()
        for mcp_idx, pip_idx in FINGER_CTRL:
            ctrl[mcp_idx] = home_ctrl[mcp_idx] + ramp * (q[0] - home_ctrl[mcp_idx])
            ctrl[pip_idx] = home_ctrl[pip_idx] + ramp * (q[1] - home_ctrl[pip_idx])
        d.ctrl[:] = ctrl
        mujoco.mj_step(m, d)

        if renderer is not None and step_i % video_every == 0:
            # カメラを真横から見る角度に設定（サイドビュー）
            renderer.update_scene(d)
            
            # カメラを真横から見る角度に設定（サイドビュー）
            frames.append(renderer.render().copy())

    if renderer is not None and frames:
        output_path = f"cpg_swing_trajectory_push{args.push:.2f}_lift{args.lift:.2f}.mp4"
        try:
            imageio.mimwrite(output_path, frames, fps=args.video_fps)
            print(f"✓ Saved: {output_path} ({len(frames)} frames)")
        except Exception as e:
            print(f"WARNING: video write failed ({e})")
    elif not HAVE_VIDEO:
        print("Video skipped: imageio not installed")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freq", type=float, default=1.0, help="CPG frequency [Hz]")
    p.add_argument("--push", type=float, default=0.02, help="stance press depth [m]")
    p.add_argument("--lift", type=float, default=0.03, help="swing lift height [m]")
    p.add_argument("--duration", type=float, default=5.0, help="recording duration [s]")
    p.add_argument("--video_fps", type=float, default=30.0)
    run(p.parse_args())
