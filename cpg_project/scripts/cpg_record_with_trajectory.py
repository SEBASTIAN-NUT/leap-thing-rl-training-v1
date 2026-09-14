#!/usr/bin/env python3
"""CPG record with trajectory overlay using mjvGeom"""
import math, argparse
import numpy as np
import mujoco

try:
    import imageio
    HAVE_VIDEO = True
except ImportError:
    HAVE_VIDEO = False

XML = "models/leap_thing_cpg.xml"

FINGER_CTRL = [(0, 2), (4, 6), (8, 10)]
REF_MCP_JOINT = "if_mcp"
REF_PIP_JOINT = "if_pip"
REF_TIP_GEOM  = "if_tip_collision"

WARMUP = 1.0
TIP_RADIUS = 0.009

trajectory_points = []

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

def body_com(m, d):
    return (d.xipos * m.body_mass[:, None]).sum(axis=0) / m.body_mass.sum()

def run_cpg(push_m=0.020, lift_m=0.030, freq_hz=1.0, duration=1.0, output_prefix="cpg_traj"):
    print(f"Loading {XML}...")
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(m, d, kid)
    d.qpos[2] = 0.035
    mujoco.mj_forward(m, d)
    
    palm_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm")
    tip_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, REF_TIP_GEOM)
    
    mcp_addr, mcp_dof, mcp_range = joint_addrs(m, REF_MCP_JOINT)
    pip_addr, pip_dof, pip_range = joint_addrs(m, REF_PIP_JOINT)
    
    print(f"CPG: push={push_m:.3f}m, lift={lift_m:.3f}m, freq={freq_hz:.1f}Hz, duration={duration:.1f}s")
    
    dt = m.opt.timestep
    nsteps = int(duration / dt)
    
    renderer = None
    if HAVE_VIDEO:
        try:
            renderer = mujoco.Renderer(m, height=480, width=640)
        except:
            renderer = None
    
    frames = []
    q = np.array([float(d.qpos[mcp_addr]), float(d.qpos[pip_addr])])
    
    d_scratch = mujoco.MjData(m)
    z_contact = -d.xpos[palm_bid, 2]
    x_front, _ = x_at_z(m, d_scratch, tip_gid, palm_bid, mcp_addr, mcp_range, z_contact)
    x_back, _  = x_at_z(m, d_scratch, tip_gid, palm_bid, mcp_addr, mcp_range, z_contact - push_m)
    
    def target_xz(theta):
        s = theta % (2 * math.pi)
        if s <= math.pi:
            frac = (1 - math.cos(s)) / 2.0
            x = x_back + (x_front - x_back) * frac
            z = z_contact + lift_m * math.sin(s)
            return x, z
        frac = (s - math.pi) / math.pi
        x = x_front - (x_front - x_back) * frac
        z = z_contact - push_m
        return x, z
    
    print(f"Running {duration:.1f}s...")
    for step_i in range(nsteps):
        t = d.time
        ramp = min(1.0, t / WARMUP)
        d_scratch.qpos[3:7] = d.qpos[3:7]
        theta = 2 * math.pi * freq_hz * t
        x_tgt, z_tgt = target_xz(theta)
        q = ik_solve(m, d_scratch, tip_gid, palm_bid, [mcp_addr, pip_addr], 
                    [mcp_dof, pip_dof], np.array([mcp_range, pip_range]),
                    q, x_tgt, z_tgt)
        
        for mcp_i, pip_i in FINGER_CTRL:
            d.qpos[mcp_i] = q[0]
            d.qpos[pip_i] = q[1]
        
        mujoco.mj_step(m, d)
        
        if t >= WARMUP:
            mujoco.mj_forward(m, d)
            tip = d.geom_xpos[tip_gid]
            trajectory_points.append(tip.copy())
        
        if renderer is not None:
            mujoco.mj_forward(m, d)
            
            # mjvScene に軌跡を追加
            renderer.update_scene(d)
            scn = renderer._scene
            
            # 軌跡をラインセグメントとして描画
            if len(trajectory_points) > 1:
                for i in range(len(trajectory_points) - 1):
                    p1 = trajectory_points[i]
                    p2 = trajectory_points[i + 1]
                    
                    # mjvGeom でラインを作成
                    geom = mujoco.MjvGeom()
                    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_LINE, 
                                       np.zeros(3), np.zeros(3), 
                                       np.zeros(9), np.array([0.0, 100.0/255.0, 1.0, 1.0]))
                    
                    # ラインの両端点を設定
                    geom.pos[:] = p1
                    geom.mat[:9] = (p2 - p1) / np.linalg.norm(p2 - p1)
                    geom.size[:] = [np.linalg.norm(p2 - p1) / 2, 0.002, 0.002]
                    
                    mujoco.mjv_makeConnector(geom, mujoco.mjtGeom.mjGEOM_LINE, 0.001, 
                                           p1[0], p1[1], p1[2], p2[0], p2[1], p2[2])
                    scn.geoms[scn.ngeom] = geom
                    scn.ngeom += 1
            
            frames.append(renderer.render().copy())
    
    if renderer is not None and frames:
        video_path = output_prefix + "_with_trajectory.mp4"
        imageio.mimwrite(video_path, frames, fps=30)
        print(f"Wrote {video_path} ({len(frames)} frames)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--freq", type=float, default=1.0)
    p.add_argument("--push", type=float, default=0.02)
    p.add_argument("--lift", type=float, default=0.03)
    p.add_argument("--duration", type=float, default=1.0)
    p.add_argument("--out_prefix", default="cpg_traj")
    args = p.parse_args()
    
    run_cpg(push_m=args.push, lift_m=args.lift, freq_hz=args.freq,
            duration=args.duration, output_prefix=args.out_prefix)
