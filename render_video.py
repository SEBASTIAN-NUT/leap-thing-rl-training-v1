#!/usr/bin/env python3
"""Render trained policy as MP4.

Usage (from thing_project/):
  Open_Duck_Playground/.venv/bin/python render_video.py \
    --onnx checkpoints/go1_exact_2026_07_02_201802/2026_07_02_234906_162201600.onnx \
    --out render_162M.mp4 --command 0.5 0.0 0.0
"""

import argparse
import os
import sys
import numpy as np

os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/jax_thing_cache"))
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import mujoco
import onnxruntime as ort


def get_sensor(mj_model, mj_data, name):
    sid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr = mj_model.sensor_adr[sid]
    dim = mj_model.sensor_dim[sid]
    return np.array(mj_data.sensordata[adr : adr + dim])


def build_obs(mj_model, mj_data, env, command, last_act, last_last_act, last_last_last_act, motor_targets):
    addrs = np.array([mj_model.jnt_qposadr[jid] for jid in env.actuator_joint_ids])
    joint_angles = mj_data.qpos[addrs]
    joint_vel = mj_data.qvel[env.actuator_qvel_addr]

    gyro = get_sensor(mj_model, mj_data, "imu_gyro")
    gravity = get_sensor(mj_model, mj_data, "upvector")
    linvel = get_sensor(mj_model, mj_data, "local_linvel")

    floor_id = mj_model.geom("floor").id
    fingertip_names = ["if_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip_collision"]
    fingertip_ids = [mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, n) for n in fingertip_names]
    contact = np.zeros(4)
    for i, gid in enumerate(fingertip_ids):
        for j in range(mj_data.ncon):
            c = mj_data.contact[j]
            if (c.geom1 == gid and c.geom2 == floor_id) or (c.geom2 == gid and c.geom1 == floor_id):
                contact[i] = 1.0
                break

    obs = np.concatenate([
        command,
        joint_angles - env._default_actuator,
        joint_vel * env._config.dof_vel_scale,
        last_act,
        last_last_act,
        last_last_last_act,
        motor_targets,
        contact,
        gyro,
        gravity,
        linvel,
    ])
    assert obs.shape == (112,), f"Expected 112, got {obs.shape}"
    return obs.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--out", default="render.mp4")
    parser.add_argument("--command", nargs=3, type=float, default=[0.5, 0.0, 0.0])
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=640)
    args = parser.parse_args()

    command = np.array(args.command)

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from thing_test import thing_walk
    env = thing_walk.Joystick(task="flat_terrain")
    mj_model = env.mj_model
    default_actuator = np.array(env._default_actuator)
    action_scale = env._config.action_scale
    ctrl_dt = env._config.ctrl_dt
    sim_dt = env._config.sim_dt
    n_substeps = int(round(ctrl_dt / sim_dt))

    sess = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    print(f"ONNX input : {sess.get_inputs()[0].name}  shape={sess.get_inputs()[0].shape}")
    print(f"ONNX output: {sess.get_outputs()[0].name}  shape={sess.get_outputs()[0].shape}")

    def run_policy(obs):
        return sess.run(None, {inp_name: obs[None]})[0][0]

    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    addrs = np.array([mj_model.jnt_qposadr[jid] for jid in env.actuator_joint_ids])
    for i, addr in enumerate(addrs):
        mj_data.qpos[addr] = default_actuator[i]
    mujoco.mj_forward(mj_model, mj_data)

    last_act = np.zeros(16)
    last_last_act = np.zeros(16)
    last_last_last_act = np.zeros(16)
    motor_targets = default_actuator.copy()

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = mj_model.body("palm").id
    cam.distance = 0.8
    cam.elevation = -20
    cam.azimuth = 120

    renderer = mujoco.Renderer(mj_model, height=args.height, width=args.width)
    frames = []

    print(f"Rendering {args.steps} steps (command={command})...")
    for step in range(args.steps):
        obs = build_obs(mj_model, mj_data, env, command, last_act, last_last_act, last_last_last_act, motor_targets)
        action = run_policy(obs)

        motor_targets = default_actuator + action * action_scale
        mj_data.ctrl[:] = motor_targets

        for _ in range(n_substeps):
            mujoco.mj_step(mj_model, mj_data)

        renderer.update_scene(mj_data, camera=cam)
        frames.append(renderer.render().copy())

        last_last_last_act = last_last_act.copy()
        last_last_act = last_act.copy()
        last_act = action.copy()

        if step % 100 == 0:
            palm_id = mj_model.body("palm").id
            z = mj_data.xpos[palm_id][2]
            print(f"  step {step:4d}  palm_z={z:.3f}m")

    import imageio
    fps = int(round(1.0 / ctrl_dt))
    imageio.mimwrite(args.out, frames, fps=fps)
    print(f"Saved: {args.out}  ({len(frames)} frames @ {fps} fps)")


if __name__ == "__main__":
    main()
