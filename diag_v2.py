import numpy as np
import mujoco
import onnxruntime as ort

MODEL_PATH = "thing_test/xmls/scene_flat_terrain.xml"
ONNX_PATH = "checkpoints/2026_06_10_202345_1064960.onnx"

ACTUATORS = ['if_mcp','if_rot','if_pip','if_dip','mf_mcp','mf_rot','mf_pip','mf_dip',
              'rf_mcp','rf_rot','rf_pip','rf_dip','th_cmc','th_axl','th_mcp','th_ipl']
DEFAULT_ACTUATOR = np.array([0.322, -0.335, 1.1318, 0.3684, 0.322, -0.0, 1.2992, -0.366,
                               0.2075, 0.5235, 1.0481, 0.0434, 1.8131, -0.349, -0.47, 0.5115])
TIP_GEOMS = ["if_tip", "mf_tip", "rf_tip", "th_tip"]

ACTION_SCALE = 0.25
DOF_VEL_SCALE = 0.05
MAX_MOTOR_VELOCITY = 5.24
SIM_DT = 0.002
CTRL_DT = 0.02
DECIMATION = round(CTRL_DT / SIM_DT)
MAX_DELTA = MAX_MOTOR_VELOCITY * SIM_DT * DECIMATION

NOISE_LEVEL = 1.0
N_JOINT_POS = 0.05
N_JOINT_VEL = 2.5
N_GRAVITY = 0.1
N_GYRO = 0.1
ACTION_MIN_DELAY = 0
ACTION_MAX_DELAY = 3  # exclusive -> {0,1,2}

N_TRIALS = 20
N_STEPS = 150
COMMAND = np.array([0.0, 0.0, 0.0], dtype=np.float32)

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

palm_body_id = model.body("palm").id
floor_geom_id = model.geom("floor").id
tip_geom_ids = [model.geom(n).id for n in TIP_GEOMS]

free_joint_id = next(j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE)
free_qpos_adr = model.jnt_qposadr[free_joint_id]
free_dof_adr = model.jnt_dofadr[free_joint_id]

joint_qpos_adr = np.array([model.jnt_qposadr[model.joint(n).id] for n in ACTUATORS])
joint_dof_adr = np.array([model.jnt_dofadr[model.joint(n).id] for n in ACTUATORS])

if model.nkey > 0:
    home_qpos = model.key_qpos[0].copy()
else:
    home_qpos = model.qpos0.copy()

print("home free-joint pos/quat:", home_qpos[free_qpos_adr:free_qpos_adr+7])

def sensor_adr(name):
    sid = model.sensor(name).id
    return model.sensor_adr[sid]

GYRO_ADR = sensor_adr("imu_gyro")
UPVEC_ADR = sensor_adr("upvector")

sess = ort.InferenceSession(ONNX_PATH)
in_name = sess.get_inputs()[0].name
out_name = sess.get_outputs()[0].name
print("onnx input:", sess.get_inputs()[0].name, sess.get_inputs()[0].shape)
print("onnx output:", sess.get_outputs()[0].name, sess.get_outputs()[0].shape)

def quat_mult(q1, q2):
    w1,x1,y1,z1 = q1
    w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ])

def reset(rng):
    mujoco.mj_resetData(model, data)
    data.qpos[:] = home_qpos
    data.qvel[:] = 0.0

    data.qpos[free_qpos_adr+0] += rng.uniform(-0.05, 0.05)
    data.qpos[free_qpos_adr+1] += rng.uniform(-0.05, 0.05)

    yaw = rng.uniform(-np.pi, np.pi)
    yaw_quat = np.array([np.cos(yaw/2), 0, 0, np.sin(yaw/2)])
    home_quat = home_qpos[free_qpos_adr+3:free_qpos_adr+7]
    data.qpos[free_qpos_adr+3:free_qpos_adr+7] = quat_mult(yaw_quat, home_quat)

    for i, adr in enumerate(joint_qpos_adr):
        data.qpos[adr] = DEFAULT_ACTUATOR[i] * rng.uniform(0.5, 1.5)

    for i in range(6):
        data.qvel[free_dof_adr+i] = rng.uniform(-0.05, 0.05)

    mujoco.mj_forward(model, data)

def get_contacts():
    c = np.zeros(4, dtype=np.float32)
    for i in range(data.ncon):
        con = data.contact[i]
        g1, g2 = con.geom1, con.geom2
        for k, tid in enumerate(tip_geom_ids):
            if (g1 == floor_geom_id and g2 == tid) or (g2 == floor_geom_id and g1 == tid):
                c[k] = 1.0
    return c

def get_obs(info, rng):
    gyro = data.sensordata[GYRO_ADR:GYRO_ADR+3]
    xmat = data.xmat[palm_body_id].reshape(3,3)
    gravity = xmat.T @ np.array([0,0,-1.0])

    joint_pos = data.qpos[joint_qpos_adr] - DEFAULT_ACTUATOR
    joint_vel = data.qvel[joint_dof_adr] * DOF_VEL_SCALE

    noisy_gyro = gyro + (2*rng.uniform(size=3)-1) * N_GYRO * NOISE_LEVEL
    noisy_gravity = gravity + (2*rng.uniform(size=3)-1) * N_GRAVITY * NOISE_LEVEL
    noisy_joint_pos = joint_pos + (2*rng.uniform(size=16)-1) * N_JOINT_POS * NOISE_LEVEL
    noisy_joint_vel = joint_vel + (2*rng.uniform(size=16)-1) * N_JOINT_VEL * DOF_VEL_SCALE * NOISE_LEVEL

    contact = get_contacts()

    obs = np.concatenate([
        noisy_gyro, noisy_gravity, COMMAND,
        noisy_joint_pos, noisy_joint_vel,
        info["last_act"], info["last_last_act"], info["last_last_last_act"],
        info["motor_targets"], contact,
    ]).astype(np.float32)
    return obs

def get_done():
    upvec = data.sensordata[UPVEC_ADR:UPVEC_ADR+3]
    return (upvec[2] < 0) or np.isnan(data.qpos).any() or np.isnan(data.qvel).any()

results = []
for trial in range(N_TRIALS):
    rng = np.random.default_rng(trial)
    reset(rng)

    info = {
        "last_act": np.zeros(16, dtype=np.float32),
        "last_last_act": np.zeros(16, dtype=np.float32),
        "last_last_last_act": np.zeros(16, dtype=np.float32),
        "motor_targets": DEFAULT_ACTUATOR.copy().astype(np.float32),
        "action_history": np.zeros(48, dtype=np.float32),
    }

    term_step = N_STEPS
    upvec_log = []
    palmz_log = []

    for step in range(N_STEPS):
        obs = get_obs(info, rng)
        action = sess.run([out_name], {in_name: obs[None, :]})[0][0]

        info["action_history"] = np.roll(info["action_history"], 16)
        info["action_history"][:16] = action
        delay = rng.integers(ACTION_MIN_DELAY, ACTION_MAX_DELAY)
        action_w_delay = info["action_history"].reshape(-1, 16)[delay]

        target = DEFAULT_ACTUATOR + action_w_delay * ACTION_SCALE
        prev = info["motor_targets"]
        target = np.clip(target, prev - MAX_DELTA, prev + MAX_DELTA)
        info["motor_targets"] = target

        data.ctrl[:] = target
        for _ in range(DECIMATION):
            mujoco.mj_step(model, data)

        upvec = data.sensordata[UPVEC_ADR:UPVEC_ADR+3]
        upvec_log.append(float(upvec[2]))
        palmz_log.append(float(data.qpos[free_qpos_adr+2]))

        info["last_last_last_act"] = info["last_last_act"]
        info["last_last_act"] = info["last_act"]
        info["last_act"] = action

        if get_done():
            term_step = step + 1
            break

    results.append((term_step, upvec_log, palmz_log))
    print(f"trial {trial:2d}: terminated at step {term_step:3d} / {N_STEPS}"
          f"  (upvec_z[0]={upvec_log[0]:.3f}, upvec_z[-1]={upvec_log[-1]:.3f},"
          f" palm_z[0]={palmz_log[0]:.4f}, palm_z[-1]={palmz_log[-1]:.4f})")

term_steps = np.array([r[0] for r in results])
print(f"\nsurvived full {N_STEPS} steps: {(term_steps == N_STEPS).sum()} / {N_TRIALS}")
print(f"mean termination step: {term_steps.mean():.1f}, std: {term_steps.std():.1f}")
