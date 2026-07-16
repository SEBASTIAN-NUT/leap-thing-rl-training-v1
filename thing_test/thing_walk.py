"""Joystick task for LEAP hand Thing -- walks on fingertips like Thing from Wednesday."""

from typing import Any, Dict, Optional, Union
import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import mjx_env
# NOTE: contact is detected via native MuJoCo <contact> sensors
# ("*_floor_found" in leap_thing.xml), matching mjx_playground's current
# convention (e.g. Go1), instead of importing geoms_colliding from
# mujoco_playground._src.collision -- that module was removed upstream
# between playground v0.0.5 and v0.2.0, and the manual Python-level
# collision query was also a likely contributor to pathological JIT trace
# times under this jax/mjx version.

from . import constants
from . import base


USE_MOTOR_SPEED_LIMITS = True


def default_config() -> config_dict.ConfigDict:
    return config_dict.create(
        ctrl_dt=0.02,
        sim_dt=0.002,
        episode_length=1000,
        action_repeat=1,
        action_scale=0.25,
        dof_vel_scale=0.05,
        history_len=0,
        soft_joint_pos_limit_factor=0.95,
        max_motor_velocity=5.24,  # rad/s
        noise_config=config_dict.create(
            level=1.0,  # Set to 0.0 to disable noise.
            action_min_delay=0,  # env steps
            action_max_delay=3,  # env steps
            scales=config_dict.create(
                joint_pos=0.03,
                joint_vel=1.5,
                gravity=0.05,
                linvel=0.1,
                gyro=0.2,
                accelerometer=0.05,
            ),
        ),
        reward_config=config_dict.create(
            scales=config_dict.create(
                tracking_lin_vel=1.5,
                tracking_ang_vel=0.5,
                alive=1.0,
                lin_vel_z=0.0,
                ang_vel_xy=0.0,
                orientation=0.0,
                dof_pos_limits=0.0,
                pose=0.0,
                termination=0.0,
                stand_still=0.0,
                torques=0.0,
                action_rate=0.0,
                energy=0.0,
                feet_air_time=0.0,
            ),
            tracking_sigma_lin=0.01,   # sweep target; lin range ±0.15 m/s
            tracking_sigma_ang=0.25,   # fixed; ang range ±1.0 rad/s
        ),
        push_config=config_dict.create(
            enable=True,
            interval_range=[5.0, 10.0],
            magnitude_range=[0.1, 1.0],
        ),
        command_config=config_dict.create(
            a=[1.5, 0.8, 1.2],
            b=[0.9, 0.25, 0.5],
        ),
    )


class Joystick(base.OpenDuckMiniV2Env):
    """Track a joystick command using LEAP hand fingertips as feet (Thing-style walking)."""

    def __init__(
        self,
        task: str = "flat_terrain",
        config: config_dict.ConfigDict = default_config(),
        config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
    ):
        super().__init__(
            xml_path=constants.task_to_xml(task).as_posix(),
            config=config,
            config_overrides=config_overrides,
        )
        self._post_init()

    def _post_init(self) -> None:
        self._init_q = jp.array(self._mj_model.keyframe("home").qpos)
        # Bug fix: keyframe ctrl is often unset (defaults to zeros).
        # Derive the home joint positions directly from qpos instead.
        self._default_actuator = self.get_actuator_joints_qpos(self._init_q)

        # Soft joint position limits (fraction of full range)
        self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
        c = (self._lowers + self._uppers) / 2
        r = self._uppers - self._lowers
        self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
        self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

        self._actuators = self._mj_model.nu  # 16 (4 joints × 4 fingers)

        # Geom IDs: floor and the 4 fingertips used as feet (position lookup,
        # e.g. swing_peak height tracking -- not used for contact detection).
        self._floor_geom_id = self._mj_model.geom("floor").id
        self._feet_geom_id = np.array([
            self._mj_model.geom(name).id
            for name in constants.FINGERTIP_GEOMS  # if_tip, mf_tip, rf_tip, th_tip
        ])

        # Native <contact> sensor ids ("*_floor_found" in leap_thing.xml) used
        # for contact detection -- see module docstring note near the imports.
        self._feet_floor_found_sensor = [
            self._mj_model.sensor(f"{geom}_floor_found").id
            for geom in constants.FINGERTIP_GEOMS
        ]

        # Uniform position noise scale across all 16 joints
        self._qpos_noise_scale = jp.full(
            self._actuators, self._config.noise_config.scales.joint_pos
        )

        # Body ID of the palm (= root body / IMU location)
        self._palm_body_id = self._mj_model.body("palm").id

        self._cmd_a = jp.array(self._config.command_config.a)
        self._cmd_b = jp.array(self._config.command_config.b)

    def reset(self, rng: jax.Array) -> mjx_env.State:
        qpos = self._init_q
        qvel = jp.zeros(self.mjx_model.nv)

        # Randomize XY position (±5 cm) and yaw
        rng, key = jax.random.split(rng)
        dxy = jax.random.uniform(key, (2,), minval=-0.05, maxval=0.05)

        base_qpos = self.get_floating_base_qpos(qpos)
        base_qpos = base_qpos.at[0:2].set(
            qpos[self._floating_base_qpos_addr : self._floating_base_qpos_addr + 2]
            + dxy
        )

        rng, key = jax.random.split(rng)
        yaw = jax.random.uniform(key, (), minval=-3.14, maxval=3.14)
        # Yaw-only rotation about +z, built directly instead of via
        # math.axis_angle_to_quat (which uses jp.insert internally -- a
        # pattern that showed pathological tracing behavior elsewhere in
        # this codebase under jax.lax.scan; avoided here as a precaution,
        # though not confirmed as a bottleneck for this specific call).
        half_yaw = 0.5 * yaw
        quat = jp.array([jp.cos(half_yaw), 0.0, 0.0, jp.sin(half_yaw)])
        new_quat = math.quat_mul(
            qpos[self._floating_base_qpos_addr + 3 : self._floating_base_qpos_addr + 7],
            quat,
        )
        base_qpos = base_qpos.at[3:7].set(new_quat)
        qpos = self.set_floating_base_qpos(base_qpos, qpos)

        # Randomize joint positions (×U(0.5, 1.5) of home pose)
        rng, key = jax.random.split(rng)
        qpos_j = self.get_actuator_joints_qpos(qpos) * jax.random.uniform(
            key, (self._actuators,), minval=0.5, maxval=1.5
        )
        qpos = self.set_actuator_joints_qpos(qpos_j, qpos)

        # Randomize base velocity (±5 cm/s, ±5 rad/s)
        rng, key = jax.random.split(rng)
        qvel = self.set_floating_base_qvel(
            jax.random.uniform(key, (6,), minval=-0.05, maxval=0.05), qvel
        )

        ctrl = self.get_actuator_joints_qpos(qpos)
        # playground v0.2.0 renamed mjx_env.init -> mjx_env.make_data, which
        # no longer calls mjx.forward internally (the caller must do so).
        data = mjx_env.make_data(self.mj_model, qpos=qpos, qvel=qvel, ctrl=ctrl)
        data = mjx.forward(self.mjx_model, data)

        rng, cmd_rng = jax.random.split(rng)
        cmd = jax.random.uniform(cmd_rng, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a)

        # Sample random push interval
        rng, push_rng = jax.random.split(rng)
        push_interval = jax.random.uniform(
            push_rng,
            minval=self._config.push_config.interval_range[0],
            maxval=self._config.push_config.interval_range[1],
        )
        push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

        info = {
            "rng": rng,
            # step/push_step must be jax arrays, not plain Python ints: a
            # bare int here is a JIT-time STATIC value, and a different
            # static value (e.g. as "step" advances run-to-run) forces a
            # full retrace/recompile every call instead of hitting jax's
            # compilation cache -- confirmed as the cause of brax's
            # training loop recompiling on every one of its ~140
            # iterations (each call carried forward a different concrete
            # "step" value from the previous call's returned state).
            "step": jp.array(0),
            "command": cmd,                                                            # [vx, vy, yaw]
            "last_act": jp.zeros(self.mjx_model.nu),
            "last_last_act": jp.zeros(self.mjx_model.nu),
            "last_last_last_act": jp.zeros(self.mjx_model.nu),
            "motor_targets": self._default_actuator,
            "feet_air_time": jp.zeros(4),                                             # one per fingertip
            "last_contact": jp.zeros(4, dtype=bool),
            "swing_peak": jp.zeros(4),                                                # peak fingertip height per swing
            "push": jp.array([0.0, 0.0]),
            "push_step": jp.array(0),
            "push_interval_steps": push_interval_steps,
            "action_history": jp.zeros(
                self._config.noise_config.action_max_delay * self._actuators
            ),
        }

        metrics = {}
        for k, v in self._config.reward_config.scales.items():
            if v != 0:
                metrics[f"reward/{k}" if v > 0 else f"cost/{k}"] = jp.zeros(())
        metrics["swing_peak"] = jp.zeros(())

        contact = jp.array([
            data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
            for sensorid in self._feet_floor_found_sensor
        ])
        obs = self._get_obs(data, info, contact)
        reward, done = jp.zeros(2)
        return mjx_env.State(data, obs, reward, done, metrics, info)

    def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
        state.info["rng"], push1_rng, push2_rng, action_delay_rng = jax.random.split(
            state.info["rng"], 4
        )

        # --- Action delay ---
        action_history = (
            jp.roll(state.info["action_history"], self._actuators)
            .at[: self._actuators]
            .set(action)
        )
        state.info["action_history"] = action_history
        action_idx = jax.random.randint(
            action_delay_rng,
            (1,),
            minval=self._config.noise_config.action_min_delay,
            maxval=self._config.noise_config.action_max_delay,
        )
        action_w_delay = action_history.reshape((-1, self._actuators))[action_idx[0]]

        # --- Random push disturbance ---
        push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
        push_magnitude = jax.random.uniform(
            push2_rng,
            minval=self._config.push_config.magnitude_range[0],
            maxval=self._config.push_config.magnitude_range[1],
        )
        push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
        push *= (
            jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"]) == 0
        )
        push *= self._config.push_config.enable

        qvel = state.data.qvel
        qvel = qvel.at[
            self._floating_base_qvel_addr : self._floating_base_qvel_addr + 2
        ].set(
            push * push_magnitude
            + qvel[self._floating_base_qvel_addr : self._floating_base_qvel_addr + 2]
        )
        data = state.data.replace(qvel=qvel)
        state = state.replace(data=data)

        # --- Motor targets with optional velocity clipping ---
        motor_targets = self._default_actuator + action_w_delay * self._config.action_scale

        if USE_MOTOR_SPEED_LIMITS:
            prev_motor_targets = state.info["motor_targets"]
            motor_targets = jp.clip(
                motor_targets,
                prev_motor_targets - self._config.max_motor_velocity * self.dt,
                prev_motor_targets + self._config.max_motor_velocity * self.dt,
            )

        data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)
        state.info["motor_targets"] = motor_targets

        # --- Contact detection (4 fingertips) ---
        contact = jp.array([
            data.sensordata[self._mj_model.sensor_adr[sensorid]] > 0
            for sensorid in self._feet_floor_found_sensor
        ])
        contact_filt = contact | state.info["last_contact"]
        first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
        state.info["feet_air_time"] += self.dt

        # Track peak fingertip height during swing (via geom z-position)
        p_fz = jp.array([data.geom_xpos[geom_id][2] for geom_id in self._feet_geom_id])
        state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

        obs = self._get_obs(data, state.info, contact)
        done = self._get_termination(data)

        rewards = self._get_reward(
            data, action, state.info, state.metrics, done, first_contact, contact
        )
        rewards = {
            k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
        }
        reward = jp.clip(sum(rewards.values()) * self.dt, 0.0, 10000.0)

        state.info["push"] = push
        state.info["step"] += 1
        state.info["push_step"] += 1
        state.info["last_last_last_act"] = state.info["last_last_act"]
        state.info["last_last_act"] = state.info["last_act"]
        state.info["last_act"] = action
        state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])

        # Re-sample command every 500 steps
        state.info["command"] = jp.where(
            state.info["step"] > 500,
            self.sample_command(cmd_rng, state.info["command"]),
            state.info["command"],
        )
        state.info["step"] = jp.where(
            done | (state.info["step"] > 500), 0, state.info["step"]
        )
        state.info["feet_air_time"] *= ~contact
        state.info["last_contact"] = contact
        state.info["swing_peak"] *= ~contact

        for k, v in rewards.items():
            rew_scale = self._config.reward_config.scales[k]
            if rew_scale != 0:
                state.metrics[f"reward/{k}" if rew_scale > 0 else f"cost/{k}"] = (
                    v if rew_scale > 0 else -v
                )
        state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

        done = done.astype(reward.dtype)
        state = state.replace(data=data, obs=obs, reward=reward, done=done)
        return state

    def _get_termination(self, data: mjx.Data) -> jax.Array:
        # Terminate if palm faces downward (fallen over) or NaN detected
        fall_termination = self.get_gravity(data)[-1] < 0.0
        return fall_termination | jp.isnan(data.qpos).any() | jp.isnan(data.qvel).any()

    def _get_obs(
        self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
    ) -> mjx_env.Observation:

        # --- Sensor readings ---
        # gyro/accelerometer/gravity are IMU-derived.
        # Phase-1: noisy IMU included in actor obs to improve learning success.
        # Phase-2: remove IMU from state to match real hardware (LEAP Hand has no IMU).
        # Clean versions always feed privileged_state.
        gyro = self.get_gyro(data)
        accelerometer = self.get_accelerometer(data)
        # Gravity direction in palm-local frame
        gravity = data.xmat[self._palm_body_id].reshape(3, 3).T @ jp.array([0, 0, -1])

        # --- Joint state ---
        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        # Re-expand backlash readings to one value per actuator (0 where the
        # actuator has no backlash joint -- true for every actuator on the
        # LEAP hand). Built with a single .at[].set() instead of a Python
        # loop of 16 sequential jp.insert calls: that chain of
        # shape-growing inserts was a likely contributor to the
        # pathological jax.lax.scan trace time (scan's carry-consistency
        # check does not play well with chains of dynamic-shape ops).
        joint_backlash_raw = self.get_actuator_backlash_qpos(data.qpos)
        joint_backlash = jp.zeros(self._actuators)
        backlash_positions = [
            i for i in range(self._actuators) if i not in self.backlash_idx_to_add
        ]
        if backlash_positions:
            joint_backlash = joint_backlash.at[jp.array(backlash_positions)].set(
                joint_backlash_raw
            )
        joint_angles = joint_angles + joint_backlash

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_angles = (
            joint_angles
            + (2.0 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1.0)
            * self._config.noise_config.level
            * self._qpos_noise_scale
        )

        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_joint_vel = (
            joint_vel
            + (2.0 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1.0)
            * self._config.noise_config.level
            * self._config.noise_config.scales.joint_vel
        )

        linvel = self.get_local_linvel(data)

        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gyro = (
            gyro
            + (2.0 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1.0)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gyro
        )
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_gravity = (
            gravity
            + (2.0 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1.0)
            * self._config.noise_config.level
            * self._config.noise_config.scales.gravity
        )
        info["rng"], noise_rng = jax.random.split(info["rng"])
        noisy_linvel = (
            linvel
            + (2.0 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1.0)
            * self._config.noise_config.level
            * self._config.noise_config.scales.linvel
        )

        # --- Policy observation (112 dims, phase-1: IMU included) ---
        state = jp.hstack(
            [
                info["command"],                                  # 3  [vx, vy, yaw]
                noisy_joint_angles - self._default_actuator,     # 16
                noisy_joint_vel * self._config.dof_vel_scale,    # 16
                info["last_act"],                                 # 16
                info["last_last_act"],                            # 16
                info["last_last_last_act"],                       # 16
                info["motor_targets"],                            # 16
                contact,                                          # 4
                noisy_gyro,                                       # 3
                noisy_gravity,                                    # 3
                noisy_linvel,                                     # 3
            ]
        )  # total: 112

        # --- Privileged observation (teacher; includes ground-truth sensor data) ---
        global_angvel = self.get_global_angvel(data)
        root_height = data.qpos[self._floating_base_qpos_addr + 2]  # palm z-height

        privileged_state = jp.hstack(
            [
                state,
                gyro,                                              # 3 (clean; state has noisy version)
                accelerometer,                                    # 3
                gravity,                                          # 3
                linvel,                                           # 3
                global_angvel,                                    # 3
                joint_angles - self._default_actuator,           # 16
                joint_vel,                                        # 16
                root_height,                                      # 1
                data.actuator_force,                              # 16
                contact,                                          # 4
                info["feet_air_time"],                            # 4
            ]
        )

        return {
            "state": state,
            "privileged_state": privileged_state,
        }

    def _get_reward(
        self,
        data: mjx.Data,
        action: jax.Array,
        info: dict[str, Any],
        metrics: dict[str, Any],
        done: jax.Array,
        first_contact: jax.Array,
        contact: jax.Array,
    ) -> dict[str, jax.Array]:
        del metrics  # Unused.

        command = info["command"]
        local_vel = self.get_local_linvel(data)
        gyro = self.get_gyro(data)
        global_linvel = self.get_global_linvel(data)
        global_angvel = self.get_global_angvel(data)
        upvector = self.get_gravity(data)
        sigma_lin = self._config.reward_config.tracking_sigma_lin
        sigma_ang = self._config.reward_config.tracking_sigma_ang

        lin_vel_error = jp.sum(jp.square(command[:2] - local_vel[:2]))
        tracking_lin = jp.exp(-lin_vel_error / sigma_lin)
        ang_vel_error = jp.square(command[2] - gyro[2])
        tracking_ang = jp.exp(-ang_vel_error / sigma_ang)

        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        cmd_norm = jp.linalg.norm(command)
        out_of_limits = -jp.clip(joint_angles - self._soft_lowers, None, 0.0)
        out_of_limits += jp.clip(joint_angles - self._soft_uppers, 0.0, None)
        air_time_rew = jp.sum((info["feet_air_time"] - 0.1) * first_contact) * (cmd_norm > 0.01)

        return {
            "tracking_lin_vel": tracking_lin,
            "tracking_ang_vel": tracking_ang,
            "lin_vel_z": jp.square(global_linvel[2]),
            "ang_vel_xy": jp.sum(jp.square(global_angvel[:2])),
            "orientation": jp.sum(jp.square(upvector[:2])),
            "pose": jp.exp(-jp.sum(jp.square(joint_angles - self._default_actuator))),
            "dof_pos_limits": jp.sum(out_of_limits),
            "stand_still": jp.sum(jp.abs(joint_angles - self._default_actuator)) * (cmd_norm < 0.01),
            "termination": done,
            "torques": jp.sqrt(jp.sum(jp.square(data.actuator_force))) + jp.sum(jp.abs(data.actuator_force)),
            "action_rate": jp.sum(jp.square(action - info["last_act"])),
            "energy": jp.sum(jp.abs(joint_vel) * jp.abs(data.actuator_force)),
            "feet_air_time": air_time_rew,
            "alive": 1.0 - done,
        }

    def sample_command(self, rng: jax.Array, x_k: jax.Array) -> jax.Array:
        rng, y_rng, w_rng, z_rng = jax.random.split(rng, 4)
        y_k = jax.random.uniform(y_rng, shape=(3,), minval=-self._cmd_a, maxval=self._cmd_a)
        z_k = jax.random.bernoulli(z_rng, self._cmd_b, shape=(3,))
        w_k = jax.random.bernoulli(w_rng, 0.5, shape=(3,))
        return x_k - w_k * (x_k - y_k * z_k)
