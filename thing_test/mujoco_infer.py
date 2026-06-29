"""Visualize a trained LEAP hand "Thing" policy in the MuJoCo viewer.

Usage:
    python -m thing_test.mujoco_infer -o checkpoints/2026_06_10_202345_1064960.onnx

Controls (click on the viewer window first so it has keyboard focus):
    Up / Down    : forward / backward command (lin_vel_x)
    Left / Right : strafe left / right command (lin_vel_y)
    A / E        : turn left / right command (ang_vel yaw)
    Space        : stop (zero all commands)
    P            : toggle policy ON/OFF (starts OFF -> holds the home pose)
    Ctrl+C in the terminal saves recorded observations to mujoco_saved_obs.pkl
"""

import argparse
import pickle
import time

import mujoco
import mujoco.viewer
import numpy as np

from . import constants
from .common.onnx_infer import OnnxInfer
from .mujoco_infer_base import MJInferBase

# Must match thing_test/thing_walk.py default_config()
USE_MOTOR_SPEED_LIMITS = True
ACTION_SCALE = 0.25
DOF_VEL_SCALE = 0.05
MAX_MOTOR_VELOCITY = 5.24  # rad/s

COMMANDS_RANGE_X = [-0.15, 0.15]
COMMANDS_RANGE_Y = [-0.2, 0.2]
COMMANDS_RANGE_THETA = [-1.0, 1.0]


class MjInfer(MJInferBase):
    def __init__(self, model_path: str, onnx_model_path: str):
        super().__init__(model_path)

        self.policy = OnnxInfer(onnx_model_path, awd=True)

        self.last_action = np.zeros(self.model.nu)
        self.last_last_action = np.zeros(self.model.nu)
        self.last_last_last_action = np.zeros(self.model.nu)
        self.motor_targets = self.default_actuator.copy()
        self.prev_motor_targets = self.default_actuator.copy()

        self.commands = [0.0, 0.0, 0.0]  # vx, vy, yaw

        # Start with the policy disabled so we can verify the home pose is
        # physically stable (PD-held) before letting the network drive it.
        self.policy_enabled = False
        self._action_print_count = 0

        self.saved_obs = []

    def get_obs(self, data, commands) -> np.ndarray:
        gyro = self.get_gyro(data)
        gravity = self.get_gravity(data)

        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_backlash = self.get_actuator_backlash_qpos(data.qpos)
        for i in self.backlash_idx_to_add:
            joint_backlash = np.insert(joint_backlash, i, 0)
        joint_angles = joint_angles + joint_backlash

        joint_vel = self.get_actuator_joints_qvel(data.qvel)

        contacts = self.get_feet_contacts(data)

        obs = np.concatenate(
            [
                gyro,
                gravity,
                np.array(commands, dtype=np.float64),
                joint_angles - self.default_actuator,
                joint_vel * DOF_VEL_SCALE,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                contacts,
            ]
        )
        return obs.astype(np.float32)

    def key_callback(self, keycode):
        lin_vel_x = 0.0
        lin_vel_y = 0.0
        ang_vel = 0.0
        if keycode == 265:  # Up
            lin_vel_x = COMMANDS_RANGE_X[1]
        elif keycode == 264:  # Down
            lin_vel_x = COMMANDS_RANGE_X[0]
        elif keycode == 263:  # Left
            lin_vel_y = COMMANDS_RANGE_Y[1]
        elif keycode == 262:  # Right
            lin_vel_y = COMMANDS_RANGE_Y[0]
        elif keycode == 65:  # A
            ang_vel = COMMANDS_RANGE_THETA[1]
        elif keycode == 69:  # E
            ang_vel = COMMANDS_RANGE_THETA[0]
        elif keycode == 32:  # Space
            pass
        elif keycode == 80:  # P
            self.policy_enabled = not self.policy_enabled
            print(f"policy_enabled: {self.policy_enabled}")
            return
        else:
            return

        self.commands[0] = lin_vel_x
        self.commands[1] = lin_vel_y
        self.commands[2] = ang_vel
        print(f"commands: {self.commands}")

    def run(self):
        try:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
                key_callback=self.key_callback,
            ) as viewer:
                counter = 0
                while viewer.is_running():
                    step_start = time.time()
                    mujoco.mj_step(self.model, self.data)
                    counter += 1

                    if counter % self.decimation == 0:
                        obs = self.get_obs(self.data, self.commands)
                        self.saved_obs.append(obs)

                        if self.policy_enabled:
                            action = np.asarray(self.policy.infer(obs))

                            if self._action_print_count < 5:
                                print(f"action: {action}")
                                self._action_print_count += 1

                            self.last_last_last_action = self.last_last_action.copy()
                            self.last_last_action = self.last_action.copy()
                            self.last_action = action.copy()

                            self.motor_targets = (
                                self.default_actuator + action * ACTION_SCALE
                            )

                            if USE_MOTOR_SPEED_LIMITS:
                                max_delta = (
                                    MAX_MOTOR_VELOCITY * self.sim_dt * self.decimation
                                )
                                self.motor_targets = np.clip(
                                    self.motor_targets,
                                    self.prev_motor_targets - max_delta,
                                    self.prev_motor_targets + max_delta,
                                )
                                self.prev_motor_targets = self.motor_targets.copy()

                            self.data.ctrl[:] = self.motor_targets

                    # Periodic status print (~every 0.5s)
                    if counter % (self.decimation * 25) == 0:
                        palm_z = self.data.qpos[self.floating_base_qpos_addr + 2]
                        print(
                            f"t={self.data.time:5.2f} policy={self.policy_enabled} "
                            f"palm_z={palm_z:+.3f} contacts={self.get_feet_contacts(self.data)} "
                            f"commands={self.commands}"
                        )

                    viewer.sync()

                    time_until_next_step = self.sim_dt - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
        except KeyboardInterrupt:
            pass
        finally:
            if self.saved_obs:
                with open("mujoco_saved_obs.pkl", "wb") as f:
                    pickle.dump(self.saved_obs, f)
                print(f"saved {len(self.saved_obs)} observations to mujoco_saved_obs.pkl")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(constants.task_to_xml("flat_terrain")),
    )
    args = parser.parse_args()

    mji = MjInfer(args.model_path, args.onnx_model_path)
    mji.run()
