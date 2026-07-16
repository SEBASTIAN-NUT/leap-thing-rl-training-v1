"""Numpy-based helpers for running a trained LEAP hand "Thing" policy in the
MuJoCo interactive viewer (no JAX / MJX dependency at runtime).

This mirrors the joint/actuator bookkeeping done in `thing_test/base.py`
(OpenDuckMiniV2Env) so that observations built here line up with the ones
produced by `thing_walk.Joystick._get_obs` during training.
"""

from etils import epath
import mujoco
import numpy as np

from . import base
from . import constants


class MJInferBase:
    def __init__(self, model_path: str):
        self.model = mujoco.MjModel.from_xml_string(
            epath.Path(model_path).read_text(), assets=base.get_assets()
        )

        self.sim_dt = 0.002
        self.decimation = 10  # ctrl_dt(0.02) / sim_dt(0.002)
        self.model.opt.timestep = self.sim_dt

        self.data = mujoco.MjData(self.model)
        home_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, home_id)

        # --- joint / actuator bookkeeping (mirrors thing_test/base.py) ---
        self.floating_base_name = [
            self.model.jnt(k).name
            for k in range(self.model.njnt)
            if self.model.jnt(k).type == 0
        ][0]

        self.floating_base_qpos_addr = int(
            self.model.jnt_qposadr[
                self.get_joint_id_from_name(self.floating_base_name)
            ]
        )

        self.actuator_names = [
            self.model.jnt(self.model.actuator_trnid[k, 0]).name
            for k in range(self.model.nu)
        ]

        self.joint_names = [self.model.jnt(k).name for k in range(self.model.njnt)]

        self.backlash_joint_names = [
            j
            for j in self.joint_names
            if j not in self.actuator_names and j != self.floating_base_name
        ]

        self.actuator_joint_ids = [
            self.get_joint_id_from_name(n) for n in self.actuator_names
        ]
        self.actuator_joint_qpos_addr = np.array(
            [self.model.jnt_qposadr[idx] for idx in self.actuator_joint_ids]
        )
        self.actuator_qvel_addr = np.array(
            [self.model.jnt_dofadr[idx] for idx in self.actuator_joint_ids]
        )

        self.backlash_joint_ids = [
            self.get_joint_id_from_name(n) for n in self.backlash_joint_names
        ]
        self.backlash_joint_qpos_addr = np.array(
            [self.model.jnt_qposadr[idx] for idx in self.backlash_joint_ids],
            dtype=int,
        )

        self.backlash_idx_to_add = []
        for i, actuator_name in enumerate(self.actuator_names):
            if actuator_name + "_backlash" not in self.backlash_joint_names:
                self.backlash_idx_to_add.append(i)

        # --- default ("home") pose, derived from qpos (NOT keyframe ctrl) ---
        self.default_actuator = self.get_actuator_joints_qpos(
            np.array(self.model.keyframe("home").qpos)
        )

        # Make sure ctrl matches the home pose from t=0 (keyframe ctrl is 0).
        self.data.ctrl[:] = self.default_actuator

        # --- body used for the gravity observation ---
        self.palm_body_id = self.model.body(constants.ROOT_BODY).id

        print(f"actuators ({self.model.nu}): {self.actuator_names}")
        print(f"backlash joints: {self.backlash_joint_names}")
        print(f"default_actuator (home qpos): {self.default_actuator}")
        print(f"home palm z: {self.model.keyframe('home').qpos[self.floating_base_qpos_addr + 2]}")

    # --- name -> id helpers ---
    def get_joint_id_from_name(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)

    # --- joint state accessors ---
    def get_actuator_joints_qpos(self, qpos: np.ndarray) -> np.ndarray:
        return qpos[self.actuator_joint_qpos_addr]

    def get_actuator_joints_qvel(self, qvel: np.ndarray) -> np.ndarray:
        return qvel[self.actuator_qvel_addr]

    def get_actuator_backlash_qpos(self, qpos: np.ndarray) -> np.ndarray:
        if len(self.backlash_joint_qpos_addr) == 0:
            return np.array([])
        return qpos[self.backlash_joint_qpos_addr]

    # --- sensors ---
    def get_sensor(self, data, name: str) -> np.ndarray:
        return data.sensor(name).data.copy()

    def get_gyro(self, data) -> np.ndarray:
        return self.get_sensor(data, constants.GYRO_SENSOR)

    def get_accelerometer(self, data) -> np.ndarray:
        return self.get_sensor(data, constants.ACCELEROMETER_SENSOR)

    def get_gravity(self, data) -> np.ndarray:
        """Gravity direction in the palm-local frame (matches thing_walk._get_obs)."""
        xmat = data.xmat[self.palm_body_id].reshape(3, 3)
        return xmat.T @ np.array([0.0, 0.0, -1.0])

    # --- contacts ---
    def check_contact(self, data, geom1_name: str, geom2_name: str) -> bool:
        geom1_id = self.model.geom(geom1_name).id
        geom2_id = self.model.geom(geom2_name).id
        for i in range(data.ncon):
            c = data.contact[i]
            if (c.geom1 == geom1_id and c.geom2 == geom2_id) or (
                c.geom1 == geom2_id and c.geom2 == geom1_id
            ):
                return True
        return False

    def get_feet_contacts(self, data) -> np.ndarray:
        """Contact state of the 4 fingertips (if/mf/rf/th) against the floor."""
        return np.array(
            [self.check_contact(data, name, "floor") for name in constants.FEET_GEOMS],
            dtype=np.float32,
        )
