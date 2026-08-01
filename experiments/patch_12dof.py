#!/usr/bin/env python3
"""12-DOF patch: rot joint を policy から除外
プロジェクトルートから実行: python experiments/patch_12dof.py
"""
import re
from pathlib import Path

# ─────────────── thing_test/thing_walk.py ───────────────
WALK = Path("thing_test/thing_walk.py")
src = WALK.read_text(encoding="utf-8")
orig = src

WALK_PATCHES = [
    # 1. rot joint 定数を USE_MOTOR_SPEED_LIMITS の直後に追加
    (
        "USE_MOTOR_SPEED_LIMITS = True\n",
        """\
USE_MOTOR_SPEED_LIMITS = True

# Rotation joints excluded from policy (ROM too limited for locomotion).
# if_rot=1, mf_rot=5, rf_rot=9, th_axl=13 in the 16-DOF actuator list.
_ROT_JOINT_IDX = [1, 5, 9, 13]
_FREE_JOINT_IDX = [0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15]  # 12 DOF
""",
    ),
    # 2. _post_init: _n_actions を追加
    (
        "        self._actuators = self._mj_model.nu  # 16 (4 joints × 4 fingers)\n",
        """\
        self._actuators = self._mj_model.nu  # 16 (4 joints × 4 fingers, sim ctrl)
        self._rot_joint_idx = jp.array(_ROT_JOINT_IDX)
        self._free_joint_idx = jp.array(_FREE_JOINT_IDX)
        self._n_actions = len(_FREE_JOINT_IDX)  # 12: policy excludes rotation joints
""",
    ),
    # 3. reset() info: last_act 系を n_actions に
    (
        '            "last_act": jp.zeros(self.mjx_model.nu),\n'
        '            "last_last_act": jp.zeros(self.mjx_model.nu),\n'
        '            "last_last_last_act": jp.zeros(self.mjx_model.nu),\n',
        '            "last_act": jp.zeros(self._n_actions),\n'
        '            "last_last_act": jp.zeros(self._n_actions),\n'
        '            "last_last_last_act": jp.zeros(self._n_actions),\n',
    ),
    # 4. action_history サイズ
    (
        "            \"action_history\": jp.zeros(\n"
        "                self._config.noise_config.action_max_delay * self._actuators\n"
        "            ),\n",
        "            \"action_history\": jp.zeros(\n"
        "                self._config.noise_config.action_max_delay * self._n_actions\n"
        "            ),\n",
    ),
    # 5. step() action_history ロール
    (
        "        action_history = (\n"
        "            jp.roll(state.info[\"action_history\"], self._actuators)\n"
        "            .at[: self._actuators]\n"
        "            .set(action)\n"
        "        )\n",
        "        action_history = (\n"
        "            jp.roll(state.info[\"action_history\"], self._n_actions)\n"
        "            .at[: self._n_actions]\n"
        "            .set(action)\n"
        "        )\n",
    ),
    # 6. action_w_delay reshape
    (
        "        action_w_delay = action_history.reshape((-1, self._actuators))[action_idx[0]]\n",
        "        action_w_delay = action_history.reshape((-1, self._n_actions))[action_idx[0]]\n",
    ),
    # 7. motor_targets: 12-dim → 16-dim expand
    (
        "        motor_targets = self._default_actuator + action_w_delay * self._config.action_scale\n",
        "        # Expand 12-dim policy action to 16-dim sim control (rot joints fixed at home)\n"
        "        full_delta = jp.zeros(self._actuators).at[self._free_joint_idx].set(action_w_delay)\n"
        "        motor_targets = self._default_actuator + full_delta * self._config.action_scale\n",
    ),
    # 8. _get_obs: state を 103-dim → 79-dim
    (
        "        state = jp.hstack(\n"
        "            [\n"
        "                info[\"command\"],                                  # 3  [vx, vy, yaw]\n"
        "                noisy_joint_angles - self._default_actuator,     # 16\n"
        "                noisy_joint_vel * self._config.dof_vel_scale,    # 16\n"
        "                info[\"last_act\"],                                 # 16\n"
        "                info[\"last_last_act\"],                            # 16\n"
        "                info[\"last_last_last_act\"],                       # 16\n"
        "                info[\"motor_targets\"],                            # 16\n"
        "                contact,                                          # 4\n"
        "            ]\n"
        "        )  # total: 103\n",
        "        state = jp.hstack(\n"
        "            [\n"
        "                info[\"command\"],                                                                         # 3\n"
        "                noisy_joint_angles[self._free_joint_idx] - self._default_actuator[self._free_joint_idx],  # 12\n"
        "                noisy_joint_vel[self._free_joint_idx] * self._config.dof_vel_scale,                     # 12\n"
        "                info[\"last_act\"],                                                                        # 12\n"
        "                info[\"last_last_act\"],                                                                   # 12\n"
        "                info[\"last_last_last_act\"],                                                              # 12\n"
        "                info[\"motor_targets\"][self._free_joint_idx],                                            # 12\n"
        "                contact,                                                                                # 4\n"
        "            ]\n"
        "        )  # total: 79\n",
    ),
]

applied = 0
for old, new in WALK_PATCHES:
    if old not in src:
        print(f"[SKIP walk] 対象が見つかりません: {repr(old[:60])}")
        continue
    src = src.replace(old, new, 1)
    applied += 1
    print(f"[OK   walk] {repr(old[:60])}")

if src != orig:
    WALK.write_text(src, encoding="utf-8")
    print(f"\n{applied} 件 → {WALK}")
else:
    print("\nwalk: 変更なし (適用済みか)")

# ─────────────── thing_test/check_command_response.py ───────────────
CHECK = Path("thing_test/check_command_response.py")
src = CHECK.read_text(encoding="utf-8")
orig = src

# 1. N_ACTIONS / FREE_JOINT_IDX 定数 (import numpy as np の直後)
old_const = "import numpy as np\n"
new_const = (
    "import numpy as np\n\n"
    "# Rotation joints excluded from policy (if_rot=1, mf_rot=5, rf_rot=9, th_axl=13)\n"
    "N_ACTIONS = 12\n"
    "FREE_JOINT_IDX = np.array([0, 2, 3, 4, 6, 7, 8, 10, 11, 12, 14, 15])\n"
)
if old_const in src and "N_ACTIONS" not in src:
    src = src.replace(old_const, new_const, 1)
    print("[OK   check] N_ACTIONS / FREE_JOINT_IDX 追加")
else:
    print("[SKIP check] 定数 (適用済みか)")

# 2. __init__: last_action を 12-dim に
for old_la, new_la in [
    ("self.last_action = np.zeros(self.model.nu)\n",
     "self.last_action = np.zeros(N_ACTIONS)\n"),
    ("self.last_last_action = np.zeros(self.model.nu)\n",
     "self.last_last_action = np.zeros(N_ACTIONS)\n"),
    ("self.last_last_last_action = np.zeros(self.model.nu)\n",
     "self.last_last_last_action = np.zeros(N_ACTIONS)\n"),
]:
    if old_la in src:
        src = src.replace(old_la, new_la, 1)
        print(f"[OK   check] {repr(old_la[:50])}")
    else:
        print(f"[SKIP check] {repr(old_la[:50])}")

# 3. get_obs() を 79-dim 版に丸ごと置き換え
GET_OBS_NEW = '''\
    def get_obs(self, data, commands) -> np.ndarray:
        # 79-dim actor state (matches thing_walk.py _get_obs state):
        #   command(3) + free_joint_angle(12) + free_joint_vel(12)
        #   + last_act(12) + last_last_act(12) + last_last_last_act(12)
        #   + free_motor_targets(12) + contact(4) = 79
        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_backlash = self.get_actuator_backlash_qpos(data.qpos)
        for i in self.backlash_idx_to_add:
            joint_backlash = np.insert(joint_backlash, i, 0)
        joint_angles = joint_angles + joint_backlash

        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        contacts = self.get_feet_contacts(data)

        free_angles = joint_angles[FREE_JOINT_IDX] - self.default_actuator[FREE_JOINT_IDX]
        free_vel = joint_vel[FREE_JOINT_IDX] * DOF_VEL_SCALE
        free_targets = self.motor_targets[FREE_JOINT_IDX]

        obs = np.concatenate(
            [
                np.array(commands, dtype=np.float64),  # 3
                free_angles,                            # 12
                free_vel,                               # 12
                self.last_action,                       # 12
                self.last_last_action,                  # 12
                self.last_last_last_action,             # 12
                free_targets,                           # 12
                contacts,                               # 4
            ]
        )
        return obs.astype(np.float32)
'''
src_new = re.sub(
    r"    def get_obs\(self, data, commands\).*?return obs\.astype\(np\.float32\)\n",
    GET_OBS_NEW,
    src,
    count=1,
    flags=re.DOTALL,
)
if src_new != src:
    src = src_new
    print("[OK   check] get_obs() を 79-dim に置換")
else:
    print("[SKIP check] get_obs() (適用済みか)")

# 4. run() の motor_targets 計算: action expand
old_mt = "                        self.motor_targets = (\n                            self.default_actuator + action * ACTION_SCALE\n                        )\n"
new_mt = (
    "                        full_delta = np.zeros(self.model.nu)\n"
    "                        full_delta[FREE_JOINT_IDX] = action\n"
    "                        self.motor_targets = self.default_actuator + full_delta * ACTION_SCALE\n"
)
if old_mt in src:
    src = src.replace(old_mt, new_mt, 1)
    print("[OK   check] motor_targets expand")
else:
    print("[SKIP check] motor_targets (適用済みか)")

if src != orig:
    CHECK.write_text(src, encoding="utf-8")
    print(f"\n変更を保存 → {CHECK}")
else:
    print("\ncheck: 変更なし (適用済みか)")

print("\n=== 完了 ===")
