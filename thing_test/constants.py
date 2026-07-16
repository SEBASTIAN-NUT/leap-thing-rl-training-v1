# Copyright 2025 DeepMind Technologies Limited
# Copyright 2025 Antoine Pirrone - Steve Nguyen
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Constants for LEAP hand Thing environment."""

from etils import epath


ROOT_PATH = epath.Path(__file__).parent
FLAT_TERRAIN_XML = ROOT_PATH / "xmls" / "scene_flat_terrain.xml"


def task_to_xml(task_name: str) -> epath.Path:
    return {
        "flat_terrain": FLAT_TERRAIN_XML,
    }[task_name]


# LEAP hand fingertip geoms used as "feet" for the walking-on-fingers task.
# NOTE: these point at the small sphere-primitive collision geoms (added in
# leap_thing.xml as "*_tip_collision"), not the original mesh tip geoms
# ("if_tip" etc.), which are now visual-only (contype=0/conaffinity=0).
FINGERTIP_GEOMS = [
    "if_tip_collision",
    "mf_tip_collision",
    "rf_tip_collision",
    "th_tip_collision",
]

FEET_GEOMS = FINGERTIP_GEOMS

ROOT_BODY = "palm"

GRAVITY_SENSOR = "upvector"
GLOBAL_LINVEL_SENSOR = "global_linvel"
GLOBAL_ANGVEL_SENSOR = "global_angvel"
LOCAL_LINVEL_SENSOR = "local_linvel"
ACCELEROMETER_SENSOR = "imu_acc"
GYRO_SENSOR = "imu_gyro"
