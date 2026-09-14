#!/usr/bin/env python3
import sys
sys.path.insert(0, '/home/elemechrl/Desktop/iizawa/iizawa_workspace/thing_project/Open_Duck_Playground/.venv/lib/python3.11/site-packages')

import mujoco
import mujoco.viewer

XML = "../thing_test/xmls/leap_thing_cpg.xml"

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)

kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(m, d, kid)
d.qpos[2] = 0.035
mujoco.mj_forward(m, d)

print("Press ENTER to close viewer")
with mujoco.viewer.launch_passive(m, d) as viewer:
    input()
