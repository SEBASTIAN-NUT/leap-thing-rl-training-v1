#!/usr/bin/env python3
"""
fk_and_ranges.py - regenerate the FK-curve and joint-range diagnostic
data as real CSV files (csv.writer), replacing the earlier inline
one-liner whose printed output had to be manually retyped. Writes:
  - fk_table_index_finger.csv   (if_mcp sweep -> tip x_rel, z_rel)
  - joint_ranges.csv            (if_mcp/if_rot/if_pip/if_dip ranges)
"""
import csv
import numpy as np
import mujoco

XML = "thing_test/xmls/scene_flat_terrain_cpg.xml"


def main():
    m = mujoco.MjModel.from_xml_path(XML)
    d = mujoco.MjData(m)
    kid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(m, d, kid)
    d.qpos[2] = 0.035
    mujoco.mj_forward(m, d)

    palm_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "palm")
    tip_gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "if_tip_collision")
    mcp_jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "if_mcp")
    mcp_qposadr = m.jnt_qposadr[mcp_jid]
    mcp_range = m.jnt_range[mcp_jid]

    with open("joint_ranges.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["joint", "qposadr", "range_min_rad", "range_max_rad"])
        for jname in ["if_mcp", "if_rot", "if_pip", "if_dip"]:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, jname)
            w.writerow([jname, int(m.jnt_qposadr[jid]),
                        round(float(m.jnt_range[jid][0]), 3),
                        round(float(m.jnt_range[jid][1]), 3)])
    print("Wrote joint_ranges.csv")

    with open("fk_table_index_finger.csv", "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["mcp_rad", "x_rel_m", "z_rel_m"])
        for a in np.linspace(mcp_range[0], mcp_range[1], 15):
            d.qpos[mcp_qposadr] = a
            mujoco.mj_forward(m, d)
            tip = d.geom_xpos[tip_gid]
            palm = d.xpos[palm_bid]
            w.writerow([round(float(a), 3),
                        round(float(tip[0] - palm[0]), 4),
                        round(float(tip[2] - palm[2]), 4)])
    print("Wrote fk_table_index_finger.csv")


if __name__ == "__main__":
    main()
