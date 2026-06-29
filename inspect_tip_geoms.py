import mujoco

model = mujoco.MjModel.from_xml_path("thing_test/xmls/scene_flat_terrain.xml")

for name in ["if_tip", "if_tip_collision", "mf_tip_collision", "rf_tip_collision", "th_tip", "th_tip_collision"]:
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
    if gid < 0:
        print(f"{name}: NOT FOUND")
        continue
    print(f"{name}: type={model.geom_type[gid]}, contype={model.geom_contype[gid]}, "
          f"conaffinity={model.geom_conaffinity[gid]}, pos={model.geom_pos[gid]}, "
          f"size={model.geom_size[gid]}")
