import mujoco

def report(label, path):
    m = mujoco.MjModel.from_xml_path(path)
    d = mujoco.MjData(m)
    key_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(m, d, key_id)
    mujoco.mj_forward(m, d)
    print(f"--- {label} ---")
    print(f"  nq (qpos dim)      : {m.nq}")
    print(f"  nv (qvel/dof dim)  : {m.nv}")
    print(f"  nu (actuators)     : {m.nu}")
    print(f"  njnt (joints)      : {m.njnt}")
    print(f"  nbody (bodies)     : {m.nbody}")
    print(f"  ngeom (geoms)      : {m.ngeom}")
    print(f"  nsensor            : {m.nsensor}")
    print(f"  ncon (active, now) : {d.ncon}")
    print(f"  nefc (constraints) : {d.nefc}")
    print(f"  solver             : {m.opt.solver} (0=PGS,1=CG,2=Newton)")
    print(f"  iterations         : {m.opt.iterations}")
    print(f"  cone               : {m.opt.cone} (0=pyramidal,1=elliptic)")
    print()

report("LEAP hand (thing_test)", "thing_test/xmls/scene_flat_terrain.xml")
report("Open Duck Mini V2", "Open_Duck_Playground/playground/open_duck_mini_v2/xmls/scene_flat_terrain.xml")
