import jax
from thing_test import base  # apply the monkeypatch (module import side-effect)
from mujoco import mjx
import mujoco

xml = "<mujoco><worldbody><body><freejoint/><geom size='0.1'/></body></worldbody></mujoco>"
model = mujoco.MjModel.from_xml_string(xml)
mx = mjx.put_model(model)

def reset(_):
    return mjx.make_data(mx)

per_device_envs = 4
dummy = jax.numpy.zeros((jax.local_device_count(), per_device_envs))
try:
    out = jax.pmap(jax.vmap(reset))(dummy)
    print("OK -> patched make_data survives pmap(vmap(...))")
except Exception as e:
    print("FAILED ->", repr(e))
