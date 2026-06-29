"""Fast bisection probe: does THIS installed mujoco-mjx work under jax.pmap
the way brax's training loop calls it (env reset batched via pmap -> vmap)?
"""

import jax
import mujoco
from mujoco import mjx

print("jax:", jax.__version__, "| mujoco:", mujoco.__version__)
print("devices:", jax.devices())

xml = (
    "<mujoco><worldbody><body><freejoint/>"
    "<geom size='0.1'/></body></worldbody></mujoco>"
)
model = mujoco.MjModel.from_xml_string(xml)
mx = mjx.put_model(model)

print("\n[1] does mjx.Data have `_impl`?")
d0 = mjx.make_data(mx)
print("    _impl present:", hasattr(d0, "_impl"))

print("\n[2] mjx.make_data under pmap(vmap(...)) (brax's reset nesting)")
try:
    def reset(_):
        return mjx.make_data(mx)

    per_device_envs = 4
    dummy = jax.numpy.zeros((jax.local_device_count(), per_device_envs))
    out = jax.pmap(jax.vmap(reset))(dummy)
    print("    OK -> survives pmap(vmap(make_data))")
except Exception as e:
    print("    FAILED ->", repr(e))
