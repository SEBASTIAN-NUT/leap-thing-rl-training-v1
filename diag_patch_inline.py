import jax
import mujoco
from mujoco import mjx
import mujoco.mjx._src.io as _mjx_io

_orig = _mjx_io._resolve_impl_and_device

def patched(impl=None, device=None):
    resolved_impl, resolved_device = _orig(impl, device)
    print("  [patch] called with device=", device, "-> returning device=",
          None if device is None else resolved_device)
    if device is None:
        return resolved_impl, None
    return resolved_impl, resolved_device

_mjx_io._resolve_impl_and_device = patched

xml = "<mujoco><worldbody><body><freejoint/><geom size='0.1'/></body></worldbody></mujoco>"
model = mujoco.MjModel.from_xml_string(xml)
mx = mjx.put_model(model)

def reset(_):
    return mjx.make_data(mx)

dummy = jax.numpy.zeros((jax.local_device_count(), 4))
try:
    out = jax.pmap(jax.vmap(reset))(dummy)
    print("OK")
except Exception as e:
    print("FAILED", repr(e))
