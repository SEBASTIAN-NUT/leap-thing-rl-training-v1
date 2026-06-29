import sys
import types

fake_collision = types.ModuleType("mujoco_playground._src.collision")
fake_collision.geoms_colliding = lambda data, g1, g2: False
sys.modules["mujoco_playground._src.collision"] = fake_collision

import time
import jax
import jax.numpy as jp
from mujoco import mjx
from mujoco_playground._src import mjx_env

# Compat shim: playground 0.2.0 renamed mjx_env.init -> mjx_env.make_data
# (+ requires an explicit mjx.forward call). Duck's vendored joystick.py
# still calls the old name -- patch it back in just for this timing test.
if not hasattr(mjx_env, "init"):
    def _compat_init(model, qpos=None, qvel=None, ctrl=None, **kwargs):
        data = mjx_env.make_data(model, qpos=qpos, qvel=qvel, ctrl=ctrl, **kwargs)
        return mjx.forward(model, data)
    mjx_env.init = _compat_init

from playground.open_duck_mini_v2 import joystick

print("Building duck env...")
env = joystick.Joystick(task="flat_terrain")
key = jax.random.PRNGKey(0)
state = env.reset(key)
action = jp.zeros(env.action_size)
unroll_length = 20

def time_it(label, fn, *args):
    print(f"\n[{label}] tracing+compiling...")
    t0 = time.time()
    jitted = jax.jit(fn)
    result = jitted(*args)
    jax.block_until_ready(result)
    print(f"[{label}] done in {time.time()-t0:.1f}s")

t0 = time.time()
jax.block_until_ready(jax.jit(env.step)(state, action))
print(f"[bare step, duck] done in {time.time()-t0:.1f}s")

def scan_body_physics_only(data, _):
    new_data = mjx_env.step(env.mjx_model, data, action, env.n_substeps)
    return new_data, None
def run_scan_physics(data):
    return jax.lax.scan(scan_body_physics_only, data, (), length=unroll_length)
time_it(f"scan(raw mjx physics only) x{unroll_length}, duck", run_scan_physics, state.data)
