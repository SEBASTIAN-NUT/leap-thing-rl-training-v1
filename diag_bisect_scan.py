import time
import jax
import jax.numpy as jp
from mujoco_playground._src import mjx_env
from thing_test import thing_walk

print("Building env...")
env = thing_walk.Joystick(task="flat_terrain")
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

# A. scan over RAW mjx physics only (no reward/obs/our wrapper logic at all)
def scan_body_physics_only(data, _):
    new_data = mjx_env.step(env.mjx_model, data, action, env.n_substeps)
    return new_data, None
def run_scan_physics(data):
    return jax.lax.scan(scan_body_physics_only, data, (), length=unroll_length)
time_it(f"scan(raw mjx physics only) x{unroll_length}", run_scan_physics, state.data)

# B. scan over our FULL env.step (for direct comparison, same as before)
def scan_body_full(state, _):
    return env.step(state, action), None
def run_scan_full(state):
    return jax.lax.scan(scan_body_full, state, (), length=unroll_length)
time_it(f"scan(full env.step) x{unroll_length}", run_scan_full, state)
