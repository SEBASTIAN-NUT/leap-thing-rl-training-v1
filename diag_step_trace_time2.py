import time
import jax
import jax.numpy as jp
from thing_test import thing_walk

print("Building env...")
env = thing_walk.Joystick(task="flat_terrain")

num_envs = 16
unroll_length = 20

key = jax.random.PRNGKey(0)
keys = jax.random.split(key, num_envs)

def time_it(label, fn, *args):
    print(f"\n[{label}] tracing+compiling...")
    t0 = time.time()
    jitted = jax.jit(fn)
    result = jitted(*args)
    jax.block_until_ready(result)
    print(f"[{label}] done in {time.time()-t0:.1f}s")

# 1. vmap over envs (batched reset+step), no scan
batched_reset = jax.vmap(env.reset)
states = batched_reset(keys)
actions = jp.zeros((num_envs, env.action_size))
time_it("vmap(step) batch=16", jax.vmap(env.step), states, actions)

# 2. vmap(step) wrapped in lax.scan for unroll_length steps (single env)
state1 = env.reset(key)
action1 = jp.zeros(env.action_size)
def scan_body(state, _):
    return env.step(state, action1), None
def run_scan(state):
    return jax.lax.scan(scan_body, state, (), length=unroll_length)
time_it(f"scan(step) x{unroll_length}, single env", run_scan, state1)

# 3. vmap + scan combined (matches brax's actual nesting, minus pmap)
def scan_body_batched(states, _):
    return jax.vmap(env.step)(states, actions), None
def run_vmap_scan(states):
    return jax.lax.scan(scan_body_batched, states, (), length=unroll_length)
time_it(f"vmap(batch=16)+scan(x{unroll_length})", run_vmap_scan, states)
