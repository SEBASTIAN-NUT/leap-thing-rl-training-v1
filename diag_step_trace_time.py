import time
import jax
from thing_test import thing_walk

print("Building env...")
env = thing_walk.Joystick(task="flat_terrain")

key = jax.random.PRNGKey(0)
state = env.reset(key)
action = jax.numpy.zeros(env.action_size)

print("Tracing+compiling a single jax.jit(env.step) call...")
t0 = time.time()
jitted_step = jax.jit(env.step)
result = jitted_step(state, action)
jax.block_until_ready(result)
elapsed = time.time() - t0
print(f"Single step trace+compile time: {elapsed:.1f}s")
