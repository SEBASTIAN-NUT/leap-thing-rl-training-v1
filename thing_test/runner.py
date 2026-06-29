"""Runs training and evaluation loop for LEAP hand."""

import argparse
import functools
import json
import os
from datetime import datetime

# Persistent JAX/XLA compilation cache: without this, every fresh process
# re-traces and re-compiles the whole physics+PPO graph from scratch (this is
# what made every debugging restart this session pay the full ~tens-of-
# minutes-or-more compile cost again). Once warmed up, a cache hit skips
# straight to execution. Must be set before jax initializes (env var, not
# jax.config), so this sits above any jax-importing import.
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR", os.path.expanduser("~/.cache/jax_thing_cache")
)
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1")

# GPU-specific XLA autotuning (searching many candidate GEMM/conv kernel
# implementations per unique op shape) was confirmed as a real cost here --
# the very first OOM crash this session showed "Autotuning failed... Failed
# to profile configs" during compilation. A local CPU-only repro of mjx.step
# (matching body/joint/sensor/solver counts) showed NO scan-related slowdown
# at all (~7-9s regardless of nesting), which rules out jax version and our
# code/XML as the cause -- pointing specifically at GPU-only autotuning
# search cost, which scales badly with the many small/unusual tensor shapes
# this many-body model produces. Disabling it trades some runtime kernel
# efficiency for what should be a large compile-time win. Must be set before
# jax initializes (env var, not jax.config), like the cache dir above.
os.environ.setdefault("XLA_FLAGS", "--xla_gpu_autotune_level=0")

import threading
import time

import jax
import jax.numpy as jp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

# brax 0.14.2 (last-maintained brax release, per upstream) still calls
# jax.device_put_replicated, which jax removed starting at 0.10.0 (raises
# AttributeError -- confirmed by reproducing this exact crash against
# mujoco_playground's own unmodified BerkeleyHumanoidJoystickFlatTerrain
# reference env, not just our model). This is JAX's own documented drop-in
# replacement (see "Migrating from pmap" guide) restoring the same call
# signature so brax's pmap-based train.py keeps working under jax>=0.10.
def _device_put_replicated(x, devices):
    mesh = Mesh(np.array(devices), ("x",))
    sharding = NamedSharding(mesh, P("x"))
    return jax.tree.map(
        lambda y: jax.device_put(jp.stack([y] * len(devices)), sharding), x
    )


jax.device_put_replicated = _device_put_replicated

# brax's train.py unconditionally wraps training_epoch in jax.pmap (even
# for a single GPU). This was confirmed as THE dominant remaining compile
# cost this session: an isolated rollout function (vmap8192 + scan20 over
# the real env.step) compiled in 415s under plain jax.jit, but the exact
# same function compiled for 1+ hour when wrapped in jax.pmap instead --
# "scan-of-pmap"/nested-pmap is a documented JAX performance footgun
# (pmap has its own heavy jit-fused compile path, separate from plain
# jit's). jax.shard_map (the modern pmap replacement) was tried first but
# its stricter manual-axis typing conflicts with mjx's own internal
# solver code (jax.lax.cond branches returning mismatched "varying" axis
# types) -- not something fixable from here. jax.vmap(fn, axis_name=...)
# is the simplest fix: it still satisfies collectives like lax.pmean that
# rely on axis_name, while avoiding both pmap's and shard_map's separate
# overheads. Only takes effect for the single local-device case this
# machine actually has; real multi-device runs still use real pmap.
_orig_pmap = jax.pmap


def _pmap_or_vmap(fun, axis_name=None, *, donate_argnums=(), **kwargs):
    if jax.local_device_count() > 1:
        return _orig_pmap(
            fun, axis_name=axis_name, donate_argnums=donate_argnums, **kwargs
        )
    return jax.jit(jax.vmap(fun, axis_name=axis_name))


jax.pmap = _pmap_or_vmap

from thing_test.common import randomize
from thing_test.common.runner import BaseRunner
from thing_test import thing_walk
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from mujoco_playground import wrapper


class _Heartbeat:
    """Prints a low-noise "still alive" line every `interval` seconds while
    the wrapped block runs. jax.config.update("jax_log_compiles", True) was
    tried first but logs every single jit call (including microsecond
    persistent-cache hits) -- far too noisy to tell progress from a hang.
    This gives one line per interval instead, regardless of what's happening
    underneath (long initial trace, slow autotune, normal training step)."""

    def __init__(self, interval: float = 15.0):
        self.interval = interval
        self._stop = threading.Event()
        self._thread = None
        self._start_time = None

    def _run(self):
        while not self._stop.wait(self.interval):
            elapsed = time.time() - self._start_time
            print(f"[heartbeat] still running... {elapsed:.0f}s elapsed", flush=True)

    def __enter__(self):
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc_info):
        self._stop.set()
        self._thread.join(timeout=1)


class LeapThingRunner(BaseRunner):

    def __init__(self, args):
        super().__init__(args)
        available_envs = {
            "joystick": (thing_walk, thing_walk.Joystick),
        }
        if args.env not in available_envs:
            raise ValueError(f"Unknown env {args.env}")

        self.env_file = available_envs[args.env]

        # Lets a sweep script override e.g. reward scales without editing
        # thing_walk.py, via mjx_env.MjxEnv's update_from_flattened_dict:
        # keys are dot-paths into the ConfigDict, e.g.
        # {"reward_config.scales.alive": 5.0, "reward_config.tracking_sigma": 0.02}
        config_overrides = (
            json.loads(args.config_overrides) if args.config_overrides else None
        )
        self.env_config = self.env_file[0].default_config()
        self.env = self.env_file[1](task=args.task, config_overrides=config_overrides)
        self.eval_env = self.env_file[1](
            task=args.task, config_overrides=config_overrides
        )
        # domain_randomize needs the real floor geom id / palm body id,
        # resolved by name here (where the actual mj_model with names is
        # available) instead of being hardcoded inside randomize.py.
        self.randomizer = functools.partial(
            randomize.domain_randomize,
            floor_geom_id=self.env.mj_model.geom("floor").id,
            torso_body_id=self.env.mj_model.body("palm").id,
        )
        self.action_size = self.env.action_size
        self.obs_size = int(
            self.env.observation_size["state"][0]
        )  # 0: state 1: privileged_state
        self.restore_checkpoint_path = args.restore_checkpoint_path
        print(f"Observation size: {self.obs_size}")

    def progress_callback(self, num_steps: int, metrics: dict) -> None:
        """Override BaseRunner.progress_callback: it hardcodes
        metrics["eval/episode_reward"], which only exists when evals run.
        With run_evals=False, metrics falls back to training_metrics (no
        eval/* keys) and the base implementation would crash with a
        KeyError right as training finishes."""
        for metric_name, metric_value in metrics.items():
            self.writer.add_scalar(metric_name, metric_value, num_steps)
        print("-----------")
        if "eval/episode_reward" in metrics:
            print(
                f'STEP: {num_steps} reward: {metrics["eval/episode_reward"]} '
                f'reward_std: {metrics["eval/episode_reward_std"]}'
            )
        else:
            print(f"STEP: {num_steps} metrics: {metrics}")
        print("-----------")

    def train(self) -> None:
        """Override train() to set num_envs=256 (avoids GPU OOM from the
        mesh-based fingertip collision SAT computation at the default 8192)
        and skip ONNX export."""
        self.ppo_params = locomotion_params.brax_ppo_config(
            "BerkeleyHumanoidJoystickFlatTerrain"
        )
        self.ppo_training_params = dict(self.ppo_params)

        # num_envs=16: reverting all the num_envs/unroll_length experiments
        # (16/256/1024/8192, unroll_length 5/20) -- none of them fixed the
        # underlying compile-time pathology, and it was independently
        # reproduced on mujoco_playground's own unmodified
        # BerkeleyHumanoidJoystickFlatTerrain reference env too, so it isn't
        # specific to this model or these knobs. What WAS real: three actual
        # library bugs (mjx defaulting to a broken "warp" impl when
        # warp-lang isn't installed; mjx.make_data's device_put crashing
        # under brax's shard_map; brax 0.14.2 calling the jax-0.10+-removed
        # device_put_replicated). All three are now patched above/here.
        # Going back to num_envs=16 -- the one config that previously
        # finished end-to-end (~90s to checkpoint+ONNX) -- as the new
        # baseline to test with all three real fixes applied.
        self.ppo_training_params["num_envs"] = 8192
        # Lightest-possible config to get a successful run first.
        # acting.Evaluator.generate_eval_unroll scans for
        # episode_length // action_repeat steps (1000 here) -- 50x longer
        # than the training rollout's unroll_length (20) -- and its
        # jax.jit(donate_argnums=...)-wrapped scan body is pathologically
        # slow to compile under this jax/mjx version (confirmed: this is
        # the dominant cost behind multi-hour compiles). run_evals=False
        # skips brax's Evaluator entirely (verified safe in train.py:786,
        # 848-855 -- metrics just falls back to training_metrics, no crash).
        self.ppo_training_params["run_evals"] = False

        # unroll_length: left at the PPO config default (20) -- shrinking it
        # only ever shrank the compile-time blowup proportionally, it never
        # eliminated it, so it wasn't masking a real fix. Reverted along
        # with num_envs now that three real library bugs are patched
        # instead (see comment above and the device_put_replicated patch
        # near the top of this file).

        if "network_factory" in self.ppo_params:
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks, **self.ppo_params.network_factory
            )
            del self.ppo_training_params["network_factory"]
        else:
            network_factory = ppo_networks.make_ppo_networks

        self.ppo_training_params["num_timesteps"] = self.num_timesteps
        print(f"PPO params: {self.ppo_training_params}")

        train_fn = functools.partial(
            ppo.train,
            **self.ppo_training_params,
            network_factory=network_factory,
            randomization_fn=self.randomizer,
            progress_fn=self.progress_callback,
            # brax 0.14.2 calls policy_params_fn unconditionally (no None
            # check) -- but the real BaseRunner.policy_params_fn (which
            # actually saves orbax checkpoints + ONNX to self.output_dir)
            # already satisfies that; a no-op lambda was used here before by
            # mistake, which silently disabled all checkpoint saving.
            policy_params_fn=self.policy_params_fn,
            restore_checkpoint_path=self.restore_checkpoint_path,
        )

        print(
            "[Train] Compiling environment + PPO graph... "
            "(first run / cache miss can take a long time; "
            "a '[heartbeat]' line every 15s confirms it isn't hung)"
        )
        with _Heartbeat(interval=15.0):
            _, params, _ = train_fn(
                environment=self.env,
                eval_env=self.eval_env,
                wrap_env_fn=wrapper.wrap_for_brax_training,
            )
        print("[Train] Compilation + training finished.")


def main() -> None:
    parser = argparse.ArgumentParser(description="LEAP hand Runner Script")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Where to save the checkpoints",
    )
    parser.add_argument("--num_timesteps", type=int, default=150000000)
    parser.add_argument("--env", type=str, default="joystick", help="env")
    parser.add_argument("--task", type=str, default="flat_terrain", help="Task to run")
    parser.add_argument(
        "--restore_checkpoint_path",
        type=str,
        default=None,
        help="Resume training from this checkpoint",
    )
    parser.add_argument(
        "--config_overrides",
        type=str,
        default=None,
        help=(
            "JSON object of dot-path env config overrides, e.g. "
            '\'{"reward_config.scales.alive": 5.0}\''
        ),
    )
    args = parser.parse_args()

    runner = LeapThingRunner(args)

    start_time = datetime.now()
    print(f"[Train] Start:   {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    runner.train()

    end_time = datetime.now()
    print(f"[Train] End:     {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[Train] Elapsed: {end_time - start_time}")


if __name__ == "__main__":
    main()
