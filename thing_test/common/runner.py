"""Runs training and evaluation loop for LEAP hand."""

import argparse
import functools

from playground.common import randomize
from playground.common.runner import BaseRunner
from thing_test import thing_walk
from mujoco_playground.config import locomotion_params
from brax.training.agents.ppo import networks as ppo_networks, train as ppo
from mujoco_playground import wrapper


class LeapThingRunner(BaseRunner):

    def __init__(self, args):
        super().__init__(args)
        available_envs = {
            "joystick": (thing_walk, thing_walk.Joystick),
        }
        if args.env not in available_envs:
            raise ValueError(f"Unknown env {args.env}")

        self.env_file = available_envs[args.env]

        self.env_config = self.env_file[0].default_config()
        self.env = self.env_file[1](task=args.task)
        self.eval_env = self.env_file[1](task=args.task)
        self.randomizer = randomize.domain_randomize
        self.action_size = self.env.action_size
        self.obs_size = int(
            self.env.observation_size["state"][0]
        )
        self.restore_checkpoint_path = args.restore_checkpoint_path
        print(f"Observation size: {self.obs_size}")

    def train(self) -> None:
        """Override train() to set num_envs=256 and skip ONNX export."""
        self.ppo_params = locomotion_params.brax_ppo_config(
            "BerkeleyHumanoidJoystickFlatTerrain"
        )
        self.ppo_training_params = dict(self.ppo_params)
        
        # 修正: num_envs を 256 に設定
        self.ppo_training_params["num_envs"] = 256

        if "network_factory" in self.ppo_params:
            network_factory = functools.partial(
                ppo_networks.make_ppo_networks, **self.ppo_params.network_factory
            )
            del self.ppo_training_params["network_factory"]
        else:
            network_factory = ppo_networks.make_ppo_networks
        
        self.ppo_training_params["num_timesteps"] = self.num_timesteps
        print(f"PPO params (num_envs=256): {self.ppo_training_params}")

        train_fn = functools.partial(
            ppo.train,
            **self.ppo_training_params,
            network_factory=network_factory,
            randomization_fn=self.randomizer,
            progress_fn=self.progress_callback,
            policy_params_fn=None,  # 修正: ONNX export をスキップ
            restore_checkpoint_path=self.restore_checkpoint_path,
        )

        _, params, _ = train_fn(
            environment=self.env,
            eval_env=self.eval_env,
            wrap_env_fn=wrapper.wrap_for_brax_training,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="LEAP hand Runner Script")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    parser.add_argument("--num_timesteps", type=int, default=150000000)
    parser.add_argument("--env", type=str, default="joystick")
    parser.add_argument("--task", type=str, default="flat_terrain")
    parser.add_argument("--restore_checkpoint_path", type=str, default=None)
    args = parser.parse_args()

    runner = LeapThingRunner(args)
    runner.train()


if __name__ == "__main__":
    main()
