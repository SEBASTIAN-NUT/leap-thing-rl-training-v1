# Run config

- Started: 2026-07-07 19:03:00
- Git commit: `14eec46acfba8ffcb9288cb5ed5f7efc89a747c1` (uncommitted changes present)

## CLI args

- `output_dir`: checkpoints/p2_siggang_0p5_20260707_190255
- `num_timesteps`: 45875200
- `num_envs`: None
- `env`: joystick
- `task`: flat_terrain
- `restore_checkpoint_path`: None
- `config_overrides`: {"reward_config.tracking_sigma_lin": 0.025, "reward_config.tracking_sigma_ang": 0.5}

## PPO training params

- `action_repeat`: 1
- `batch_size`: 256
- `clipping_epsilon`: 0.2
- `discounting`: 0.97
- `entropy_cost`: 0.01
- `episode_length`: 1000
- `learning_rate`: 0.0003
- `max_grad_norm`: 1.0
- `normalize_observations`: True
- `num_envs`: 8192
- `num_evals`: 15
- `num_minibatches`: 32
- `num_resets_per_eval`: 10
- `num_timesteps`: 45875200
- `num_updates_per_batch`: 4
- `reward_scaling`: 1.0
- `unroll_length`: 20
- `run_evals`: True

## Observation space

- Actor obs dim : 112
- Privileged obs dim: 184
- IMU in actor  : yes (phase-1)
- Actor components: command(3)+noisy_joint_angle(16)+noisy_joint_vel(16)+last_act(16)+last_last_act(16)+last_last_last_act(16)+motor_targets(16)+contact(4)+noisy_gyro(3)+noisy_gravity(3)+noisy_linvel(3)

## Env config

See `run_config.json` (`env_config`) for the full nested config actually used (after config_overrides were applied).
