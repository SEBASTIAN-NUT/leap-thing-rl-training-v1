#!/bin/bash
# Usage: ./run_one_alive.sh <alive_value>
# e.g.   ./run_one_alive.sh 0.5
set -uo pipefail
ALIVE="${1:?Usage: $0 <alive_value>}"
ALIVE_SLUG=$(echo "$ALIVE" | tr '.' 'p')
RUN_DIR="checkpoints/p3_alive_${ALIVE_SLUG}_$(date +%Y%m%d_%H%M%S)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== Clearing XLA cache before run ==="
rm -rf ~/.cache/jax_thing_cache

OVERRIDES=$(python3 -c "
import json, sys
alive = float(sys.argv[1])
print(json.dumps({
    'reward_config.scales.alive':              alive,
    'reward_config.scales.tracking_lin_vel':   3.0,
    'reward_config.scales.tracking_ang_vel':   1.0,
    'reward_config.tracking_sigma_lin':        0.025,
    'reward_config.tracking_sigma_ang':        2.0,
}))
" "$ALIVE")

echo "alive=$ALIVE  ->  $RUN_DIR"
echo "overrides: $OVERRIDES"

Open_Duck_Playground/.venv/bin/python -m thing_test.runner \
    --env joystick \
    --task flat_terrain \
    --num_timesteps 150000000 \
    --output_dir "$RUN_DIR" \
    --config_overrides "$OVERRIDES"
