#!/bin/bash
# Sweep a single config parameter across several values, running a short
# training job for each so the learning-curve TREND (not convergence) can
# be compared in TensorBoard.
#
# Usage:
#   ./sweep_reward.sh <dot.path.to.param> <num_timesteps> <value1> [value2] ...
#
# Each <valueN> is parsed as JSON first, falling back to a plain float if
# that fails -- so both scalar params (reward scales) and list-valued ones
# (command ranges like lin_vel_x, which are [min, max] pairs) work with the
# same script:
#   ./sweep_reward.sh reward_config.scales.alive 500 1.0 5.0 20.0
#   ./sweep_reward.sh lin_vel_x 500 "[-0.1,0.1]" "[-0.2,0.2]" "[-0.3,0.3]"
#
# Each run gets its own checkpoints/sweep_<param>_<value>_<timestamp>/
# directory, so `tensorboard --logdir checkpoints` overlays them all for
# direct comparison. NOTE: changing a swept value changes a constant baked
# into the traced/compiled graph, so each sweep point pays its own JIT
# compile cost (not shared via the persistent cache) -- expect each run to
# take roughly as long as a single fresh run does right now.

set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <dot.path.to.param> <num_timesteps> <value1> [value2] ..."
    echo 'Example (scalar):   $0 reward_config.scales.alive 500 1.0 5.0 20.0'
    echo 'Example (list):     $0 lin_vel_x 500 "[-0.1,0.1]" "[-0.2,0.2]"'
    exit 1
fi

PARAM_PATH="$1"
NUM_TIMESTEPS="$2"
shift 2
VALUES=("$@")

PARAM_SLUG=$(echo "$PARAM_PATH" | tr '.' '_')
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "Sweeping '$PARAM_PATH' over: ${VALUES[*]}"
echo "num_timesteps=$NUM_TIMESTEPS for every run"
echo ""

for VALUE in "${VALUES[@]}"; do
    TIMESTAMP=$(date +%Y_%m_%d_%H%M%S)
    VALUE_SLUG=$(echo "$VALUE" | tr -c 'A-Za-z0-9._-' '_')
    RUN_DIR="checkpoints/sweep_${PARAM_SLUG}_${VALUE_SLUG}_${TIMESTAMP}"
    OVERRIDES=$(python3 -c "
import json, sys
key, raw = sys.argv[1], sys.argv[2]
try:
    value = json.loads(raw)
except json.JSONDecodeError:
    value = float(raw)
print(json.dumps({key: value}))
" "$PARAM_PATH" "$VALUE")

    echo "=================================================="
    echo "Run: $PARAM_PATH = $VALUE  ->  $RUN_DIR"
    echo "config_overrides: $OVERRIDES"
    echo "=================================================="

    Open_Duck_Playground/.venv/bin/python -m thing_test.runner \
        --env joystick \
        --task flat_terrain \
        --num_timesteps "$NUM_TIMESTEPS" \
        --output_dir "$RUN_DIR" \
        --config_overrides "$OVERRIDES" \
        2>&1 | tee "/tmp/sweep_${PARAM_SLUG}_${VALUE_SLUG}.log"

    echo ""
    echo "Done: $PARAM_PATH = $VALUE"
    echo ""
done

echo "All sweep runs finished. Compare with:"
echo "  tensorboard --logdir $SCRIPT_DIR/checkpoints --port 6006"
