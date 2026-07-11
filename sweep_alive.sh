#!/bin/bash
# Phase 3: alive scale sweep  (with per-run JAX cache isolation)
set -uo pipefail

NUM_TIMESTEPS="${1:-150000000}"
ALIVE_VALUES=(0.0 0.05 0.1 0.2 0.3 0.5 1.0)
SIGMA_LIN=0.025
SIGMA_ANG=2.0
TRK_LIN=3.0
TRK_ANG=1.0

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Phase 3: alive scale sweep  (per-run JAX cache isolation)"
echo "  num_timesteps : $NUM_TIMESTEPS"
echo "  sigma_lin=$SIGMA_LIN  sigma_ang=$SIGMA_ANG"
echo "  tracking_lin=$TRK_LIN  tracking_ang=$TRK_ANG"
echo "  alive values  : ${ALIVE_VALUES[*]}"
echo "============================================================"

TOTAL="${#ALIVE_VALUES[@]}"
IDX=0
FAILED=()

for ALIVE in "${ALIVE_VALUES[@]}"; do
    IDX=$((IDX + 1))
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    ALIVE_SLUG=$(echo "$ALIVE" | tr '.' 'p')
    RUN_DIR="checkpoints/p3_alive_${ALIVE_SLUG}_${TIMESTAMP}"

    # 使い捨てキャッシュ: ランごとに隔離してコンパイルキャッシュ汚染を防ぐ
    JAX_CACHE=$(mktemp -d /tmp/jax_cache_XXXXXX)

    OVERRIDES=$(python3 -c "
import json, sys
alive, sigma_l, sigma_a, trk_l, trk_a = (
    float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]),
    float(sys.argv[4]), float(sys.argv[5]),
)
print(json.dumps({
    'reward_config.scales.alive':              alive,
    'reward_config.scales.tracking_lin_vel':   trk_l,
    'reward_config.scales.tracking_ang_vel':   trk_a,
    'reward_config.tracking_sigma_lin':        sigma_l,
    'reward_config.tracking_sigma_ang':        sigma_a,
}))
" "$ALIVE" "$SIGMA_LIN" "$SIGMA_ANG" "$TRK_LIN" "$TRK_ANG")

    echo "=============================="
    echo "[$IDX/$TOTAL] alive=$ALIVE  ->  $RUN_DIR"
    echo "  JAX cache: $JAX_CACHE"
    echo "=============================="

    if JAX_COMPILATION_CACHE_DIR="$JAX_CACHE" \
       Open_Duck_Playground/.venv/bin/python -m thing_test.runner \
           --env joystick \
           --task flat_terrain \
           --num_timesteps "$NUM_TIMESTEPS" \
           --output_dir "$RUN_DIR" \
           --config_overrides "$OVERRIDES" \
           2>&1 | tee "/tmp/p3_alive_${ALIVE_SLUG}.log"; then
        echo "Done: alive=$ALIVE  (run $IDX/$TOTAL)"
        SLEEP_SEC=300   # 成功後は長めに待つ（GPUメモリ解放待ち）
    else
        echo "WARNING: alive=$ALIVE CRASHED, continuing..."
        FAILED+=("$ALIVE")
        SLEEP_SEC=60    # クラッシュ後は短くてよい
    fi

    rm -rf "$JAX_CACHE"   # このランのキャッシュを削除

    if [ "$IDX" -lt "$TOTAL" ]; then
        echo "Sleeping ${SLEEP_SEC}s before next run..."
        sleep "$SLEEP_SEC"
    fi
done

echo "============================================================"
echo "Finished. CRASHED: ${FAILED[*]:-none}"
echo "============================================================"
