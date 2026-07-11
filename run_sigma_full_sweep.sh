#!/bin/bash
set -o pipefail
cd ~/Desktop/iizawa/iizawa_workspace/thing_project
PYTHON=Open_Duck_Playground/.venv/bin/python
mkdir -p logs

best_reward_of() {
    local logfile=$1
    grep "^STEP:" "$logfile" 2>/dev/null | tail -1 | awk '{print $4}'
}

run_one() {
    local outdir=$1 logfile=$2 overrides=$3
    $PYTHON -m thing_test.runner \
        --num_timesteps 45875200 \
        --output_dir "$outdir" \
        --config_overrides "$overrides" \
        2>&1 | tee "$logfile"
    return ${PIPESTATUS[0]}
}

# ============================================================
# Phase 1: sigma_lin sweep  (sigma_ang=0.25 固定)
# ============================================================
SIGMA_LINS=(0.005 0.01 0.025)
BEST_LIN="0.025"   # fallback
BEST_LIN_REWARD=-999

echo "========================================"
echo " Phase 1: sigma_lin sweep  (sigma_ang=0.25 fixed)"
echo " Start: $(date)"
echo "========================================"

for sigma in "${SIGMA_LINS[@]}"; do
    tag=$(echo $sigma | sed 's/\./p/')
    outdir="checkpoints/p1_siglin_${tag}_$(date +%Y%m%d_%H%M%S)"
    logfile="logs/p1_siglin_${tag}.log"

    echo ""
    echo "--- [Phase1] sigma_lin=$sigma  $(date) ---"
    if run_one "$outdir" "$logfile" \
        "{\"reward_config.tracking_sigma_lin\": $sigma}"; then
        final_reward=$(best_reward_of "$logfile")
        echo ">>> sigma_lin=$sigma  final_reward=${final_reward:-N/A}"
        if [ -n "$final_reward" ] && \
           python3 -c "import sys; sys.exit(0 if float('$final_reward') > float('$BEST_LIN_REWARD') else 1)"; then
            BEST_LIN_REWARD=$final_reward
            BEST_LIN=$sigma
        fi
    else
        echo "!!! sigma_lin=$sigma FAILED — skipping"
    fi
done

echo ""
echo "========================================"
echo " Phase 1 DONE — Best sigma_lin: $BEST_LIN (reward=$BEST_LIN_REWARD)"
echo "========================================"

# ============================================================
# Phase 2: sigma_ang sweep  (sigma_lin=best 固定)
# ============================================================
SIGMA_ANGS=(0.1 0.25 0.5 1.0 2.0)
BEST_ANG="0.25"    # fallback
BEST_ANG_REWARD=-999

echo ""
echo "========================================"
echo " Phase 2: sigma_ang sweep  (sigma_lin=$BEST_LIN fixed)"
echo " Start: $(date)"
echo "========================================"

for sigma in "${SIGMA_ANGS[@]}"; do
    tag=$(echo $sigma | sed 's/\./p/')
    outdir="checkpoints/p2_siggang_${tag}_$(date +%Y%m%d_%H%M%S)"
    logfile="logs/p2_siggang_${tag}.log"

    echo ""
    echo "--- [Phase2] sigma_ang=$sigma  $(date) ---"
    if run_one "$outdir" "$logfile" \
        "{\"reward_config.tracking_sigma_lin\": $BEST_LIN, \"reward_config.tracking_sigma_ang\": $sigma}"; then
        final_reward=$(best_reward_of "$logfile")
        echo ">>> sigma_ang=$sigma  final_reward=${final_reward:-N/A}"
        if [ -n "$final_reward" ] && \
           python3 -c "import sys; sys.exit(0 if float('$final_reward') > float('$BEST_ANG_REWARD') else 1)"; then
            BEST_ANG_REWARD=$final_reward
            BEST_ANG=$sigma
        fi
    else
        echo "!!! sigma_ang=$sigma FAILED — skipping"
    fi
done

echo ""
echo "========================================"
echo " ALL DONE: $(date)"
echo " Best sigma_lin : $BEST_LIN  (reward=$BEST_LIN_REWARD)"
echo " Best sigma_ang : $BEST_ANG  (reward=$BEST_ANG_REWARD)"
echo "========================================"
