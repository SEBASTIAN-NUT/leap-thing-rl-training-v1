#!/bin/bash
# 推奨3実験を推論実行し、CSV に時系列データを保存

PYTHON="Open_Duck_Playground/.venv/bin/python"
EVAL_DIR="eval_results"
mkdir -p "$EVAL_DIR"

# Phase 3: alive=0.5
echo "[1/3] Phase 3: alive=0.5..."
ONNX1=$(ls -t checkpoints/p3_alive_0p5_20260711_105838/*.onnx 2>/dev/null | head -1)
if [ -n "$ONNX1" ]; then
  $PYTHON -m thing_test.check_command_response \
    -o "$ONNX1" \
    --no_viewer \
    --axes vx,yaw \
    --fractions 0.25,0.5,0.75,1.0 \
    --csv_out "$EVAL_DIR/p3_alive_0p5.csv"
fi

# Phase 5: stand_still=-1.0
echo "[2/3] Phase 5: stand_still=-1.0..."
ONNX2=$(ls -t checkpoints/p5_stand_still_m1p0_20260717_161824/*.onnx 2>/dev/null | head -1)
if [ -n "$ONNX2" ]; then
  $PYTHON -m thing_test.check_command_response \
    -o "$ONNX2" \
    --no_viewer \
    --axes vx,yaw \
    --fractions 0.25,0.5,0.75,1.0 \
    --csv_out "$EVAL_DIR/p5_stand_still_m1p0.csv"
fi

# Phase 5: feet_air_time=0.1
echo "[3/3] Phase 5: feet_air_time=0.1..."
ONNX3=$(ls -t checkpoints/p5_feet_air_time_0p1_20260720_151328/*.onnx 2>/dev/null | head -1)
if [ -n "$ONNX3" ]; then
  $PYTHON -m thing_test.check_command_response \
    -o "$ONNX3" \
    --no_viewer \
    --axes vx,yaw \
    --fractions 0.25,0.5,0.75,1.0 \
    --csv_out "$EVAL_DIR/p5_feet_air_time_0p1.csv"
fi

echo "Done. CSV files saved to $EVAL_DIR/"
