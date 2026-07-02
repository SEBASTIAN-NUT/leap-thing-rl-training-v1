#!/bin/bash
# thing_project (LEAP Hand RL) 作業環境起動スクリプト
# Usage:
#   bash ~/Desktop/iizawa/iizawa_workspace/thing_project/tools/start_thing_env.sh
#   または: chmod +x tools/start_thing_env.sh && ./tools/start_thing_env.sh

THING_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="$THING_DIR/Open_Duck_Playground/.venv"

if [ ! -d "$VENV" ]; then
    echo "venvが見つかりません: $VENV"
    echo "先に: cd $THING_DIR && uv sync を実行してください"
    exit 1
fi

cd "$THING_DIR"

echo "================================================"
echo "  LEAP Hand RL 作業環境 (uv venv)"
echo "  $("$VENV/bin/python" --version)"
echo "  ディレクトリ: $THING_DIR"
echo "================================================"
echo ""
echo "よく使うコマンド:"
echo "  # sigma候補スクリーニング（学習前の報酬確認）"
echo "  python -m thing_test.screen_sigma_candidates"
echo "  python -m thing_test.analyze_reward_gradient"
echo ""
echo "  # 学習"
echo "  python -m thing_test.thing_walk"
echo ""
echo "  # 学習済みモデルの動作確認"
echo "  python -m thing_test.check_command_response \\"
echo "      -o checkpoints/<run>/<ts>_<step>.onnx \\"
echo "      --vx_max 0.15 --yaw_max 1.0 --phase_duration 4.0"
echo ""
echo "  # Discord通知セットアップ（初回のみ）"
echo "  python tools/setup_notify_env.py"
echo ""
echo "================================================"
echo "exit で環境を抜けます"
echo ""

VIRTUAL_ENV="$VENV" PATH="$VENV/bin:$PATH" exec bash
