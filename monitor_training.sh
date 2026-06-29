#!/bin/bash
PID=1044320
INTERVAL=30
PREV_LINE=""
SAME_COUNT=0
PYSPY="Open_Duck_Playground/.venv/bin/py-spy"

while kill -0 "$PID" 2>/dev/null; do
    ELAPSED=$(ps -o etimes= -p "$PID" | tr -d ' ')
    STACK=$("$PYSPY" dump --pid "$PID" 2>&1)
    LINE=$(echo "$STACK" | grep -E "mujoco/mjx/_src|thing_test/thing_walk|brax/training" | head -1 | sed 's/^ *//')

    if [ -z "$LINE" ]; then
        LINE="(取得失敗: $(echo "$STACK" | head -1))"
    fi

    if [ "$LINE" == "$PREV_LINE" ]; then
        SAME_COUNT=$((SAME_COUNT + 1))
    else
        SAME_COUNT=0
    fi
    PREV_LINE="$LINE"

    if [ "$SAME_COUNT" -ge 10 ]; then
        FLAG=" <<< 同じ場所が続いている(${SAME_COUNT}回, 要注意)"
    else
        FLAG=""
    fi

    echo "[${ELAPSED}s] ${LINE}${FLAG}"
    sleep "$INTERVAL"
done

echo ">>> プロセス $PID は終了しました(完了 or クラッシュ)。ターミナルの出力を確認してください。"
