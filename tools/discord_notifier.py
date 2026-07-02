import sys
import requests

WEBHOOK_URL = "https://discordapp.com/api/webhooks/1488848937719697408/MthRZCuK_S6IQtev_ntKeN6qzKzf-I_AKM0dfaQQzv43Ydp7-iH_5ralFVNidSxbrpq6"

if len(sys.argv) < 3:
    sys.exit(0)

mode = sys.argv[1]
cmd = sys.argv[2]

if mode == 'start':
    msg = f"🚀 **実行を開始しました**\n`{cmd}`"
elif mode == 'success':
    sec = float(sys.argv[3])
    msg = f"✅ **【正常終了】** ⏱️ {sec/60:.1f}分\n`{cmd}`"
elif mode == 'fail':
    sec = float(sys.argv[3])
    err_file = sys.argv[4]
    try:
        with open(err_file, 'r') as f:
            err_text = f.read()[-1500:]  # 直近のエラーメッセージ末尾1500文字を抽出
    except:
        err_text = "エラーログの取得に失敗しました。"
    msg = f"🚨 **【異常終了】** ⏱️ {sec/60:.1f}分\n`{cmd}`\n```python\n{err_text}\n```"

try:
    requests.post(WEBHOOK_URL, json={"content": msg})
except:
    pass
