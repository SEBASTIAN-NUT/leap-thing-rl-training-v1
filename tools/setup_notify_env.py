import os
import stat
import shutil

# 🚨 インストールする場合は、ここにあなたのDiscord Webhook URLを貼り付けてください
# （削除するだけの場合は空欄のままで大丈夫です）
WEBHOOK_URL = "https://discordapp.com/api/webhooks/1488848937719697408/MthRZCuK_S6IQtev_ntKeN6qzKzf-I_AKM0dfaQQzv43Ydp7-iH_5ralFVNidSxbrpq6"

WORKSPACE = "/home/elemechrl/Desktop/iizawa/iizawa_workspace"
VENV_BIN = f"{WORKSPACE}/mujoco_playground/.venv/bin"
NOTIFIER_PATH = f"{WORKSPACE}/discord_notifier.py"
REAL_PYTHON = f"{VENV_BIN}/python.real"

def install():
    if "xxxxxxxx" in WEBHOOK_URL:
        print("\n🚨 エラー: DiscordのWebhook URLが書き換えられていません！")
        print("スクリプトの6行目にある 'WEBHOOK_URL' をお使いのURLに書き換えてから再実行してください。\n")
        return

    # 1. Discord通知用ヘルパースクリプトの作成
    notifier_code = f"""import sys
import requests

WEBHOOK_URL = "{WEBHOOK_URL}"

if len(sys.argv) < 3:
    sys.exit(0)

mode = sys.argv[1]
cmd = sys.argv[2]

if mode == 'start':
    msg = f"🚀 **実行を開始しました**\\n`{{cmd}}`"
elif mode == 'success':
    sec = float(sys.argv[3])
    msg = f"✅ **【正常終了】** ⏱️ {{sec/60:.1f}}分\\n`{{cmd}}`"
elif mode == 'fail':
    sec = float(sys.argv[3])
    err_file = sys.argv[4]
    try:
        with open(err_file, 'r') as f:
            err_text = f.read()[-1500:]  # 直近のエラーメッセージ末尾1500文字を抽出
    except:
        err_text = "エラーログの取得に失敗しました。"
    msg = f"🚨 **【異常終了】** ⏱️ {{sec/60:.1f}}分\\n`{{cmd}}`\\n```python\\n{{err_text}}\\n```"

try:
    requests.post(WEBHOOK_URL, json={{"content": msg}})
except:
    pass
"""

    with open(NOTIFIER_PATH, "w") as f:
        f.write(notifier_code)
    print(f"[1/3] 通知ヘルパーを作成しました: {NOTIFIER_PATH}")

    # 2. 本物のPythonバイナリの退避 (python.real の作成)
    orig_python = f"{VENV_BIN}/python"
    if not os.path.exists(REAL_PYTHON):
        if os.path.islink(orig_python):
            target = os.readlink(orig_python)
            if not os.path.isabs(target):
                target = os.path.normpath(os.path.join(VENV_BIN, target))
            os.symlink(target, REAL_PYTHON)
        else:
            shutil.copy2(orig_python, REAL_PYTHON)
        print(f"[2/3] 本物のPythonを退避しました: {REAL_PYTHON}")
    else:
        print("[2/3] すでに本物のPythonは退避済みです。")

    # 3. 通知機能付きラッパー（シェルスクリプト）の作成
    wrapper_code = f"""#!/bin/bash
REAL_PYTHON="{REAL_PYTHON}"
NOTIFIER="{NOTIFIER_PATH}"

IS_SCRIPT=false
for arg in "$@"; do
    if [[ "$arg" == *.py ]]; then
        IS_SCRIPT=true
        break
    fi
done

if [ "$IS_SCRIPT" = true ] && [ -f "$NOTIFIER" ]; then
    echo "----------------------------------------"
    echo "🚀 [飯沢さん専用自動通知] 実行: python $@"
    echo "----------------------------------------"
    read -p "🔔 Discord通知を有効にしますか？ (y/n) [default: y]: " ans
    if [[ -z "$ans" || "$ans" == "y" || "$ans" == "yes" ]]; then
        CMD_STR="python $@"
        "$REAL_PYTHON" "$NOTIFIER" start "$CMD_STR"
        
        START_TIME=$(date +%s)
        ERR_LOG="/tmp/iizawa_py_err.log"
        rm -f "$ERR_LOG"
        
        "$REAL_PYTHON" "$@" 2> >(tee "$ERR_LOG" >&2)
        EXIT_CODE=$?
        
        END_TIME=$(date +%s)
        ELAPSED_SEC=$(( END_TIME - START_TIME ))
        
        if [ $EXIT_CODE -eq 0 ]; then
            "$REAL_PYTHON" "$NOTIFIER" success "$CMD_STR" "$ELAPSED_SEC"
        else
            "$REAL_PYTHON" "$NOTIFIER" fail "$CMD_STR" "$ELAPSED_SEC" "$ERR_LOG"
        fi
        exit $EXIT_CODE
    else
        echo "🟡 Discord通知を【無効】にして実行します..."
        exec "$REAL_PYTHON" "$@"
    fi
else
    exec "$REAL_PYTHON" "$@"
fi
"""

    for name in ["python", "python3", "python3.13"]:
        path = f"{VENV_BIN}/{name}"
        if os.path.exists(path) or os.path.islink(path):
            try:
                os.remove(path)
            except:
                pass
        with open(path, "w") as f:
            f.write(wrapper_code)
        os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)

    print("[3/3] 仮想環境のPythonのラッパー化が完了しました！")
    print("\n✨ インストール完了！いつも通りのコマンドで自動通知が発動します。")

def uninstall():
    print("\n🧹 自動通知機能を削除し、元の環境に復元します...")
    
    if not os.path.exists(REAL_PYTHON) and not os.path.islink(REAL_PYTHON):
        print("🚨 エラー: python.real が見つかりません。すでに削除されているか、一度もインストールされていません。")
        return
    
    # 退避していた元のPythonがシンボリックリンクか実体ファイルか確認
    is_link = os.path.islink(REAL_PYTHON)
    if is_link:
        target = os.readlink(REAL_PYTHON)
    
    # 既存のラッパー（python, python3, python3.13）を削除
    for name in ["python", "python3", "python3.13"]:
        path = f"{VENV_BIN}/{name}"
        if os.path.exists(path) or os.path.islink(path):
            try:
                os.remove(path)
            except Exception as e:
                print(f"🚨 {name} の削除に失敗: {e}")

    # 本物のPythonを元の名前（python, python3, python3.13）で再配置
    for name in ["python", "python3", "python3.13"]:
        path = f"{VENV_BIN}/{name}"
        try:
            if is_link:
                os.symlink(target, path)
            else:
                shutil.copy2(REAL_PYTHON, path)
                os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
            print(f"✅ 復元しました: {path}")
        except Exception as e:
            print(f"🚨 {name} の復元に失敗: {e}")

    # 退避用ファイルを削除
    try:
        os.remove(REAL_PYTHON)
        print(f"✅ 削除しました: {REAL_PYTHON}")
    except Exception as e:
        print(f"🚨 python.real の削除に失敗: {e}")

    # discord_notifier.py を削除
    if os.path.exists(NOTIFIER_PATH):
        try:
            os.remove(NOTIFIER_PATH)
            print(f"✅ 削除しました: {NOTIFIER_PATH}")
        except Exception as e:
            print(f"🚨 {NOTIFIER_PATH} の削除に失敗: {e}")

    print("\n✨ すべての削除と復元が完了しました！完全に元のクリーンな仮想環境に戻りました。\n")

if __name__ == "__main__":
    print("\n============================================")
    print(" 🤖 飯沢さん専用 Python自動通知管理ツール")
    print("============================================")
    print("1 : 自動通知機能を【インストール / 更新】する")
    print("2 : 自動通知機能を【完全に削除（アンインストール）】する")
    print("--------------------------------------------")
    
    try:
        choice = input("実行する番号を選んでください (1 または 2): ").strip()
        if choice == "1":
            install()
        elif choice == "2":
            uninstall()
        else:
            print("❌ 無効な選択です。何もせず終了します。")
    except KeyboardInterrupt:
        print("\n取消されました。")
