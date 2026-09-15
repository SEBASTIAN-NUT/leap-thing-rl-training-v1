import sys as _sys; from pathlib import Path as _Path; _sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))
#!/usr/bin/env python3
"""
deploy_main.py
"""
import time
import signal
import threading
import asyncio
import json
import socket
import numpy as np
import onnxruntime as ort
import h5py

from leap_hardware import LeapHardware
from leap_env import LeapEnvironment
from leap_logger import LeapLogger
from leap_policy import DemoPolicy, AIPolicy

MOTION_MODE = "replay"
ONNX_MODEL_PATH = "/workspace/policy.onnx"
MEAN_NPY_PATH   = "/workspace/obs_mean.npy"
STD_NPY_PATH    = "/workspace/obs_std.npy"
LIMITS_YAML     = "/workspace/leap_hand_limits.yaml"
H5_REPLAY_PATH  = "/workspace/rollout.h5"

running = True
UDP_PORT = 5005
latest_target_task_data = [0.0, 0.0, -0.04, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
task_data_lock = threading.Lock()

def udp_receiver_thread():
    global latest_target_task_data, running
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("0.0.0.0", UDP_PORT))
    sock.settimeout(0.5)
    print(f"📡 UDP受信スレッド起動成功: Port {UDP_PORT} でPCからのデータを待機中...")
    while running:
        try:
            data, addr = sock.recvfrom(1024)
            payload = json.loads(data.decode('utf-8'))
            if isinstance(payload, list) and len(payload) == 9:
                with task_data_lock:
                    latest_target_task_data = payload
        except socket.timeout:
            continue
        except Exception:
            time.sleep(0.1)
    sock.close()

def shutdown_handler(signum, frame):
    global running
    print("\n🛑 緊急停止シグナルを受信しました。終了処理に入ります...")
    running = False

signal.signal(signal.SIGINT, shutdown_handler)

def listen_enter_key():
    global running
    input("⌨️  [Enter] キーを押すといつでも安全に緊急停止できます...\n")
    running = False

async def main():
    global running
    print(f"🚀 LEAP Hand 実機デプロイを開始します (モード: {MOTION_MODE})")

    hardware = LeapHardware(port_name='/dev/ttyUSB0', baudrate=3000000, dry_run=False)
    env = LeapEnvironment(LIMITS_YAML, mean_npy_path=MEAN_NPY_PATH, std_npy_path=STD_NPY_PATH)
    logger = LeapLogger()

    obs_mean = np.load(MEAN_NPY_PATH).astype(np.float32)
    obs_std  = np.load(STD_NPY_PATH).astype(np.float32)

    policy = None
    replay_timestamps = None
    replay_actions = None

    if MOTION_MODE.startswith("demo_"):
        demo_mode = MOTION_MODE.replace("demo_", "")
        policy = DemoPolicy(mode=demo_mode)
    elif MOTION_MODE == "ai_policy":
        policy = AIPolicy(model_path=ONNX_MODEL_PATH)
    elif MOTION_MODE == "replay":
        print(f"📦 リプレイデータを読み込み中: {H5_REPLAY_PATH}")
        with h5py.File(H5_REPLAY_PATH, "r") as f:
            replay_timestamps = f["timestamps"][:]
            replay_actions = f["raw_actions"][:]
        print(f"✅ ロード完了: {len(replay_timestamps)} ステップを再生します。")

    threading.Thread(target=udp_receiver_thread, daemon=True).start()
    threading.Thread(target=listen_enter_key, daemon=True).start()

    current_pulses, raw_velocities, current_mA = hardware.read_hardware_state()
    pulse_cmds = current_pulses.copy()

    start_time = time.time()
    last_print_time = time.time()
    last_loop_time = time.time()

    print("▶️  制御ループを開始します。")
    
    while running:
        loop_start = time.time()
        t = loop_start - start_time

        with task_data_lock:
            task_raw = np.array(latest_target_task_data, dtype=np.float32)

        current_pulses, raw_velocities, current_mA = hardware.read_hardware_state()
        norm_obs, raw_obs = env.make_observation(current_pulses, raw_velocities, task_raw)
        vel_rad_s = raw_velocities * 0.229 * 2.0 * np.pi / 60.0

        ai_actions = None

        if MOTION_MODE == "home":
            pass

        elif MOTION_MODE == "sinusoid":
            ai_actions = np.zeros(16, dtype=np.float32)
            ai_actions[0] = np.sin(2 * np.pi * 0.5 * t) * 0.5
            raw_target_pulses = env.action_to_pulses(ai_actions, pulse_cmds)
            pulse_cmds = pulse_cmds + np.clip(raw_target_pulses - pulse_cmds, -40, 40)

        elif MOTION_MODE.startswith("demo_") or MOTION_MODE == "ai_policy" or MOTION_MODE == "replay":
            
            if MOTION_MODE.startswith("demo_"):
                ai_actions = policy.get_action(norm_obs, t)
                ai_actions = np.clip(ai_actions, -1.0, 1.0)
                raw_target_pulses = env.action_to_pulses(ai_actions, pulse_cmds)
                delta = raw_target_pulses - pulse_cmds
                max_open  = 70
                max_close = 10
                clipped = np.where(delta > 0, np.clip(delta, 0, max_open),
                                              np.clip(delta, -max_close, 0))
                pulse_cmds = pulse_cmds + clipped

            elif MOTION_MODE == "ai_policy":
                task_mean = obs_mean[48:57]
                task_std  = obs_std[48:57]
                norm_obs[48:57] = (task_raw - task_mean) / (task_std + 1e-8)
                ai_actions = policy.get_action(norm_obs, t)
                ai_actions = np.clip(ai_actions, -1.0, 1.0)
                raw_target_pulses = env.action_to_pulses(ai_actions, pulse_cmds)
                raw_target_pulses = np.clip(raw_target_pulses, current_pulses - 150, current_pulses + 150)
                pulse_cmds = pulse_cmds + np.clip(raw_target_pulses - pulse_cmds, -40, 40)

            elif MOTION_MODE == "replay":
                if t > replay_timestamps[-1] + 1.0:
                    print("\n🎬 リプレイ再生が完了しました。")
                    running = False
                    break

                ai_actions = np.zeros(16, dtype=np.float32)
                for j in range(16):
                    ai_actions[j] = np.interp(t, replay_timestamps, replay_actions[:, j])

                ai_actions = np.clip(ai_actions, -1.0, 1.0)
                raw_target_pulses = env.action_to_pulses(ai_actions, pulse_cmds)
                raw_target_pulses = np.clip(raw_target_pulses, current_pulses - 150, current_pulses + 150)
                pulse_cmds = pulse_cmds + np.clip(raw_target_pulses - pulse_cmds, -40, 40)

        pulse_cmds_int = np.round(pulse_cmds).astype(np.int32)
        hardware.write_hardware_positions(pulse_cmds_int)

        loop_end = time.time()
        loop_period_ms = (loop_end - last_loop_time) * 1000.0
        last_loop_time = loop_end
        rtt_ms = (loop_end - loop_start) * 1000.0

        try:
            logger.log_state(current_pulses, pulse_cmds_int, vel_rad_s, current_mA, loop_period_ms, rtt_ms)
        except TypeError:
            logger.log_state(current_pulses, pulse_cmds_int, vel_rad_s, loop_period_ms, rtt_ms)

        if time.time() - last_print_time > 0.5:
            print(f"\n--- 🖐 モニター [{MOTION_MODE}]  t={t:.1f}s ---")
            if MOTION_MODE == "replay":
                progress_percent = (np.searchsorted(replay_timestamps, t) / len(replay_timestamps)) * 100 if replay_timestamps is not None else 0
                print(f" 📼 再生進捗       : {progress_percent:.1f}%")
            print(f"  ・現在位置   (0-3): {np.round(current_pulses[0:4]).astype(int)}")
            print(f"  ・目標パルス (0-3): {pulse_cmds_int[0:4]}")
            if ai_actions is not None:
                print(f"  🧪 アクション(先頭5): {np.round(ai_actions[0:5], 2)}")
            print(f"  ・追従誤差   (0-3): {pulse_cmds_int[0:4] - current_pulses[0:4]}")
            print(f"  ・速度[rad/s](0-3): {np.round(vel_rad_s[0:4], 2)}")
            with task_data_lock:
                print(f"  ・タスク目標(位置): {np.round(latest_target_task_data[0:3], 3)}")
                print(f"  ・タスク目標(回転): {np.round(latest_target_task_data[3:6], 2)}")
            print(f"  ・ループ周期      : {loop_period_ms:.1f} ms  (処理+通信RTT: {rtt_ms:.1f} ms)")
            last_print_time = time.time()

        await asyncio.sleep(0.001)

    print("\n🏁 制御ループを終了しました。安全のためトルクを解除します...")
    try:
        for motor_id in range(0, 17):
            hardware.packetHandler.write1ByteTxRx(hardware.portHandler, motor_id, 64, 0)
        print("✅ 全モーターのトルクをOFFにしました。")
    except Exception as e:
        print(f"⚠️ トルク解除中にエラーが発生しました: {e}")

    try:
        hardware.portHandler.closePort()
    except:
        pass

    print("👋 デプロイスクリプトを安全に終了しました。")

if __name__ == "__main__":
    asyncio.run(main())
