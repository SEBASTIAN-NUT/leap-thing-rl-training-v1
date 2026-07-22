"""Check how much a trained LEAP hand policy actually moves for different
commanded velocities, while showing it in the MuJoCo viewer.

Everything that previously would have required editing this file is now a
CLI flag, so the same file can be re-run repeatedly with different values
for command-range experiments:

    Open_Duck_Playground/.venv/bin/python -m thing_test.check_command_response \
        -o checkpoints/<run>/<ts>_<step>.onnx \
        --vx_max 0.15 --vy_max 0.2 --yaw_max 1.0 \
        --fractions 0,0.25,0.5,0.75,1.0 \
        --axes vx,vy,yaw \
        --phase_duration 4.0

Each phase commands a fixed body-frame velocity for --phase_duration
seconds, then reports the *measured* velocity (from actual palm
displacement) next to the commanded one -- useful both as a quick video
check and to see whether a command range is physically reasonable before
spending time on reward-weight tuning.
"""

import argparse
import csv
import json
import math
import time
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from . import constants
from .common.onnx_infer import OnnxInfer
from .mujoco_infer_base import MJInferBase

ACTION_SCALE = 0.25
DOF_VEL_SCALE = 0.05
MAX_MOTOR_VELOCITY = 5.24
USE_MOTOR_SPEED_LIMITS = True


def build_phases(vx_max, vy_max, yaw_max, fractions, axes):
    phases = [("stop", [0.0, 0.0, 0.0])]
    axis_max = {"vx": vx_max, "vy": vy_max, "yaw": yaw_max}
    axis_index = {"vx": 0, "vy": 1, "yaw": 2}
    for axis in axes:
        for frac in fractions:
            if frac == 0:
                continue
            cmd = [0.0, 0.0, 0.0]
            cmd[axis_index[axis]] = axis_max[axis] * frac
            phases.append((f"{axis} {frac*100:.0f}%", cmd))
    return phases


class CommandResponseCheck(MJInferBase):
    def __init__(self, model_path: str, onnx_model_path: str):
        super().__init__(model_path)
        self.policy = OnnxInfer(onnx_model_path, awd=True)
        self.last_action = np.zeros(self.model.nu)
        self.last_last_action = np.zeros(self.model.nu)
        self.last_last_last_action = np.zeros(self.model.nu)
        self.motor_targets = self.default_actuator.copy()
        self.prev_motor_targets = self.default_actuator.copy()
        self.commands = [0.0, 0.0, 0.0]

    def get_obs(self, data, commands) -> np.ndarray:
        # 112-dim obs matching thing_walk.py _get_obs() state:
        # cmd(3) + joint_pos(16) + joint_vel(16) + last_act(16)*3 +
        # motor_targets(16) + contact(4) + gyro(3) + gravity(3) + linvel(3)
        joint_angles = self.get_actuator_joints_qpos(data.qpos)
        joint_backlash = self.get_actuator_backlash_qpos(data.qpos)
        for i in self.backlash_idx_to_add:
            joint_backlash = np.insert(joint_backlash, i, 0)
        joint_angles = joint_angles + joint_backlash

        joint_vel = self.get_actuator_joints_qvel(data.qvel)
        contacts = self.get_feet_contacts(data)
        gyro = self.get_gyro(data)
        gravity = self.get_gravity(data)
        linvel = self.get_sensor(data, "local_linvel")

        obs = np.concatenate(
            [
                np.array(commands, dtype=np.float64),
                joint_angles - self.default_actuator,
                joint_vel * DOF_VEL_SCALE,
                self.last_action,
                self.last_last_action,
                self.last_last_last_action,
                self.motor_targets,
                contacts,
                gyro,
                gravity,
                linvel,
            ]
        )
        return obs.astype(np.float32)

    def palm_xy_yaw(self, data):
        x = data.qpos[self.floating_base_qpos_addr + 0]
        y = data.qpos[self.floating_base_qpos_addr + 1]
        w, qx, qy, qz = data.qpos[
            self.floating_base_qpos_addr + 3 : self.floating_base_qpos_addr + 7
        ]
        yaw = np.arctan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return float(x), float(y), float(yaw)

    def run(self, phases, phase_duration, show_viewer=True, policy_warmup=0.0,
            video_out=None, csv_out=None):
        """policy_warmup: seconds to hold the home pose (policy disabled,
        motor_targets pinned to default_actuator) before letting the
        policy drive at all. Set > 0 to isolate "is the home pose itself
        physically stable" from "is the (possibly undertrained) policy
        producing bad actions" when something looks unstable -- mirrors
        mujoco_infer.py's policy_enabled-starts-False safety check, which
        this script originally lacked (it ran the policy from frame 1)."""
        results = []
        frames = []
        renderer = None
        csv_rows = [] if csv_out else None
        if video_out is not None:
            renderer = mujoco.Renderer(self.model, height=480, width=640)

        def loop(viewer):
            counter = 0
            for label, cmd in phases:
                if viewer is not None and not viewer.is_running():
                    break
                self.commands = cmd
                x0, y0, yaw0 = self.palm_xy_yaw(self.data)
                t0 = self.data.time
                phase_steps = int(phase_duration / self.sim_dt)

                for _ in range(phase_steps):
                    if viewer is not None and not viewer.is_running():
                        break
                    step_start = time.time()
                    mujoco.mj_step(self.model, self.data)
                    counter += 1

                    if counter % self.decimation == 0:
                        if self.data.time < policy_warmup:
                            self.motor_targets = self.default_actuator.copy()
                            self.prev_motor_targets = self.motor_targets.copy()
                            self.data.ctrl[:] = self.motor_targets
                            continue
                        obs = self.get_obs(self.data, self.commands)
                        action = np.asarray(self.policy.infer(obs))

                        self.last_last_last_action = self.last_last_action.copy()
                        self.last_last_action = self.last_action.copy()
                        self.last_action = action.copy()

                        self.motor_targets = (
                            self.default_actuator + action * ACTION_SCALE
                        )
                        if USE_MOTOR_SPEED_LIMITS:
                            max_delta = (
                                MAX_MOTOR_VELOCITY * self.sim_dt * self.decimation
                            )
                            self.motor_targets = np.clip(
                                self.motor_targets,
                                self.prev_motor_targets - max_delta,
                                self.prev_motor_targets + max_delta,
                            )
                            self.prev_motor_targets = self.motor_targets.copy()
                        self.data.ctrl[:] = self.motor_targets

                        if csv_rows is not None:
                            lv = self.get_sensor(self.data, "local_linvel")
                            gy = self.get_sensor(self.data, "imu_gyro")
                            csv_rows.append({
                                "time": round(float(self.data.time), 4),
                                "phase": label,
                                "cmd_vx": float(cmd[0]),
                                "cmd_vy": float(cmd[1]),
                                "cmd_yaw": float(cmd[2]),
                                "inst_vx": round(float(lv[0]), 4),
                                "inst_vy": round(float(lv[1]), 4),
                                "inst_yaw_rate": round(float(gy[2]), 4),
                            })

                    if renderer is not None and counter % self.decimation == 0:
                        renderer.update_scene(self.data)
                        frames.append(renderer.render().copy())

                    if viewer is not None:
                        viewer.sync()
                        time_until_next_step = self.sim_dt - (
                            time.time() - step_start
                        )
                        if time_until_next_step > 0:
                            time.sleep(time_until_next_step)

                x1, y1, yaw1 = self.palm_xy_yaw(self.data)
                dt = self.data.time - t0
                dx, dy = x1 - x0, y1 - y0
                c, s = np.cos(-yaw0), np.sin(-yaw0)
                local_dx = c * dx - s * dy
                local_dy = s * dx + c * dy
                dyaw = np.arctan2(np.sin(yaw1 - yaw0), np.cos(yaw1 - yaw0))

                measured_vx = local_dx / dt if dt > 0 else 0.0
                measured_vy = local_dy / dt if dt > 0 else 0.0
                measured_wyaw = dyaw / dt if dt > 0 else 0.0
                results.append((label, cmd, measured_vx, measured_vy, measured_wyaw))
                print(
                    f"[{label:14s}] cmd=(vx={cmd[0]:+.3f}, vy={cmd[1]:+.3f}, "
                    f"yaw={cmd[2]:+.3f})  ->  measured "
                    f"(vx={measured_vx:+.3f}, vy={measured_vy:+.3f}, "
                    f"yaw_rate={measured_wyaw:+.3f})"
                )

        if show_viewer:
            with mujoco.viewer.launch_passive(
                self.model,
                self.data,
                show_left_ui=False,
                show_right_ui=False,
            ) as viewer:
                loop(viewer)
        else:
            loop(None)

        if renderer is not None:
            renderer.close()

        if csv_out and csv_rows:
            Path(csv_out).parent.mkdir(parents=True, exist_ok=True)
            with open(csv_out, "w", newline="", encoding="utf-8") as _f:
                _w = csv.DictWriter(_f, fieldnames=list(csv_rows[0].keys()))
                _w.writeheader()
                _w.writerows(csv_rows)
            print(f"CSV saved -> {csv_out}")

        if video_out and frames:
            try:
                import imageio
                ctrl_hz = int(1 / (self.sim_dt * self.decimation))
                Path(video_out).parent.mkdir(parents=True, exist_ok=True)
                imageio.mimsave(video_out, frames, fps=ctrl_hz)
                print(f"Video saved → {video_out}")
            except ImportError:
                print("動画保存には imageio が必要: pip install imageio[ffmpeg]")

        print("\n=== Summary: commanded vs measured ===")
        for label, cmd, mvx, mvy, mwyaw in results:
            print(
                f"{label:14s} cmd_vx={cmd[0]:+.3f} meas_vx={mvx:+.3f}   "
                f"cmd_vy={cmd[1]:+.3f} meas_vy={mvy:+.3f}   "
                f"cmd_yaw={cmd[2]:+.3f} meas_yaw={mwyaw:+.3f}"
            )
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    parser.add_argument(
        "--model_path",
        type=str,
        default=str(constants.task_to_xml("flat_terrain")),
    )
    parser.add_argument(
        "--vx_max", type=float, default=0.15, help="lin_vel_x command range max"
    )
    parser.add_argument(
        "--vy_max", type=float, default=0.2, help="lin_vel_y command range max"
    )
    parser.add_argument(
        "--yaw_max", type=float, default=1.0, help="ang_vel_yaw command range max"
    )
    parser.add_argument(
        "--fractions",
        type=str,
        default="0,0.25,0.5,0.75,1.0",
        help="Comma-separated fractions of each axis max to test",
    )
    parser.add_argument(
        "--axes",
        type=str,
        default="vx,vy,yaw",
        help="Comma-separated subset of vx,vy,yaw to test",
    )
    parser.add_argument(
        "--phase_duration",
        type=float,
        default=4.0,
        help="Sim seconds to hold each command before measuring",
    )
    parser.add_argument(
        "--no_viewer",
        action="store_true",
        help="Run headless (no MuJoCo window), just print measured numbers",
    )
    parser.add_argument(
        "--policy_warmup",
        type=float,
        default=0.0,
        help=(
            "Sim seconds to hold the home pose before the policy takes "
            "over. Use e.g. 2.0 to check whether the home pose itself is "
            "physically stable, isolated from the policy."
        ),
    )
    parser.add_argument(
        "--json_out",
        type=str,
        default=None,
        help="結果を JSON ファイルに保存 (例: eval_results/sigma_0p005.json)",
    )
    parser.add_argument(
        "--video_out",
        type=str,
        default=None,
        help="動画を保存 (例: eval_results/sigma_0p005.mp4)  要: pip install imageio[ffmpeg]",
    )
    parser.add_argument(
        "--csv_out",
        type=str,
        default=None,
        help="瞬時速度の時系列データを CSV に保存",
    )
    args = parser.parse_args()

    fractions = [float(f) for f in args.fractions.split(",") if f != ""]
    axes = [a.strip() for a in args.axes.split(",") if a.strip()]
    phases = build_phases(args.vx_max, args.vy_max, args.yaw_max, fractions, axes)

    checker = CommandResponseCheck(args.model_path, args.onnx_model_path)
    results = checker.run(
        phases,
        args.phase_duration,
        show_viewer=not args.no_viewer,
        policy_warmup=args.policy_warmup,
        video_out=args.video_out,
        csv_out=args.csv_out,
    )

    if args.json_out and results:
        records = [
            {
                "phase": label,
                "cmd_vx": cmd[0], "cmd_vy": cmd[1], "cmd_yaw": cmd[2],
                "meas_vx": float(mvx), "meas_vy": float(mvy), "meas_yaw": float(mwyaw),
                "err_vx": abs(cmd[0] - mvx),
                "err_vy": abs(cmd[1] - mvy),
                "err_yaw": abs(cmd[2] - mwyaw),
            }
            for label, cmd, mvx, mvy, mwyaw in results
        ]
        n = len(records)
        rmse_vx  = math.sqrt(sum(r["err_vx"]  ** 2 for r in records) / n)
        rmse_yaw = math.sqrt(sum(r["err_yaw"] ** 2 for r in records) / n)
        output = {
            "onnx": args.onnx_model_path,
            "timestamp": datetime.now().isoformat(),
            "protocol": {
                "vx_max": args.vx_max, "vy_max": args.vy_max,
                "yaw_max": args.yaw_max, "fractions": fractions,
                "axes": axes, "phase_duration": args.phase_duration,
            },
            "rmse_vx": rmse_vx,
            "rmse_yaw": rmse_yaw,
            "phases": records,
        }
        from pathlib import Path
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            import json as _json
            _json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\nJSON saved → {args.json_out}")


if __name__ == "__main__":
    main()
