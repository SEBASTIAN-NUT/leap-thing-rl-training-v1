"""p2a 2条件の ONNX でヨー CCR を走らせて JSON に保存"""
import subprocess, json, re
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
PYTHON = PROJECT_DIR / "Open_Duck_Playground/.venv/bin/python"

TARGETS = [
    ("p2a_siglin_0p01",
     "checkpoints/p2a_sigma_lin_0p01_20260731_014025/2026_07_31_065315_160563200.onnx"),
    ("p2a_siglin_0p05",
     "checkpoints/p2a_sigma_lin_0p05_20260731_173229/2026_07_31_175532_11468800.onnx"),
]
FRACTIONS = "-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1"

def parse_yaw(output):
    rows = []
    in_summary = False
    for line in output.splitlines():
        if "Summary" in line:
            in_summary = True; continue
        if not in_summary: continue
        cm = re.search(r'cmd[_\s]*yaw[^-+\d]*([+-]?[\d.]+)', line, re.I)
        mm = re.search(r'meas[_\s]*yaw[^-+\d]*([+-]?[\d.]+)', line, re.I)
        if cm and mm:
            rows.append({"cmd_yaw": float(cm.group(1)), "meas_yaw": float(mm.group(1))})
    return rows

for name, onnx_rel in TARGETS:
    onnx = PROJECT_DIR / onnx_rel
    print(f"\n--- {name} (yaw) ---")
    result = subprocess.run(
        [str(PYTHON), "-m", "thing_test.check_command_response",
         "-o", str(onnx),
         "--vx_max", "0.15", "--vy_max", "0.2", "--yaw_max", "1.0",
         f"--fractions={FRACTIONS}",
         "--axes", "yaw", "--phase_duration", "4.0"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, errors="replace"
    )
    out = result.stdout + "\n" + result.stderr
    rows = parse_yaw(out)
    if not rows:
        print("パース失敗"); print(out[-2000:]); continue
    cache = PROJECT_DIR / f"ccr_{name}_yaw.json"
    cache.write_text(json.dumps({"yaw_sweep": rows}, indent=2))
    print(f"{len(rows)} 点 -> {cache}")
    for r in rows:
        print(f"  cmd={r['cmd_yaw']:+.3f}  meas={r['meas_yaw']:+.3f}")
