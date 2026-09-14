#!/usr/bin/env python3
"""Batch CCR runner -- auto-discovers ONNX, runs headless, saves pairs JSON."""
from pathlib import Path
import subprocess, json, tempfile

BASE = Path('/home/elemechrl/Desktop/iizawa/iizawa_workspace')
WD   = BASE  # workspace root so -m imports work
MODULE = 'thing_project.check_command_response'

CKPT_DIRS = [
    BASE / 'transfer/thing_project/checkpoints',
    BASE / 'transfer/checkpoints',
]

CONDITIONS = [
    ('p2b_sigang_0p25', 'siggang_0p25',       [BASE/'transfer/thing_project/checkpoints']),
    ('p2b_sigang_1p0',  'siggang_1p0',        [BASE/'transfer/thing_project/checkpoints']),
    ('p2b_sigang_2p0',  'siggang_rerun_2p0',  [BASE/'transfer/thing_project/checkpoints']),
    ('p3_alive_0p1',    'alive_0p1',          [BASE/'transfer/thing_project/checkpoints',
                                               BASE/'transfer/checkpoints']),
    ('p3_alive_0p3',    'alive_0p3',          [BASE/'transfer/checkpoints']),
    ('p3_alive_1p0',    'alive_1p0',          [BASE/'transfer/thing_project/checkpoints',
                                               BASE/'transfer/checkpoints']),
]

FRACTIONS   = '-1.0,-0.75,-0.5,-0.25,0.25,0.5,0.75,1.0'
AXIS_CMD    = {'vx': 'cmd_vx',  'yaw': 'cmd_yaw'}
AXIS_MEAS   = {'vx': 'meas_vx', 'yaw': 'meas_yaw'}

def find_onnx(keyword, dirs):
    for d in dirs:
        if not d.exists():
            continue
        for folder in sorted(d.iterdir()):
            if keyword in folder.name:
                onnx = [f for f in sorted(folder.glob('*.onnx'))
                        if not f.name.endswith('_0.onnx')]
                if onnx:
                    return onnx[-1]
    return None

def run_one(onnx, axis, json_out):
    cmd = [
        'python3', '-m', MODULE,
        '-o', str(onnx),
        '--no_viewer',
        '--axes', axis,
        '--fractions', FRACTIONS,
        '--phase_duration', '3.0',
        '--json_out', str(json_out),
    ]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(WD))

for name, keyword, dirs in CONDITIONS:
    print(f'\n=== {name} ===')
    onnx = find_onnx(keyword, dirs)
    if onnx is None:
        print(f'  ERROR: no ONNX for keyword "{keyword}"')
        continue
    print(f'  ONNX: {onnx.name}')
    for axis in ['vx', 'yaw']:
        out = Path(f'thing_project/ccr_{name}_{axis}.json')
        tmp = Path(tempfile.mktemp(suffix='.json'))
        print(f'  {axis}: running...', end=' ', flush=True)
        r = run_one(onnx, axis, tmp)
        if tmp.exists():
            raw = json.loads(tmp.read_text())
            pairs = [
                {'cmd': p[AXIS_CMD[axis]], 'meas': p[AXIS_MEAS[axis]]}
                for p in raw.get('phases', [])
                if p.get(AXIS_CMD[axis], 0.0) != 0.0
            ]
            out.write_text(json.dumps({'pairs': pairs}))
            print(f'{len(pairs)} pts -> {out.name}')
            tmp.unlink(missing_ok=True)
        else:
            print(f'FAILED (rc={r.returncode})')
            print(f'    stderr: {r.stderr[:400]}')

print('\nDone!')
