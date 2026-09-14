#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
HOME = Path.home()
EVAL = ROOT / "results/eval_results" / "archived"
FX = "-1.0,-0.75,-0.5,-0.25,0.25,0.5,0.75,1.0"
CONDS = [
    "p2b_sigang_0p25", "p2b_sigang_1p0", "p2b_sigang_2p0",
    "p3_alive_0p1",    "p3_alive_0p3",    "p3_alive_1p0",
]
ENV = {**os.environ, "PYTHONPATH": str(ROOT) + ":" + os.environ.get("PYTHONPATH", "")}
def find_onnx(rel):
    p = Path(rel)
    if p.exists(): return str(p)
    alt = str(rel).replace("/transfer/checkpoints/", "/transfer/thing_project/checkpoints/")
    if Path(alt).exists(): return alt
    for base in [ROOT, ROOT.parent/"transfer/thing_project", ROOT.parent/"transfer"]:
        c = base / p.relative_to(p.anchor) if p.is_absolute() else base / rel
        try:
            if c.exists(): return str(c)
        except Exception: pass
    return None
for stem in CONDS:
    d = json.loads((EVAL / f"{stem}.json").read_text())
    onnx = find_onnx(d.get("onnx", ""))
    if not onnx:
        print(f"SKIP {stem}: ONNX not found  ({d.get('onnx','')})"); continue
    out = str(EVAL / f"{stem}_full.json")
    print(f"\n=== {stem}  {onnx} ===")
    subprocess.run([sys.executable, "-m", "thing_test.check_command_response",
        "-o", onnx, f"--fractions={FX}", "--axes=vx,yaw",
        "--no_viewer", "--json_out", out],
        env=ENV, cwd=str(ROOT), check=True)
    print(f"-> {out}")
