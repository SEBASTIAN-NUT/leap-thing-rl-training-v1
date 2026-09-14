import shutil
from pathlib import Path

SCRIPT_DIR = Path.cwd()
LOGS_DIR = SCRIPT_DIR / "logs"

LOG_PATTERNS = [
    "tb_*.csv",
    "cpg_*.csv",
    "*_experiment.log",
    "*.log",
]

SKIP_DIRS = {
    ".git", "checkpoints", "eval_results", "poster_summary", "Open_Duck_Playground",
    "Open_Duck_Mini_Runtime", "experiments", ".claude", "thing_test",
    "presentations", "mujoco_playground",
}

LOGS_DIR.mkdir(exist_ok=True)

print("=" * 70)
print("Organizing log files into logs/ folder...")
print("=" * 70)

moved_count = 0
for pattern in LOG_PATTERNS:
    for file_path in SCRIPT_DIR.glob(pattern):
        if file_path.is_dir():
            continue
        if file_path.parent == LOGS_DIR:
            continue

        dst_path = LOGS_DIR / file_path.name
        try:
            shutil.move(str(file_path), str(dst_path))
            print(f"✓ {file_path.name} -> logs/")
            moved_count += 1
        except Exception as e:
            print(f"✗ {file_path.name}: {e}")

print("\n" + "=" * 70)
print(f"Moved {moved_count} files to logs/")
print("=" * 70)
