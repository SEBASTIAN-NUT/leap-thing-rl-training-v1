"""
this is made to output the result of RL as csv file.
how to use: Open_Duck_Playground/.venv/bin/python export_tb__csv.py checkpoints/2026_06_13_193454
"""


import argparse
import csv
import os
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def export_run(logdir, out_csv):
    if not os.path.isdir(logdir):
        print(f"skip {logdir} (not found)")
        return
    ea = EventAccumulator(logdir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tag", "step", "wall_time", "value"])
        for tag in tags:
            for e in ea.Scalars(tag):
                w.writerow([tag, e.step, e.wall_time, e.value])
    print(f"{logdir}: {len(tags)} tags -> {out_csv}")


def main():
    parser = argparse.ArgumentParser(
        description="Export one TensorBoard run's scalars to a CSV file"
    )
    parser.add_argument(
        "run_dir",
        help="Run directory, e.g. checkpoints/2026_06_13_193454",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="Output CSV path (default: <run_dir>/scalars.csv)",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.rstrip("/")
    out_csv = args.output or os.path.join(run_dir, "scalars.csv")
    export_run(run_dir, out_csv)


if __name__ == "__main__":
    main()
