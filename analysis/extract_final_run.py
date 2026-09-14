#!/usr/bin/env python3
"""
extract_final_run.py - downsample the real cpg_record.py output
(cpg_run_final_com.csv, cpg_run_final_if_finger.csv) into poster-sized
CSV files, writing them with csv.writer instead of printing a table
that would need to be manually retyped. Writes:
  - final_com_timeseries_20s.csv        (every 0.2s, t=0..20)
  - final_if_finger_one_cycle_t10to11.csv (t=10.0-11.0s, every 10th row)
"""
import csv

with open("cpg_run_final_com.csv") as f:
    com_rows = list(csv.DictReader(f))

with open("final_com_timeseries_20s.csv", "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["t_s", "com_x_mm", "palm_x_mm", "tilt_deg"])
    for i in range(0, len(com_rows), 100):
        r = com_rows[i]
        w.writerow([round(float(r['t']), 3),
                    round(float(r['com_x']) * 1000, 2),
                    round(float(r['palm_x']) * 1000, 2),
                    round(float(r['tilt_deg']), 2)])
print("Wrote final_com_timeseries_20s.csv")

with open("cpg_run_final_if_finger.csv") as f:
    fin_rows = list(csv.DictReader(f))

with open("final_if_finger_one_cycle_t10to11.csv", "w", newline="") as fp:
    w = csv.writer(fp)
    w.writerow(["t_s", "tip_x_rel_mm", "tip_z_rel_mm", "x_target_mm", "z_target_mm",
                "if_mcp_rad", "if_pip_rad"])
    for idx, r in enumerate(fin_rows):
        t = float(r['t'])
        if 10.0 <= t <= 11.0 and idx % 10 == 0:
            w.writerow([f"{t:.3f}",
                        round(float(r['tip_x_rel']) * 1000, 2),
                        round(float(r['tip_z_rel']) * 1000, 2),
                        round(float(r['x_target_rel']) * 1000, 2),
                        round(float(r['z_target_rel']) * 1000, 2),
                        round(float(r['if_mcp']), 4),
                        round(float(r['if_pip']), 4)])
print("Wrote final_if_finger_one_cycle_t10to11.csv")
