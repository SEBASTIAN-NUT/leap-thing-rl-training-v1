#!/usr/bin/env python3
"""CPG Parametric Sweep with Automatic Metadata Recording"""

import json, csv, subprocess, sys, os
from pathlib import Path
from datetime import datetime

class CPGSweep:
    def __init__(self):
        self.experiments = []
        self.summary_path = Path("experiments_summary.csv")
        self.metadata_dir = Path("experiment_metadata")
        self.metadata_dir.mkdir(exist_ok=True)

    def run_single_experiment(self, push, lift, freq, param_set_name):
        """Run CPG with specific parameters and record metadata."""
        
        timestamp = datetime.now().isoformat(timespec='seconds')
        exp_id = f"push{push:.3f}_lift{lift:.3f}_freq{freq:.1f}_{param_set_name}"
        
        print(f"\n{'='*60}")
        print(f"Running: {exp_id}")
        print(f"  push={push:.3f}m, lift={lift:.3f}m, freq={freq:.1f}Hz")
        print(f"  Timestamp: {timestamp}")
        print(f"{'='*60}")
        
        # Run cpg_record.py with parameters
        try:
            cmd = [
                sys.executable, "cpg_record.py",
                f"--push={push}",
                f"--lift={lift}",
                f"--freq={freq}",
                f"--out_prefix=cpg_run_{exp_id}"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                print(f"Error running CPG: {result.stderr}")
                return None
            
            print(result.stdout)
            
        except subprocess.TimeoutExpired:
            print(f"Timeout")
            return None
        except Exception as e:
            print(f"Exception: {e}")
            return None
        
        # Analyze results
        analysis = self._analyze_experiment(exp_id)
        
        # Create metadata record
        metadata = {
            "experiment_id": exp_id,
            "timestamp": timestamp,
            "parameters": {
                "push_m": push,
                "lift_m": lift,
                "freq_hz": freq
            },
            "output_files": {
                "com_csv": f"cpg_run_{exp_id}_com.csv",
                "finger_csv": f"cpg_run_{exp_id}_if_finger.csv"
            },
            "analysis": analysis
        }
        
        # Save metadata
        meta_file = self.metadata_dir / f"{exp_id}.json"
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Metadata: {meta_file.name}")
        self.experiments.append(metadata)
        return metadata

    def _analyze_experiment(self, exp_id):
        """Analyze experiment results."""
        
        com_csv = f"cpg_run_{exp_id}_com.csv"
        
        analysis = {"status": "unknown", "metrics": {}}
        
        try:
            if not Path(com_csv).exists():
                analysis["status"] = "file_missing"
                return analysis
            
            with open(com_csv, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            
            if not rows:
                analysis["status"] = "empty"
                return analysis
            
            analysis["status"] = "success"
            
            # Extract metrics
            palm_x_vals = [float(r['palm_x']) for r in rows if r['palm_x']]
            tilt_vals = [float(r['tilt_deg']) for r in rows if r['tilt_deg']]
            
            if palm_x_vals:
                total_advance = palm_x_vals[-1] - palm_x_vals[0]
                num_cycles = len(rows) / 500  # ~500Hz from dt=0.002
                advance_per_cycle = total_advance / num_cycles if num_cycles > 0 else 0
                
                analysis["metrics"]["advance_m"] = round(total_advance, 6)
                analysis["metrics"]["advance_mm_per_cycle"] = round(advance_per_cycle * 1000, 3)
            
            if tilt_vals:
                analysis["metrics"]["avg_tilt_deg"] = round(sum(tilt_vals) / len(tilt_vals), 2)
                analysis["metrics"]["tilt_min_max_deg"] = [round(min(tilt_vals), 2), round(max(tilt_vals), 2)]
            
            analysis["metrics"]["num_samples"] = len(rows)
        
        except Exception as e:
            analysis["status"] = f"error: {e}"
        
        return analysis

    def save_summary(self):
        """Save summary to CSV."""
        
        if not self.experiments:
            print("No experiments")
            return
        
        summary_data = []
        for exp in self.experiments:
            row = {
                "experiment_id": exp["experiment_id"],
                "timestamp": exp["timestamp"],
                "push_m": exp["parameters"]["push_m"],
                "lift_m": exp["parameters"]["lift_m"],
                "freq_hz": exp["parameters"]["freq_hz"],
                "status": exp["analysis"]["status"],
            }
            row.update(exp["analysis"]["metrics"])
            summary_data.append(row)
        
        with open(self.summary_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
            writer.writeheader()
            writer.writerows(summary_data)
        
        print(f"\n{'='*60}")
        print(f"Summary: {self.summary_path.name}")
        print(f"Total experiments: {len(summary_data)}")
        print(f"Metadata directory: {self.metadata_dir.name}/")
        print(f"{'='*60}\n")
        
        for row in summary_data:
            print(f"{row['experiment_id']}")
            print(f"  push={row['push_m']}m lift={row['lift_m']}m freq={row['freq_hz']}Hz")
            if 'advance_mm_per_cycle' in row:
                print(f"  Advance: {row['advance_mm_per_cycle']} mm/cycle")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CPG Parametric Sweep")
    parser.add_argument("--push-sweep", action="store_true")
    parser.add_argument("--lift-sweep", action="store_true")
    parser.add_argument("--freq-sweep", action="store_true")
    parser.add_argument("--single", nargs=3, type=float, metavar=("PUSH", "LIFT", "FREQ"))
    
    args = parser.parse_args()
    
    sweep = CPGSweep()
    
    if args.single:
        push, lift, freq = args.single
        sweep.run_single_experiment(push, lift, freq, "manual")
    elif args.push_sweep:
        for push in [0.015, 0.020, 0.025, 0.030, 0.035]:
            sweep.run_single_experiment(push, 0.030, 1.0, "push_sweep")
    elif args.lift_sweep:
        for lift in [0.020, 0.030, 0.040, 0.050]:
            sweep.run_single_experiment(0.020, lift, 1.0, "lift_sweep")
    elif args.freq_sweep:
        for freq in [0.7, 1.0, 1.5, 2.0]:
            sweep.run_single_experiment(0.020, 0.030, freq, "freq_sweep")
    else:
        print("Use: --push-sweep, --lift-sweep, --freq-sweep, or --single PUSH LIFT FREQ")
        sys.exit(1)

    sweep.save_summary()

if __name__ == "__main__":
    main()
