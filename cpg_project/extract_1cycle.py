import csv

# Read CSV
with open('cpg_5sec_if_finger.csv', 'r') as f:
    reader = csv.DictReader(f)
    rows = list(reader)

# Filter t <= 1.0
cycle_rows = [r for r in rows if float(r['t']) <= 1.0]

# Normalize time
t_start = float(cycle_rows[0]['t'])
for r in cycle_rows:
    r['t'] = float(r['t']) - t_start

# Save
with open('cpg_trajectory.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['t', 'if_mcp', 'if_pip'])
    writer.writeheader()
    for r in cycle_rows:
        writer.writerow({'t': f"{r['t']:.4f}", 'if_mcp': r['if_mcp'], 'if_pip': r['if_pip']})

print(f"Saved {len(cycle_rows)} samples (1 cycle)")
