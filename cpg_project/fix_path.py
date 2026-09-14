import sys

# Read the script
with open('cpg_record_with_trajectory.py', 'r') as f:
    lines = f.readlines()

# Find the XML line and insert os.chdir before run()
output = []
for i, line in enumerate(lines):
    if i == 63:  # Before run() call
        output.append("    os.chdir('..')\n")
    output.append(line)

# Add import os at top
if not any('import os' in line for line in output):
    for i, line in enumerate(output):
        if 'import math' in line:
            output.insert(i+1, 'import os\n')
            break

with open('cpg_record_with_trajectory.py', 'w') as f:
    f.writelines(output)
