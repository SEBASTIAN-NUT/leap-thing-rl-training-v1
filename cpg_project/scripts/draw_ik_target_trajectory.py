#!/usr/bin/env python3
"""IK target trajectory (no floor, fixed wrist)"""
import csv
from PIL import Image, ImageDraw, ImageFont

csv_file = 'results/data/cpg_run_final_if_finger.csv'

# 1サイクル分
x_target = []
z_target = []

with open(csv_file) as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 500:
            break
        x_target.append(float(row['x_target_rel']) * 1000)  # mm
        z_target.append(float(row['z_target_rel']) * 1000)  # mm

# 画像
width, height = 1000, 800
img = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(img)

margin = 100
plot_w = width - 2*margin
plot_h = height - 2*margin

min_x, max_x = min(x_target), max(x_target)
min_z, max_z = min(z_target), max(z_target)

pad = 10
min_x -= pad
max_x += pad
min_z -= pad
max_z += pad

def scale_x(v):
    return margin + (v - min_x) / (max_x - min_x) * plot_w

def scale_z(v):
    return margin + plot_h - (v - min_z) / (max_z - min_z) * plot_h

# IK ターゲット軌跡
for i in range(len(x_target) - 1):
    x1 = scale_x(x_target[i])
    z1 = scale_z(z_target[i])
    x2 = scale_x(x_target[i+1])
    z2 = scale_z(z_target[i+1])
    draw.line([(x1, z1), (x2, z2)], fill='red', width=3)

# グリッド
draw.line([(margin, margin), (margin, margin+plot_h)], fill='lightgray', width=1)
draw.line([(margin, margin+plot_h), (margin+plot_w, margin+plot_h)], fill='lightgray', width=1)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
except:
    font = ImageFont.load_default()

draw.text((margin, 10), f"IK Target (床なし・手首固定): X={min_x:.0f}~{max_x:.0f}mm, Z={min_z:.0f}~{max_z:.0f}mm", fill='black', font=font)

img.save('results/videos/ik_target_trajectory.png')
print("✓ Saved: ik_target_trajectory.png")
print(f"  X target range: {min_x:.1f} ~ {max_x:.1f} mm")
print(f"  Z target range: {min_z:.1f} ~ {max_z:.1f} mm")
