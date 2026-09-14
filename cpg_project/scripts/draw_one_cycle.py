#!/usr/bin/env python3
"""Single cycle trajectory"""
import csv
from PIL import Image, ImageDraw, ImageFont

csv_file = 'results/data/cpg_run_final_if_finger.csv'

# 1サイクル分を抽出（500行 = 1秒）
tip_x_rel = []
tip_z_rel = []

with open(csv_file) as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 500:  # 1サイクルまで
            break
        tip_x_rel.append(float(row['tip_x_rel']) * 1000)  # mm
        tip_z_rel.append(float(row['tip_z_rel']) * 1000)  # mm

# 画像
width, height = 1000, 800
img = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(img)

margin = 100
plot_w = width - 2*margin
plot_h = height - 2*margin

min_x, max_x = min(tip_x_rel), max(tip_x_rel)
min_z, max_z = min(tip_z_rel), max(tip_z_rel)

# 余裕を持たせる
pad = 10
min_x -= pad
max_x += pad
min_z -= pad
max_z += pad

def scale_x(v):
    return margin + (v - min_x) / (max_x - min_x) * plot_w

def scale_z(v):
    return margin + plot_h - (v - min_z) / (max_z - min_z) * plot_h

# 軌跡（段階的に色を変えて）
for i in range(len(tip_x_rel) - 1):
    x1 = scale_x(tip_x_rel[i])
    z1 = scale_z(tip_z_rel[i])
    x2 = scale_x(tip_x_rel[i+1])
    z2 = scale_z(tip_z_rel[i+1])
    
    # 進行に従って色を変える
    ratio = i / len(tip_x_rel)
    if ratio < 0.5:
        color = (0, 0, int(255 * (1 - ratio*2)))  # 青→紫
    else:
        color = (int(255 * (ratio-0.5)*2), 0, 255)  # 紫→赤
    
    draw.line([(x1, z1), (x2, z2)], fill=color, width=3)

# グリッド（スケール確認用）
draw.line([(margin, margin), (margin, margin+plot_h)], fill='gray', width=1)
draw.line([(margin, margin+plot_h), (margin+plot_w, margin+plot_h)], fill='gray', width=1)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
except:
    font = ImageFont.load_default()

draw.text((margin, 10), f"One cycle: X={min_x:.0f}~{max_x:.0f}mm, Z={min_z:.0f}~{max_z:.0f}mm", fill='black', font=font)
draw.text((margin, 35), "Blue→Red: 時間経過", fill='black', font=font)

img.save('results/videos/one_cycle_trajectory.png')
print("✓ Saved: one_cycle_trajectory.png")
print(f"  Cycle time: {len(tip_x_rel)} samples")
print(f"  X: {min_x:.1f} ~ {max_x:.1f} mm")
print(f"  Z: {min_z:.1f} ~ {max_z:.1f} mm")
