#!/usr/bin/env python3
"""Hand-relative fingertip trajectory (fixed wrist)"""
import csv
from PIL import Image, ImageDraw, ImageFont

# ワーキングディレクトリが cpg_project/ なので
csv_file = 'results/data/cpg_run_final_if_finger.csv'

with open(csv_file) as f:
    reader = csv.DictReader(f)
    tip_x_rel = []
    tip_z_rel = []
    for row in reader:
        tip_x_rel.append(float(row['tip_x_rel']) * 1000)  # mm
        tip_z_rel.append(float(row['tip_z_rel']) * 1000)  # mm

# 画像作成
width, height = 800, 600
img = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(img)

# スケーリング
margin = 50
plot_w = width - 2*margin
plot_h = height - 2*margin

min_x, max_x = min(tip_x_rel), max(tip_x_rel)
min_z, max_z = min(tip_z_rel), max(tip_z_rel)

def scale_x(v):
    return margin + (v - min_x) / (max_x - min_x) * plot_w if max_x > min_x else margin

def scale_z(v):
    return margin + plot_h - (v - min_z) / (max_z - min_z) * plot_h if max_z > min_z else margin

# 軌跡描画
for i in range(len(tip_x_rel) - 1):
    x1 = scale_x(tip_x_rel[i])
    z1 = scale_z(tip_z_rel[i])
    x2 = scale_x(tip_x_rel[i+1])
    z2 = scale_z(tip_z_rel[i+1])
    draw.line([(x1, z1), (x2, z2)], fill='blue', width=2)

try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
except:
    font = ImageFont.load_default()

draw.text((margin, height - 30), f"Hand-relative fingertip path", fill='black', font=font)
draw.text((margin, 10), f"X: {min_x:.1f}~{max_x:.1f}mm, Z: {min_z:.1f}~{max_z:.1f}mm", fill='black', font=font)

img.save('results/videos/hand_relative_trajectory.png')
print("✓ Saved: hand_relative_trajectory.png")
print(f"  X range: {min_x:.1f} ~ {max_x:.1f} mm")
print(f"  Z range: {min_z:.1f} ~ {max_z:.1f} mm")
