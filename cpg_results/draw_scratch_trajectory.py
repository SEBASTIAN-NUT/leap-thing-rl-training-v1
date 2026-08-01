#!/usr/bin/env python3
"""
指先の軌跡と手のひら全体の移動を描画（PIL使用）
"""
import csv
from PIL import Image, ImageDraw, ImageFont

# データを読み込む
with open('cpg_run_final_if_finger.csv') as f:
    reader = csv.DictReader(f)
    tip_x_world = []
    tip_z_world = []
    for row in reader:
        tip_x_world.append(float(row['tip_x_world']))
        tip_z_world.append(float(row['tip_z_world']))

with open('cpg_run_final_com.csv') as f:
    reader = csv.DictReader(f)
    com_times = []
    com_x = []
    for row in reader:
        com_times.append(float(row['t']))
        com_x.append(float(row['palm_x']) * 1000)  # mm

# 画像作成
width, height = 1200, 800
img = Image.new('RGB', (width, height), color='white')
draw = ImageDraw.Draw(img)

# フォント
try:
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
except:
    font = ImageFont.load_default()

# 1. 指先の軌跡（左側）
margin = 50
plot_width = (width - 3*margin) // 2
plot_height = height - 2*margin

# スケーリング
min_x = min(tip_x_world)
max_x = max(tip_x_world)
min_z = min(tip_z_world)
max_z = max(tip_z_world)

def scale_x(val):
    return margin + (val - min_x) / (max_x - min_x) * plot_width if max_x > min_x else margin

def scale_z(val):
    return margin + plot_height - (val - min_z) / (max_z - min_z) * plot_height if max_z > min_z else margin

# 指先軌跡を描画
for i in range(len(tip_x_world) - 1):
    x1 = scale_x(tip_x_world[i])
    z1 = scale_z(tip_z_world[i])
    x2 = scale_x(tip_x_world[i+1])
    z2 = scale_z(tip_z_world[i+1])
    draw.line([(x1, z1), (x2, z2)], fill='blue', width=2)

draw.text((margin, height - 30), "Fingertip Trajectory (3D path)", fill='black', font=font)

# 2. 手のひいの移動（右側）
plot_x_offset = width // 2 + margin

# スケーリング
min_t = min(com_times)
max_t = max(com_times)
min_com = min(com_x)
max_com = max(com_x)

def scale_t(val):
    return plot_x_offset + (val - min_t) / (max_t - min_t) * plot_width if max_t > min_t else plot_x_offset

def scale_com(val):
    return margin + plot_height - (val - min_com) / (max_com - min_com) * plot_height if max_com > min_com else margin

# 手のひいの移動を描画
for i in range(len(com_times) - 1):
    x1 = scale_t(com_times[i])
    y1 = scale_com(com_x[i])
    x2 = scale_t(com_times[i+1])
    y2 = scale_com(com_x[i+1])
    draw.line([(x1, y1), (x2, y2)], fill='red', width=2)

# 平均前進速度
total_advance = com_x[-1] - com_x[0]
num_cycles = com_times[-1]
avg_per_cycle = total_advance / num_cycles if num_cycles > 0 else 0
draw.text((plot_x_offset, height - 30), f"Palm Movement ({avg_per_cycle:.2f}mm/cycle)", fill='black', font=font)

img.save('cpg_scratch_trajectory.png')
print("✓ Saved: cpg_scratch_trajectory.png")
print(f"  - Fingertip trajectory: {len(tip_x_world)} points")
print(f"  - Palm movement: {len(com_x)} samples")
print(f"  - Average advance: {avg_per_cycle:.2f} mm/cycle")
