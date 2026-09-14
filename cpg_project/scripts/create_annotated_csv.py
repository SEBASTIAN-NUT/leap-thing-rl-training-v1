#!/usr/bin/env python3
"""Create CSV with intermediate calculations for transparency"""
import csv

input_csv = 'results/data/cpg_run_final_if_finger.csv'
output_csv = 'results/data/cpg_run_final_ik_target_annotated.csv'

# データ読み込み
times = []
x_targets = []
z_targets = []

with open(input_csv) as f:
    reader = csv.DictReader(f)
    for i, row in enumerate(reader):
        if i >= 500:  # 1サイクル
            break
        times.append(float(row['t']))
        x_targets.append(float(row['x_target_rel']))
        z_targets.append(float(row['z_target_rel']))

# スケーリング計算
min_x = min(x_targets)
max_x = max(x_targets)
min_z = min(z_targets)
max_z = max(z_targets)

# グラフパラメータ
margin = 100
plot_w = 800
plot_h = 600

# 出力 CSV を作成
with open(output_csv, 'w', newline='') as f:
    writer = csv.writer(f)
    
    # ヘッダー
    writer.writerow([
        't (sec)',
        'x_target_rel (m)',
        'z_target_rel (m)',
        'x_target_mm',
        'z_target_mm',
        '--- 計算パラメータ ---',
        'min_x_mm',
        'max_x_mm',
        'min_z_mm',
        'max_z_mm',
        'plot_width_px',
        'plot_height_px',
        'margin_px',
        '--- グラフ座標 ---',
        'x_pixel',
        'z_pixel (Y軸)',
    ])
    
    # パラメータ行
    writer.writerow([
        '',
        '',
        '',
        '',
        '',
        '',
        f'{min_x*1000:.2f}',
        f'{max_x*1000:.2f}',
        f'{min_z*1000:.2f}',
        f'{max_z*1000:.2f}',
        plot_w,
        plot_h,
        margin,
        '',
        '',
        '',
    ])
    
    # 計算式の説明行
    writer.writerow([
        '',
        '',
        '',
        'x_target_rel * 1000',
        'z_target_rel * 1000',
        '',
        '',
        '',
        '',
        '',
        '',
        '',
        '',
        '計算式:',
        'margin + (x_mm - min_x_mm) / (max_x_mm - min_x_mm) * plot_width',
        'margin + plot_height - (z_mm - min_z_mm) / (max_z_mm - min_z_mm) * plot_height',
    ])
    
    # データ行
    for i in range(len(times)):
        x_mm = x_targets[i] * 1000
        z_mm = z_targets[i] * 1000
        
        # スケーリング計算
        x_pixel = margin + (x_mm - min_x*1000) / (max_x*1000 - min_x*1000) * plot_w
        z_pixel = margin + plot_h - (z_mm - min_z*1000) / (max_z*1000 - min_z*1000) * plot_h
        
        writer.writerow([
            f'{times[i]:.3f}',
            f'{x_targets[i]:.6f}',
            f'{z_targets[i]:.6f}',
            f'{x_mm:.2f}',
            f'{z_mm:.2f}',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            '',
            f'{x_pixel:.1f}',
            f'{z_pixel:.1f}',
        ])

print(f"✓ Created: {output_csv}")
print(f"  Open in Excel and plot X_pixel vs Z_pixel as XY Scatter")
