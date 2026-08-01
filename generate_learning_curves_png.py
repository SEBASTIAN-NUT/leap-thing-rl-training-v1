# -*- coding: utf-8 -*-
#!/usr/bin/env python3

import csv
import os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

PROJECT_DIR = Path(__file__).parent

CSV_FILE = os.environ.get('CSV_FILE')
if not CSV_FILE:
    if (PROJECT_DIR / 'analysis_output/03_learning_curves.csv').exists():
        CSV_FILE = str(PROJECT_DIR / 'analysis_output/03_learning_curves.csv')
    else:
        CSV_FILE = str(PROJECT_DIR / '04_poster_graphs/03_learning_curves.csv')

OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER', str(PROJECT_DIR / 'analysis_output'))

GRAPHS_TO_GENERATE = [
    ("Total Reward (Normalized)", "Timesteps (Million steps)", "Total Reward",
     ["phase2a_sigma_lin_reward", "phase2b_sigma_ang_reward", "phase3_alive_reward"]),

    ("Tracking Lin Vel (Normalized)", "Timesteps (Million steps)", "Lin Vel Component",
     ["phase2a_sigma_lin_lin_vel", "phase2b_sigma_ang_lin_vel", "phase3_alive_lin_vel"]),

    ("Tracking Ang Vel (Normalized)", "Timesteps (Million steps)", "Ang Vel Component",
     ["phase2a_sigma_lin_ang_vel", "phase2b_sigma_ang_ang_vel", "phase3_alive_ang_vel"]),

    ("Alive (Normalized)", "Timesteps (Million steps)", "Alive Component",
     ["phase2a_sigma_lin_alive", "phase2b_sigma_ang_alive", "phase3_alive_alive"]),
]

COLORS = {
    0: (6, 168, 125),
    2: (46, 134, 171),
    4: (114, 9, 183),
    5: (255, 153, 0),
}

def draw_graph(datasets, x_label, y_label, title, output_path):
    width, height = 2600, 1200
    margin_left, margin_right = 400, 120
    margin_top, margin_bottom = 180, 250

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    try:
        title_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 50)
        label_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 60)
        legend_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 40)
        tick_font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 60)
    except:
        title_font = label_font = legend_font = tick_font = ImageFont.load_default()

    min_x, max_x = float('inf'), float('-inf')
    min_y, max_y = float('inf'), float('-inf')

    for ds in datasets:
        for x, y in ds['data']:
            min_x = min(min_x, x)
            max_x = max(max_x, x)
            min_y = min(min_y, y)
            max_y = max(max_y, y)

    if min_x == float('inf'):
        min_x = 0
    if max_x == float('-inf'):
        max_x = 1
    if min_y == float('inf'):
        min_y = 0
    if max_y == float('-inf'):
        max_y = 1

    x_range = max_x - min_x if max_x > min_x else 1
    y_range = max_y - min_y if max_y > min_y else 1
    min_x -= x_range * 0.05
    max_x += x_range * 0.05
    min_y -= y_range * 0.1
    max_y += y_range * 0.1

    def scale_x(x):
        return margin_left + (x - min_x) / (max_x - min_x) * plot_width

    def scale_y(y):
        return margin_top + plot_height - (y - min_y) / (max_y - min_y) * plot_height

    def get_nice_ticks(min_val, max_val, n_ticks=5):
        range_val = max_val - min_val
        if range_val <= 0:
            return [0]
        magnitude = 10 ** max(0, len(str(int(range_val))) - 1)
        if magnitude == 0:
            magnitude = 1
        tick_interval = magnitude / 2
        ticks = []
        tick_val = int(min_val / tick_interval) * tick_interval
        while tick_val <= max_val:
            if tick_val >= min_val:
                ticks.append(tick_val)
            tick_val += tick_interval
        return ticks if ticks else [min_val, max_val]

    y_ticks = get_nice_ticks(min_y, max_y)
    for y_val in y_ticks:
        y_px = scale_y(y_val)
        draw.line([(margin_left, y_px), (width - margin_right, y_px)],
                 fill=(220, 220, 220), width=2)
        label = f"{int(y_val)}" if y_val == int(y_val) else f"{y_val:.1f}"
        draw.text((margin_left - 80, y_px - 30), label, fill=(40, 40, 40), font=tick_font)

    x_ticks = get_nice_ticks(min_x, max_x, n_ticks=4)
    for x_val in x_ticks:
        x_px = scale_x(x_val)
        draw.line([(x_px, margin_top), (x_px, margin_top + plot_height)],
                 fill=(220, 220, 220), width=2)
        label = f"{int(x_val)}" if x_val == int(x_val) else f"{x_val:.0f}"
        draw.text((x_px - 30, margin_top + plot_height + 25), label, fill=(40, 40, 40), font=tick_font)

    draw.rectangle([(margin_left, margin_top), (width - margin_right, margin_top + plot_height)],
                   outline=(40, 40, 40), width=4)

    plot_left = margin_left
    plot_right = width - margin_right
    plot_top = margin_top
    plot_bottom = margin_top + plot_height

    for ds in datasets:
        points = []
        for x, y in ds['data']:
            px = scale_x(x)
            py = scale_y(y)
            points.append((px, py))

        for i in range(len(points) - 1):
            x1, y1 = points[i]
            x2, y2 = points[i + 1]

            if (plot_left <= x1 <= plot_right and plot_top <= y1 <= plot_bottom and
                plot_left <= x2 <= plot_right and plot_top <= y2 <= plot_bottom):
                draw.line([(x1, y1), (x2, y2)], fill=ds['color'], width=5)

        for x, y in points:
            if plot_left <= x <= plot_right and plot_top <= y <= plot_bottom:
                dark_color = tuple(max(0, c - 100) for c in ds['color'])
                draw.ellipse([(x - 12, y - 12), (x + 12, y + 12)],
                            fill=dark_color, outline=dark_color, width=2)

    legend_x = margin_left + 60
    legend_y = margin_top + 80
    for i, ds in enumerate(datasets):
        y_pos = legend_y + i * 100

        line_x_start = legend_x
        line_x_end = legend_x + 50
        line_y = y_pos + 15
        draw.line([(line_x_start, line_y), (line_x_end, line_y)],
                  fill=ds['color'], width=5)

        marker_x = (line_x_start + line_x_end) // 2
        dark_color = tuple(max(0, c - 100) for c in ds['color'])
        draw.ellipse([(marker_x - 8, line_y - 8), (marker_x + 8, line_y + 8)],
                    fill=dark_color, outline=dark_color, width=2)

        label_text = ds['label']
        draw.text((legend_x + 70, y_pos), label_text, fill=(30, 30, 30), font=legend_font)

    draw.text((150, 50), title, fill=(20, 20, 20), font=title_font)
    draw.text((margin_left + plot_width // 2 - 100, margin_top + plot_height + 100),
             x_label, fill=(20, 20, 20), font=label_font)

    img.save(output_path)
    print(f"Saved: {output_path}")

csv_path = Path(CSV_FILE)
if not csv_path.exists():
    csv_path = Path.cwd() / CSV_FILE

print(f"CSV path: {csv_path}")

X_COLUMN_INDEX = 0

for title, x_label, y_label, col_names in GRAPHS_TO_GENERATE:
    print(f"\nGenerating: {title}")

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        data_dict = {col: [] for col in col_names}

        for row in reader:
            try:
                x = float(row[list(row.keys())[X_COLUMN_INDEX]])
                for col_name in col_names:
                    if col_name in row and row[col_name]:
                        y = float(row[col_name])
                        data_dict[col_name].append((x, y))
            except (ValueError, KeyError):
                continue

    datasets = []
    color_map = {
        "phase2a_sigma_lin": (46, 134, 171),
        "phase2b_sigma_ang": (255, 100, 100),
        "phase3_alive": (230, 57, 70),
    }

    for col_name in col_names:
        if data_dict[col_name]:
            if "phase2a_sigma_lin" in col_name:
                color = color_map["phase2a_sigma_lin"]
                label = "Phase 2a (Sigma Lin)"
            elif "phase2b_sigma_ang" in col_name:
                color = color_map["phase2b_sigma_ang"]
                label = "Phase 2b (Sigma Ang)"
            elif "phase3_alive" in col_name:
                color = color_map["phase3_alive"]
                label = "Phase 3 (Alive)"
            else:
                color = (100, 100, 100)
                label = col_name

            datasets.append({
                'data': data_dict[col_name],
                'color': color,
                'label': label
            })

    if not datasets:
        print(f"Warning: No data for {title}")
        continue

    output_filename = title.lower().replace(' ', '_').replace('(', '').replace(')', '') + '.png'
    output_path = Path(OUTPUT_FOLDER) / output_filename

    print(f"  -> {output_path}")
    draw_graph(datasets, x_label, y_label, title, output_path)

print(f"\nAll graphs generated!")
