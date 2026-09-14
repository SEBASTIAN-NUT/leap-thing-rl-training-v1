import re
from pathlib import Path

def patch(fn):
    text = Path(fn).read_text()

    # 1. leg_map = {} を ann_map の後に追加
    text = re.sub(r'(ann_map = \{\})', r'\1\nleg_map = {}', text)

    # 2. ax.legend を保存・復元対応版に置換
    def replace_legend(m):
        indent, fs = m.group(1), m.group(2)
        return (
            f'{indent}lp = offsets.get("legend_pos", {{}}).get(str(col))\n'
            f'{indent}if lp:\n'
            f'{indent}    leg = ax.legend(fontsize={fs}, bbox_to_anchor=(lp["x"], lp["y"]), loc="lower left", borderaxespad=0)\n'
            f'{indent}else:\n'
            f'{indent}    leg = ax.legend(fontsize={fs}, loc="upper left")\n'
            f'{indent}leg_map[col] = (ax, leg)'
        )
    text = re.sub(r'( +)ax\.legend\(fontsize=(\d+), loc="upper left"\)', replace_legend, text)

    # 3. _drags 行の後に凡例ドラッグを追加
    text = re.sub(
        r'(_drags = \[ann\.draggable\(True\) for ann in ann_map\.values\(\)\][^\n]*)',
        r'\1\n    for _, lr in leg_map.values(): lr.set_draggable(True)',
        text
    )

    # 4. on_close に凡例位置保存コードを追加
    leg_code = (
        '        try:\n'
        '            renderer = fig.canvas.get_renderer()\n'
        '            lpos = {}\n'
        '            for c, (ax_r, lr) in leg_map.items():\n'
        '                lw = lr.get_window_extent(renderer)\n'
        '                aw = ax_r.get_window_extent(renderer)\n'
        '                lpos[str(c)] = {"x": float((lw.x0-aw.x0)/aw.width),\n'
        '                                "y": float((lw.y0-aw.y0)/aw.height)}\n'
        '            data["legend_pos"] = lpos\n'
        '        except Exception: pass\n'
    )
    text = re.sub(
        r'(        OFFSETS_JSON\.write_text)',
        leg_code + r'        OFFSETS_JSON.write_text',
        text
    )

    Path(fn).write_text(text)
    print(f"Patched: {fn}")

patch('prove_vx_failure.py')
patch('prove_yaw_tracking.py')
