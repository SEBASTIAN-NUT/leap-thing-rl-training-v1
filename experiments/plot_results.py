#!/usr/bin/env python3
"""
学習結果の一括可視化
全チェックポイントを自動分類し、報酬割合・速度追従性能を比較する
実行: Open_Duck_Playground/.venv/bin/python experiments/plot_results.py

追加機能 (支配的な報酬パラメータを数値で断定するため):
  - 各カテゴリについて、reward/cost の全項目(alive, tracking_*, stand_still,
    orientation, feet_air_time, torques, action_rate, termination等)を
    TFEventsから自動検出し、
      1. <category>__breakdown.png : 全項目を同一スケール(共通y軸)で重ね描き
      2. <category>__share.png     : 項目合計に対する%シェアのスタック図
      3. reward_dominance_summary.csv : run毎の tail値・%シェアを数値化
    を出力する。
  - 注意: 既存の plot_category() の下段(Proportion)グラフは
    reward/{k}(dtなし) を eval/episode_reward(dtあり, sum*dt)で割っており、
    分母と分子の単位が違うため比率が1.0を超えて ylim(-0.05,1.05) で
    クリップされうる(=支配項ほど頭打ちで見分けがつかなくなる)。
    新しい __share.png は reward/cost 項目の合計を分母にするため
    この単位不一致が起きず、0-100%で正しく収まる。
"""
import csv
import re
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

CHECKPOINTS = Path("checkpoints")
OUTPUT_DIR  = Path("analysis_plots")

# ─── フェーズ定義（上から順にマッチ試行）───────────────────────
CATEGORIES = [
    ("p1",    "Phase 1: sigma_lin sweep",        r"p1_siglin_(.+?)_\d{8}_\d{6}"),
    ("p2ang", "Phase 2: sigma_ang sweep",         r"p2_siggang_(.+?)_\d{8}_\d{6}"),
    ("p2a",   "Phase 2a: sigma_lin refined",      r"p2a_sigma_lin_(.+?)_\d{8}_\d{6}"),
    ("p3",    "Phase 3: alive sweep",             r"p3_alive_(.+?)_\d{8}_\d{6}"),
    ("p5ss",  "Phase 5: stand_still",             r"p5_stand_still_(.+?)_\d{8}_\d{6}"),
    ("p5ori", "Phase 5: orientation",             r"p5_orientation_(.+?)_\d{8}_\d{6}"),
    ("p5fat", "Phase 5: feet_air_time",           r"p5_feet_air_time_(.+?)_\d{8}_\d{6}"),
    ("swang", "Sweep: sigma_ang (early)",         r"sweep_reward_config_tracking_sigma_ang_(.+?)_\d{4}_\d{2}_\d{2}_\d+"),
    ("swne",  "Sweep: sigma_ang ne4096",          r"sweep_sigma_ang_(.+)"),
    ("siglin","Sigma_lin (early)",                r"sigma_lin_(.+?)_\d{4}_\d{2}_\d{2}_\d+"),
    ("go1",   "Go1 exact",                        r"go1_exact_(.+)"),
    ("scale", "Scale adjustment",                 r"scale_adj_(.+)"),
    ("sigma025", "Sigma025 fixed",               r"sigma025_fixed_(.+)"),
    ("early", "Early experiments",               r"(2026_\d{2}_\d{2}_\d+.*)"),
]

TAGS = [
    "eval/episode_reward",
    "eval/episode_reward/tracking_lin_vel",
    "eval/episode_reward/tracking_ang_vel",
    "eval/episode_reward/alive",
    "eval/avg_episode_length",
]

# thing_walk.py の state.metrics[f"reward/{k}" if scale>0 else f"cost/{k}"]
# = scale*raw (dtなし) に対応。TensorBoard上では "eval/" が前置される。
# reward/cost の全項目はこの命名規則で統一されているので、項目同士は
# 単位を揃えて直接比較できる(eval/episode_reward は別途 *dt されているので
# 内訳項とは比較しない)。
BREAKDOWN_TAG_RE = re.compile(r"^eval/(reward|cost)/(.+)$")

# ─── TFEvents 読み込み ────────────────────────────────────────────
def read_scalars(event_path: Path) -> dict:
    ea = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))
    result = {}
    for tag in available:
        evs = ea.Scalars(tag)
        result[tag] = (
            np.array([e.step  for e in evs], dtype=float),
            np.array([e.value for e in evs], dtype=float),
        )
    return result

def best_event_file(d: Path):
    files = list(d.glob("events.out.tfevents.*"))
    return max(files, key=lambda p: p.stat().st_size) if files else None

def smooth_ema(v: np.ndarray, alpha=0.25) -> np.ndarray:
    out = np.empty_like(v)
    out[0] = v[0]
    for i in range(1, len(v)):
        out[i] = alpha * v[i] + (1 - alpha) * out[i - 1]
    return out

def breakdown_tags(data: dict) -> list:
    """dataに含まれる reward/cost 内訳タグを項目名でソートして返す"""
    return sorted((t for t in data if BREAKDOWN_TAG_RE.match(t)))

# ─── 全チェックポイント収集 ───────────────────────────────────────
def collect_all() -> dict:
    """category_key -> {label: data}"""
    groups = defaultdict(dict)
    unmatched = []

    for ckpt in sorted(CHECKPOINTS.iterdir()):
        if not ckpt.is_dir():
            continue
        evf = best_event_file(ckpt)
        if evf is None:
            continue

        matched = False
        for cat_key, _, pattern in CATEGORIES:
            m = re.fullmatch(pattern, ckpt.name)
            if m:
                label = m.group(1)
                try:
                    data = read_scalars(evf)
                except Exception as e:
                    print(f"  [WARN] {ckpt.name}: {e}")
                    break
                if not data:
                    break
                # 同ラベルなら長い方を採用
                existing = groups[cat_key].get(label)
                def max_step(d):
                    return max((v[0][-1] for v in d.values() if len(v[0])), default=0)
                if existing is None or max_step(data) > max_step(existing):
                    groups[cat_key][label] = data
                matched = True
                break
        if not matched:
            unmatched.append(ckpt.name)

    if unmatched:
        print(f"未分類 ({len(unmatched)}): {', '.join(unmatched[:5])}{'...' if len(unmatched)>5 else ''}")
    return groups

# ─── プロット (既存: 変更なし) ────────────────────────────────────
def plot_category(title: str, runs: dict, out_path: Path):
    if not runs:
        return

    tag_total  = "eval/episode_reward"
    tag_lin    = "eval/episode_reward/tracking_lin_vel"
    tag_ang    = "eval/episode_reward/tracking_ang_vel"
    tag_alive  = "eval/episode_reward/alive"
    tag_eplen  = "eval/avg_episode_length"

    labels  = sorted(runs.keys())
    colors  = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax_total, ax_track, ax_eplen = axes[0]
    ax_prop_lin, ax_prop_ang, ax_prop_alive = axes[1]

    # ── 絶対値グラフ (上段) ──────────────────────────────────────
    def plot_abs(ax, tag, ylabel, ylim=None):
        has_data = False
        for i, label in enumerate(labels):
            data = runs[label]
            if tag not in data:
                continue
            steps, values = data[tag]
            c = colors[i % len(colors)]
            ax.plot(steps / 1e6, values, alpha=0.15, color=c, linewidth=0.8)
            ax.plot(steps / 1e6, smooth_ema(values), color=c, linewidth=2, label=label)
            has_data = True
        ax.set_xlabel("Timesteps (M)")
        ax.set_title(ylabel, fontsize=10)
        if has_data:
            ax.legend(fontsize=8)
        if ylim:
            ax.set_ylim(ylim)
        ax.grid(alpha=0.3)

    plot_abs(ax_total, tag_total, "Total reward")
    plot_abs(ax_track, tag_lin,   "Tracking lin vel reward")
    plot_abs(ax_eplen, tag_eplen, "Episode length (steps)")

    # ── 割合グラフ (下段) ─────────────────────────────────────────
    def interp_to(src_steps, src_vals, ref_steps):
        if len(src_steps) < 2:
            return np.full_like(ref_steps, np.nan, dtype=float)
        return np.interp(ref_steps, src_steps, src_vals,
                         left=np.nan, right=np.nan)

    def plot_proportion(ax, tag_num, ylabel):
        has_data = False
        for i, label in enumerate(labels):
            data = runs[label]
            if tag_total not in data or tag_num not in data:
                continue
            steps_t, total  = data[tag_total]
            steps_n, numer  = data[tag_num]
            ref = steps_t
            total_i = total
            numer_i = interp_to(steps_n, numer, ref)
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = np.where(total_i > 0.01, numer_i / total_i, np.nan)
            c = colors[i % len(colors)]
            ax.plot(ref / 1e6, ratio, alpha=0.15, color=c, linewidth=0.8)
            ax.plot(ref / 1e6, smooth_ema(np.nan_to_num(ratio)), color=c, linewidth=2, label=label)
            has_data = True
        ax.set_xlabel("Timesteps (M)")
        ax.set_title(ylabel, fontsize=10)
        ax.set_ylim(-0.05, 1.05)
        ax.set_ylabel("Proportion of total reward")
        if has_data:
            ax.legend(fontsize=8)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.grid(alpha=0.3)

    plot_proportion(ax_prop_lin,   tag_lin,   "Lin vel tracking / total")
    plot_proportion(ax_prop_ang,   tag_ang,   "Ang vel tracking / total")
    plot_proportion(ax_prop_alive, tag_alive, "Alive / total")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path}")


# ─── プロット (新規: 全項目の内訳比較) ─────────────────────────────
def plot_breakdown(title: str, runs: dict, out_path: Path) -> bool:
    labels = sorted(runs.keys())
    all_tags = sorted(set(t for lb in labels for t in breakdown_tags(runs[lb])))
    if not all_tags:
        return False

    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{title} - Reward/cost breakdown (shared scale, points)",
                 fontsize=13, fontweight="bold")

    term_colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_tags), 1)))
    tag_color = dict(zip(all_tags, term_colors))

    global_max = 0.0
    for lb in labels:
        data = runs[lb]
        for tag in all_tags:
            if tag in data and len(data[tag][1]):
                global_max = max(global_max, float(np.max(data[tag][1])))

    for ax, lb in zip(axes, labels):
        data = runs[lb]
        for tag in all_tags:
            if tag not in data:
                continue
            steps, values = data[tag]
            term_name = BREAKDOWN_TAG_RE.match(tag).group(2)
            ax.plot(steps / 1e6, smooth_ema(values), color=tag_color[tag],
                     linewidth=2, label=term_name)
        ax.set_title(lb, fontsize=10)
        ax.set_xlabel("Timesteps (M)")
        ax.set_ylim(0, global_max * 1.08 if global_max > 0 else 1)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Reward contribution (scale x raw, points)")
    axes[-1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path}")
    return True


def plot_share(title: str, runs: dict, out_path: Path) -> bool:
    """内訳項の合計に対する%シェアをスタック面グラフで表示 (0-100%に必ず収まる)"""
    labels = sorted(runs.keys())
    all_tags = sorted(set(t for lb in labels for t in breakdown_tags(runs[lb])))
    if not all_tags:
        return False

    n = len(labels)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.5), sharey=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"{title} - Reward/cost share of total (%)",
                 fontsize=13, fontweight="bold")

    term_colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_tags), 1)))
    tag_color = dict(zip(all_tags, term_colors))

    for ax, lb in zip(axes, labels):
        data = runs[lb]
        present = [t for t in all_tags if t in data and len(data[t][0])]
        if not present:
            continue
        ref_steps = max((data[t][0] for t in present), key=len)
        stacks = []
        for tag in present:
            steps, values = data[tag]
            interp = np.interp(ref_steps, steps, values)
            stacks.append(np.clip(interp, 0, None))
        stacks = np.array(stacks)
        total = stacks.sum(axis=0)
        total[total == 0] = 1.0
        shares = stacks / total * 100.0

        term_names = [BREAKDOWN_TAG_RE.match(t).group(2) for t in present]
        colors = [tag_color[t] for t in present]
        ax.stackplot(ref_steps / 1e6, shares, labels=term_names, colors=colors, alpha=0.85)
        ax.set_title(lb, fontsize=10)
        ax.set_xlabel("Timesteps (M)")
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Share of total reward/cost magnitude (%)")
    axes[-1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1.0))

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path}")
    return True


def tail_mean(data: dict, tag: str, frac: float = 0.85) -> float:
    if tag not in data:
        return float("nan")
    v = data[tag][1]
    if len(v) == 0:
        return float("nan")
    return float(np.mean(v[max(0, int(len(v) * frac)):]))


def write_dominance_csv(groups: dict, out_path: Path) -> None:
    """どの報酬/コスト項が支配的かを数値で断定するための CSV。"""
    cat_map = {k: title for k, title, _ in CATEGORIES}
    rows = []
    for cat_key in [k for k, _, _ in CATEGORIES]:
        runs = groups.get(cat_key)
        if not runs:
            continue
        title = cat_map[cat_key]
        for label in sorted(runs.keys()):
            data = runs[label]
            tags = breakdown_tags(data)
            if not tags:
                continue
            tail_vals = {t: tail_mean(data, t) for t in tags}
            total = sum(v for v in tail_vals.values() if not np.isnan(v))
            for t in tags:
                term = BREAKDOWN_TAG_RE.match(t).group(2)
                kind = BREAKDOWN_TAG_RE.match(t).group(1)
                v = tail_vals[t]
                share = (v / total * 100.0) if total > 0 and not np.isnan(v) else float("nan")
                rows.append({
                    "category": title, "label": label, "kind": kind, "term": term,
                    "tail_value": round(v, 4) if not np.isnan(v) else "",
                    "share_pct": round(share, 1) if not np.isnan(share) else "",
                })
    if not rows:
        print("\n[dominance CSV] reward/cost 内訳タグが見つかりません")
        return

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["category", "label", "kind", "term", "tail_value", "share_pct"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n数値サマリー(支配項の断定用) -> {out_path}")

    print("\n" + "=" * 80)
    print("支配的な報酬/コスト項 (run毎、share_pct降順)")
    print("=" * 80)
    seen = set()
    for r in rows:
        key = (r["category"], r["label"])
        if key in seen:
            continue
        seen.add(key)
        same = [x for x in rows if (x["category"], x["label"]) == key]
        same.sort(key=lambda x: -(x["share_pct"] if isinstance(x["share_pct"], float) else -1))
        top = same[0]
        print(f"  {r['category'][:28]:<28} {r['label']:<16} -> {top['kind']}/{top['term']:<16} "
              f"{top['share_pct']}% (value={top['tail_value']})")
    print("=" * 80)


# ─── サマリー ─────────────────────────────────────────────────────
def print_summary(groups: dict):
    cat_map = {k: title for k, title, _ in CATEGORIES}
    tag_r = "eval/episode_reward"
    tag_l = "eval/episode_reward/tracking_lin_vel"
    tag_a = "eval/episode_reward/tracking_ang_vel"

    print("\n" + "=" * 80)
    print(f"{'Category':<25} {'Label':<22} {'Total':>8} {'Lin':>8} {'Ang':>8} {'Steps':>12}")
    print("=" * 80)

    def tail(data, tag, frac=0.85):
        if tag not in data:
            return float("nan")
        v = data[tag][1]
        return float(np.mean(v[max(0, int(len(v) * frac)):]))

    for cat_key in [k for k, _, _ in CATEGORIES]:
        runs = groups.get(cat_key)
        if not runs:
            continue
        title = cat_map[cat_key]
        for label in sorted(runs.keys()):
            data = runs[label]
            ms = max((v[0][-1] for v in data.values() if len(v[0])), default=0)
            print(f"{title[:24]:<25} {label[:21]:<22} "
                  f"{tail(data,tag_r):>8.3f} {tail(data,tag_l):>8.3f} "
                  f"{tail(data,tag_a):>8.3f} {int(ms):>12,}")
    print("=" * 80)

# ─── メイン ──────────────────────────────────────────────────────
def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    print("チェックポイントを収集中...")
    groups = collect_all()

    cat_map = {k: title for k, title, _ in CATEGORIES}
    for cat_key in [k for k, _, _ in CATEGORIES]:
        runs = groups.get(cat_key)
        if not runs:
            continue
        title = cat_map[cat_key]
        print(f"\n[{cat_key}] {title}  ({len(runs)} runs)")
        for lb in sorted(runs.keys()):
            ms = max((v[0][-1] for v in runs[lb].values() if len(v[0])), default=0)
            n_terms = len(breakdown_tags(runs[lb]))
            print(f"  {lb:20s}: {int(ms):,} steps, reward/cost terms={n_terms}")
        safe = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")
        plot_category(title, runs, OUTPUT_DIR / f"{safe}.png")
        plot_breakdown(title, runs, OUTPUT_DIR / f"{safe}__breakdown.png")
        plot_share(title, runs, OUTPUT_DIR / f"{safe}__share.png")

    print_summary(groups)
    write_dominance_csv(groups, OUTPUT_DIR / "reward_dominance_summary.csv")
    print(f"\nプロット保存先: {OUTPUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
