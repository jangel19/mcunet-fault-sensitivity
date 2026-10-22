"""Render the week 6 figures and verify random summaries against the report."""

import argparse
from collections import defaultdict
import csv
import math
from pathlib import Path
import re
import statistics
import tempfile
import os

# keep teh font cache outside the checkout
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "mcunet-matplotlib"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


LAYERS = (1, 3, 9, 22, 36, 42, 56, 60, 62, 64, 89, 102, 116, 120)
NEUTRAL = "#88949c"
ACCENT = "#277c83"
INK = "#25323a"


def read_cells(path, random):
    cells = defaultdict(list)
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            op, bit = int(row["op_index"]), int(row["bit"])
            n = int(row["n_images"])
            if op not in LAYERS or bit not in range(8) or n != 200:
                raise ValueError(f"Unexpected layer, bit or image count in {path}")
            if float(row["acc_clean"]) != 0.65:
                raise ValueError(f"Expected clean accuracy 130/200 in {path}")
            if random and row["elem_rule"] != "random":
                raise ValueError(f"Expected seeded random draws in {path}")
            mismatch = (int(row["n_catastrophic"]) + int(row["n_due"])) / n
            drop = float(row["acc_clean"]) - float(row["acc_faulted"])
            cells[op, bit].append((mismatch, drop))
    expected = {(op, bit) for op in LAYERS for bit in range(8)}
    if set(cells) != expected or {len(v) for v in cells.values()} != {30 if random else 5}:
        raise ValueError(f"Expected complete balanced week 6 campaign in {path}")
    return cells


def summary(cells, metric):
    # seperate strata, just like the report
    samples = [[row[metric] for row in cell] for cell in cells]
    mean = statistics.mean(statistics.mean(values) for values in samples)
    count = sum(map(len, samples))
    bounds = (0, 1) if metric == 0 else (-0.35, 0.65)
    if all(value == 0 for values in samples for value in values):
        q = 1 - 0.05 ** (1 / count)
        return tuple(100 * x for x in (mean, bounds[0] * q, bounds[1] * q))
    variance = sum(statistics.variance(v) / len(v) for v in samples) / len(samples) ** 2
    margin = 1.96 * math.sqrt(variance)
    if variance == 0:
        margin = (bounds[1] - bounds[0]) * math.sqrt(math.log(40) / (2 * count))
    return tuple(100 * x for x in (mean, mean - margin, mean + margin))


def groups(cells):
    return {
        "all selected cells": list(cells.values()),
        **{f"bit {b}": [cells[op, b] for op in LAYERS] for b in range(8)},
        **{f"op {op}": [cells[op, b] for b in range(8)] for op in LAYERS},
    }


def verify_report(cells, path):
    rows = {}
    for line in path.read_text().splitlines():
        parts = [p.strip() for p in line.split("|")]
        if len(parts) == 7 and parts[1] in groups(cells):
            rows[parts[1]] = parts[2:6]
    for name, selected in groups(cells).items():
        count = sum(map(len, selected))
        if name not in rows or rows[name][:2] != [str(count), str(count * 200)]:
            raise ValueError(f"Report denominator mismatch: {name}")
        for metric in range(2):
            actual = [f"{v:.3f}" for v in summary(selected, metric)]
            reference = re.findall(r"-?\d+\.\d{3}", rows[name][metric + 2])
            if actual != reference:
                raise ValueError(f"Report summary mismatch: {name}, metric {metric}: {actual} != {reference}")
    print("Verified all 23 random groups, both measures and interval endpoints against the report.")


def canvas(title, subtitle, caption, ylabel, xlabel):
    fig, ax = plt.subplots(figsize=(12, 7.5))
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.23, top=0.80)
    fig.text(0.10, 0.94, title, fontsize=21, weight="bold", va="top")
    fig.text(0.10, 0.88, subtitle, fontsize=12, va="top")
    fig.text(0.10, 0.055, caption, fontsize=11, va="bottom", linespacing=1.5)
    ax.set(xlabel=xlabel, ylabel=ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#c6cdd1")
    ax.tick_params(length=0, pad=8)
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="#e1e6e8", linewidth=0.8)
    return fig, ax


def save(fig, out, stem):
    # fixed metadata makes repeat runs byte identical
    fig.savefig(out / f"{stem}.png", dpi=180, metadata={"Software": None})
    fig.savefig(out / f"{stem}.svg", metadata={"Date": None, "Creator": None})
    plt.close(fig)


def bars(cells, out, by_layer=False, metric=0):
    keys = LAYERS if by_layer else range(8)
    selected = [[cells[key, b] for b in range(8)] if by_layer else
                [cells[op, key] for op in LAYERS] for key in keys]
    stats = [summary(group, metric) for group in selected]
    means = [s[0] for s in stats]
    title = "Top-1 mismatch by layer" if by_layer else (
        "Accuracy drop by bit position" if metric else "Top-1 mismatch by bit position")
    denominator = "240 trials / 48,000 trial-image evaluations per layer" if by_layer else (
        "420 trials / 84,000 trial-image evaluations per bit")
    caption = ("Error bars: report's approximate 95% Monte Carlo intervals, using stratified standard errors.\n"
               "Fixed 200 images reused; selected layers and bits weighted equally.")
    if metric:
        caption += " Negative drop means improved accuracy."
    fig, ax = canvas(title, f"Seeded random campaign  |  {denominator}", caption,
                     "Accuracy drop (percentage points)" if metric else "Top-1 mismatch (%)",
                     "Layer (operator index)" if by_layer else "Bit position (7 = sign bit)")
    colors = [ACCENT if (k in (1, 3) if by_layer else k == 7) else NEUTRAL for k in keys]
    ax.bar(range(len(means)), means, color=colors, width=0.68,
           yerr=[[s[0] - s[1] for s in stats], [s[2] - s[0] for s in stats]],
           error_kw={"ecolor": INK, "capsize": 4, "elinewidth": 1.2})
    ax.set_xticks(range(len(means)), [str(k) for k in keys])
    ax.axhline(0, color=INK, linewidth=0.8)
    span = max(s[2] for s in stats) - min(0, min(s[1] for s in stats))
    ax.set_ylim(min(0, min(s[1] for s in stats)) - span * 0.13, max(s[2] for s in stats) + span * 0.20)
    for i, (mean, low, high) in enumerate(stats):
        ax.text(i, high + span * 0.035 if mean >= 0 else low - span * 0.035,
                f"{mean:.3f}", ha="center", va="bottom" if mean >= 0 else "top", fontsize=10)
    stem = "03_mismatch_by_layer" if by_layer else (
        "02_accuracy_drop_by_bit" if metric else "01_mismatch_by_bit")
    save(fig, out, stem)


def heatmap(cells, out):
    fig, ax = canvas("Top-1 mismatch across layers and bits",
                     "Seeded random campaign  |  30 trials / 6,000 trial-image evaluations per cell",
                     "112 cells: 3,360 trials / 672,000 trial-image evaluations. Fixed 200 images reused.\n"
                     "Each cell is the mean trial-level top-1 mismatch; values shown in percent.",
                     "Layer (operator index)", "Bit position (7 = sign bit)")
    fig.set_size_inches(12, 10)
    fig.subplots_adjust(top=0.81, bottom=0.20)
    values = [[summary([cells[op, b]], 0)[0] for b in range(8)] for op in LAYERS]
    cmap = LinearSegmentedColormap.from_list("mismatch", ["#f1f5f5", ACCENT, "#123d42"])
    im = ax.imshow(values, cmap=cmap, vmin=0, aspect="auto")
    ax.grid(False)
    ax.set_xticks(range(8))
    ax.set_yticks(range(len(LAYERS)), [str(op) for op in LAYERS])
    for i, row in enumerate(values):
        for j, value in enumerate(row):
            ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=11,
                    color="white" if value > im.norm.vmax * 0.45 else INK)
    fig.colorbar(im, ax=ax, pad=0.025).set_label("Mean top-1 mismatch (%)")
    save(fig, out, "04_mismatch_heatmap")


def comparison(random, corner, out):
    fig, ax = canvas("Corner vs random: top-1 mismatch by bit",
                     "Per bit: random 420 trials / 84,000 evaluations; corner 70 trials / 14,000 evaluations",
                     "Different sampling designs: rule-weighted corner trials vs seeded random draws.\n"
                     "Not comparable as unbiased estimates. Evaluations are trial-image evaluations on 200 reused images.\n"
                     "Totals: random 3,360 trials / 672,000 evaluations; corner 560 trials / 112,000 evaluations.",
                     "Top-1 mismatch (%)", "Bit position (7 = sign bit)")
    for cells, shift, color, label in ((corner, -0.19, NEUTRAL, "Corner rules"),
                                      (random, 0.19, ACCENT, "Seeded random")):
        means = [summary([cells[op, b] for op in LAYERS], 0)[0] for b in range(8)]
        ax.bar([b + shift for b in range(8)], means, width=0.36, color=color, label=label)
    ax.set_xticks(range(8))
    ax.legend(frameon=False)
    ax.set_ylim(bottom=0)
    save(fig, out, "05_corner_vs_random_by_bit")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--random-csv", type=Path, default=Path("results/random_k30.csv"))
    parser.add_argument("--corner-csv", type=Path, default=Path("results/first_pass_corner.csv"))
    parser.add_argument("--report", type=Path, default=Path("results/random_k30_report.md"))
    parser.add_argument("--out-dir", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    random = read_cells(args.random_csv, random=True)
    corner = read_cells(args.corner_csv, random=False)
    verify_report(random, args.report)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK,
                         "svg.hashsalt": "mcunet-week6", "svg.fonttype": "path",
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    bars(random, args.out_dir)
    bars(random, args.out_dir, metric=1)
    bars(random, args.out_dir, by_layer=True)
    heatmap(random, args.out_dir)
    comparison(random, corner, args.out_dir)
    print(f"Saved five PNG/SVG pairs to {args.out_dir}")


if __name__ == "__main__":
    main()
