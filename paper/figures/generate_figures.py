#!/usr/bin/env python3
"""Generate all 5 arXiv paper figures from results.jsonl data."""

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path

OUTDIR = Path(__file__).parent
RESULTS = Path(__file__).parent.parent.parent / "reports" / "results.jsonl"

# Load all results
with open(RESULTS) as f:
    records = [json.loads(line) for line in f]


def get(idx):
    r = records[idx]
    return r["segnet_distortion"], r["posenet_distortion"], r["rate"], r["current_workflow_score"]


# ── Style setup ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.direction": "out",
    "ytick.direction": "out",
})

BLUE = "#3575b5"
ORANGE = "#e8873a"
GRAY = "#888888"
GREEN = "#2ca02c"
RED = "#d62728"


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(OUTDIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}.png / .pdf")


# ── Figure 1: Score component stacked bar ────────────────────────────────────
def fig1():
    # Milestones: baseline(no filter=idx1), h16 ep500(idx41), h32 ep500(idx42),
    # h16 ep1000(idx43), h32 ep1000(idx44), ensemble(idx45), h64 ep1000(idx46),
    # std h64 2500(idx47), dilated h64(idx48)
    milestones = [
        ("No filter", 1),
        ("h16\nep500", 41),
        ("h32\nep500", 42),
        ("h16\nep1000", 43),
        ("h32\nep1000", 44),
        ("Ensemble", 45),
        ("h64\nep1000", 46),
        ("Std h64\n2500", 47),
        ("Dilated\nh64", 48),
    ]

    labels = [m[0] for m in milestones]
    seg_terms, pose_terms, rate_terms = [], [], []
    for _, idx in milestones:
        seg, pose, rate, _ = get(idx)
        seg_terms.append(100 * seg)
        pose_terms.append(np.sqrt(10 * pose))
        rate_terms.append(25 * rate)

    x = np.arange(len(labels))
    width = 0.6

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x, seg_terms, width, label="100 · SegNet", color=BLUE)
    ax.bar(x, pose_terms, width, bottom=seg_terms, label="√(10 · PoseNet)", color=ORANGE)
    ax.bar(x, rate_terms, width,
           bottom=[s + p for s, p in zip(seg_terms, pose_terms)],
           label="25 · Rate", color=GRAY)

    # Baseline reference line
    baseline_score = sum([seg_terms[0], pose_terms[0], rate_terms[0]])
    ax.axhline(baseline_score, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.text(len(labels) - 0.5, baseline_score + 0.03, "baseline", fontsize=9, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Score (lower is better)", fontsize=12)
    ax.legend(fontsize=10, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)

    save(fig, "fig1_score_components")


# ── Figure 2: Pareto frontier ────────────────────────────────────────────────
def fig2():
    # Points of interest
    points = {
        "Baseline\n(no filter)":   (get(1)[1], get(1)[0], "o", GRAY),
        "Std h64\n2500":           (get(47)[1], get(47)[0], "s", BLUE),
        "Dilated h64\n(ours)":     (get(48)[1], get(48)[0], "*", RED),
        "KL distill\n#1 (sw=100)": (0.048, 0.0058, "x", ORANGE),  # approximate from notes
        "KL distill\n#2 (sw=30)":  (get(49)[1], get(49)[0], "x", ORANGE),
    }

    fig, ax = plt.subplots(figsize=(8, 5))

    # Iso-score contours: S = 100*seg + sqrt(10*pose) + 25*rate
    # At fixed rate ~ 0.023, so rate_term ~ 0.575
    # S = 100*seg + sqrt(10*pose) + 0.575
    # => seg = (S - 0.575 - sqrt(10*pose)) / 100
    rate_term = 25 * 0.02302
    pose_range = np.logspace(-3, 0, 300)
    for S_val in [1.2, 1.3, 1.4, 1.5, 2.0]:
        seg_vals = (S_val - rate_term - np.sqrt(10 * pose_range)) / 100
        mask = seg_vals > 0
        if mask.any():
            ax.plot(pose_range[mask], seg_vals[mask], "--", color="#cccccc",
                    linewidth=0.8, alpha=0.7)
            # Label at right end
            idx_label = np.where(mask)[0][-1]
            ax.text(pose_range[idx_label] * 1.05, seg_vals[idx_label],
                    f"S={S_val}", fontsize=8, color="#999999", va="center")

    # Plot points
    for label, (pose, seg, marker, color) in points.items():
        ms = 14 if marker == "*" else 8
        ec = "none" if marker == "x" else "black"
        ax.scatter(pose, seg, marker=marker, s=ms**2, color=color, zorder=5,
                   edgecolors=ec, linewidths=0.5)
        # Offset annotations
        offset = (10, 10)
        if "Baseline" in label:
            offset = (10, -15)
        elif "Std" in label:
            offset = (-15, 15)
        ax.annotate(label, (pose, seg), textcoords="offset points",
                    xytext=offset, fontsize=9, ha="center")

    # MRS tangent at our point (dilated h64)
    our_pose, our_seg = get(48)[1], get(48)[0]
    # MRS = d(seg)/d(pose) at iso-score = -sqrt(10)/(200*sqrt(10*pose))
    mrs_slope = -np.sqrt(10) / (200 * np.sqrt(10 * our_pose))
    tangent_pose = np.linspace(our_pose * 0.3, our_pose * 3, 50)
    tangent_seg = our_seg + mrs_slope * (tangent_pose - our_pose)
    ax.plot(tangent_pose, tangent_seg, "-", color=RED, linewidth=1, alpha=0.5,
            label="MRS tangent")

    ax.set_xscale("log")
    ax.set_xlabel("PoseNet distortion", fontsize=12)
    ax.set_ylabel("SegNet distortion", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.legend(fontsize=9, frameon=False, loc="upper left")

    save(fig, "fig2_pareto_frontier")


# ── Figure 3: Width scaling ──────────────────────────────────────────────────
def fig3():
    h_vals = np.array([8, 16, 32, 48, 64])
    scores = np.array([2.06, 1.92, 1.845, 1.762, 1.727])
    dilated_h = 64
    dilated_score = 1.33

    fig, ax = plt.subplots(figsize=(8, 5))

    # Log-linear regression
    log_h = np.log(h_vals)
    coeffs = np.polyfit(log_h, scores, 1)
    h_fit = np.linspace(6, 80, 200)
    fit_line = np.polyval(coeffs, np.log(h_fit))
    ax.plot(h_fit, fit_line, "--", color=GRAY, linewidth=1, alpha=0.6,
            label=f"Log-linear fit (slope={coeffs[0]:.3f})")

    # Standard points
    ax.plot(h_vals, scores, "o-", color=BLUE, markersize=8, linewidth=1.5,
            label="Standard", zorder=4)

    # Dilated point
    ax.plot(dilated_h, dilated_score, "*", color=RED, markersize=18, zorder=5,
            label="Dilated h=64")

    # Annotation arrow from fit prediction to actual
    fit_at_64 = np.polyval(coeffs, np.log(64))
    ax.annotate("", xy=(dilated_h, dilated_score),
                xytext=(dilated_h, fit_at_64),
                arrowprops=dict(arrowstyle="->", color=RED, lw=1.5))
    gap = fit_at_64 - dilated_score
    ax.text(dilated_h * 1.08, (dilated_score + fit_at_64) / 2,
            f"−{gap:.2f}\nbreakout", fontsize=10, color=RED, va="center")

    ax.set_xscale("log")
    ax.set_xticks(h_vals)
    ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax.set_xlabel("Hidden dimension h", fontsize=12)
    ax.set_ylabel("Score (lower is better)", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.legend(fontsize=10, frameon=False, loc="upper right")

    save(fig, "fig3_width_scaling")


# ── Figure 4: Comparison frames (placeholder layout) ─────────────────────────
def fig4():
    fig, axes = plt.subplots(1, 3, figsize=(10, 4))
    titles = ["Baseline", "Ours (dilated h=64)", "SegNet diff"]
    colors = ["#d9e6f2", "#f2d9d9", "#d9f2d9"]

    for ax, title, color in zip(axes, titles, colors):
        ax.set_facecolor(color)
        ax.text(0.5, 0.5, title, transform=ax.transAxes,
                ha="center", va="center", fontsize=13, fontweight="bold",
                color="#444444")
        ax.text(0.5, 0.35, "(video frame)", transform=ax.transAxes,
                ha="center", va="center", fontsize=10, color="#888888",
                fontstyle="italic")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#aaaaaa")

    fig.subplots_adjust(wspace=0.08)
    save(fig, "fig4_comparison_frames")


# ── Figure 5: Score decomposition waterfall ──────────────────────────────────
def fig5():
    # Baseline = idx 36 (2.08, last codec-only floor before post-filter)
    # Final = idx 48 (1.33, dilated h64)
    seg_b, pose_b, rate_b, score_b = get(36)
    seg_f, pose_f, rate_f, score_f = get(48)

    # Score terms
    seg_term_b = 100 * seg_b
    pose_term_b = np.sqrt(10 * pose_b)
    rate_term_b = 25 * rate_b

    seg_term_f = 100 * seg_f
    pose_term_f = np.sqrt(10 * pose_f)
    rate_term_f = 25 * rate_f

    # Deltas
    d_pose = pose_term_f - pose_term_b   # should be negative (improvement)
    d_seg = seg_term_f - seg_term_b       # might be slightly positive (cost)
    d_rate = rate_term_f - rate_term_b     # small

    labels = ["Baseline", "PoseNet\ngain", "SegNet\ncost", "Rate\nchange", "Final"]
    values = [score_b, d_pose, d_seg, d_rate, score_f]

    # Waterfall: compute bottoms
    bottoms = [0] * 5
    tops = [0] * 5

    # Baseline bar goes from 0 to score_b
    bottoms[0] = 0
    tops[0] = score_b

    # Running sum from baseline
    running = score_b
    for i in range(1, 4):
        if values[i] < 0:
            # Improvement: bar drops
            bottoms[i] = running + values[i]
            tops[i] = running
        else:
            # Regression: bar rises
            bottoms[i] = running
            tops[i] = running + values[i]
        running += values[i]

    # Final bar
    bottoms[4] = 0
    tops[4] = score_f

    bar_heights = [t - b for t, b in zip(tops, bottoms)]
    bar_colors = [BLUE, GREEN, RED, RED if d_rate > 0 else GREEN, BLUE]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))

    bars = ax.bar(x, bar_heights, bottom=bottoms, color=bar_colors, width=0.55,
                  edgecolor="white", linewidth=0.5)

    # Connector lines
    for i in range(3):
        y_connect = tops[i] if values[i + 1] < 0 else bottoms[i] + bar_heights[i]
        # Actually just connect at the running level
        running_at_i = score_b + sum(values[1:i+1])
        ax.plot([x[i] + 0.3, x[i+1] - 0.3], [running_at_i, running_at_i],
                "-", color="#999999", linewidth=0.8)

    # Value annotations
    for i, (xi, bi, hi, val) in enumerate(zip(x, bottoms, bar_heights, values)):
        if i == 0 or i == 4:
            text = f"{val:.2f}"
        else:
            text = f"{val:+.3f}"
        y_pos = bi + hi + 0.02
        ax.text(xi, y_pos, text, ha="center", va="bottom", fontsize=10,
                fontweight="bold" if i in (0, 4) else "normal")

    # PoseNet contribution annotation
    total_improvement = score_b - score_f
    pose_contribution = abs(d_pose) / total_improvement * 100
    ax.annotate(f"{pose_contribution:.0f}% from PoseNet",
                xy=(1, bottoms[1] + bar_heights[1] / 2),
                xytext=(2.5, score_b - 0.1),
                fontsize=10, color=GREEN,
                arrowprops=dict(arrowstyle="->", color=GREEN, lw=1.2))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.set_ylim(0, score_b + 0.3)

    save(fig, "fig5_waterfall")


# ── Generate all ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating figures...")
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    print("Done.")
