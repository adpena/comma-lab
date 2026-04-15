#!/usr/bin/env python
"""Shannon's Optimal Capacity Allocation Analysis.

Analyzes the comma.ai video compression challenge scoring formula:

    score = 100 * seg + sqrt(10 * pose) + 25 * rate

to determine, at any operating point, where marginal improvement yields
the greatest score reduction.  This is the classic resource-allocation
problem: given finite engineering effort, should we improve PoseNet,
SegNet, or the rate codec?

The answer comes from the partial derivatives (marginal costs):

    d(score)/d(seg)  = 100                     (constant)
    d(score)/d(pose) = sqrt(10) / (2*sqrt(pose))  (decreasing in pose)
    d(score)/d(rate) = 25                      (constant)

Equal-marginal-improvement surfaces are where two partial derivatives
are equal.  This script computes them analytically, plots the surface,
and gives a concrete recommendation at our current operating point.

Usage::

    python experiments/analysis/optimal_allocation.py

Outputs:
    reports/graphs/optimal_allocation.png
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Scoring formula and its derivatives
# ---------------------------------------------------------------------------

def score(seg: float, pose: float, rate: float) -> float:
    """Comma challenge score (lower is better)."""
    return 100.0 * seg + math.sqrt(10.0 * pose) + 25.0 * rate


def d_score_d_seg(seg: float, pose: float, rate: float) -> float:
    """Partial derivative w.r.t. SegNet distortion.  Constant = 100."""
    return 100.0


def d_score_d_pose(seg: float, pose: float, rate: float) -> float:
    """Partial derivative w.r.t. PoseNet distortion.  Decreasing in pose."""
    if pose <= 0:
        return float("inf")
    return math.sqrt(10.0) / (2.0 * math.sqrt(pose))


def d_score_d_rate(seg: float, pose: float, rate: float) -> float:
    """Partial derivative w.r.t. rate.  Constant = 25."""
    return 25.0


# ---------------------------------------------------------------------------
# Analysis at a specific operating point
# ---------------------------------------------------------------------------

def analyze_operating_point(
    seg: float, pose: float, rate: float, label: str = "",
) -> dict:
    """Compute marginal sensitivities and optimal allocation at (seg, pose, rate)."""
    s = score(seg, pose, rate)
    ds = d_score_d_seg(seg, pose, rate)
    dp = d_score_d_pose(seg, pose, rate)
    dr = d_score_d_rate(seg, pose, rate)

    # Which dimension gives the most score reduction per unit improvement?
    # Higher partial derivative = more score reduction per unit decrease.
    # But this is "per absolute unit" -- the STRATEGIC question is which
    # dimension has the most headroom * sensitivity (i.e., which accounts
    # for the most score contribution right now).
    dims = {"seg": ds, "pose": dp, "rate": dr}
    best_marginal_dim = max(dims, key=dims.get)

    # Score contribution: where is the score coming from?
    seg_contrib = 100.0 * seg
    pose_contrib = math.sqrt(10.0 * pose)
    rate_contrib = 25.0 * rate
    contribs = {"seg": seg_contrib, "pose": pose_contrib, "rate": rate_contrib}
    best_dim = max(contribs, key=contribs.get)  # largest contributor = biggest reduction target

    # Leverage ratios
    seg_pose = ds / dp if dp > 0 else float("inf")
    pose_rate = dp / dr
    seg_rate = ds / dr

    result = {
        "label": label,
        "seg": seg,
        "pose": pose,
        "rate": rate,
        "score": s,
        "d_score_d_seg": ds,
        "d_score_d_pose": dp,
        "d_score_d_rate": dr,
        "best_dimension": best_dim,
        "best_marginal_dim": best_marginal_dim,
        "seg_pose_leverage": seg_pose,
        "pose_rate_leverage": pose_rate,
        "seg_rate_leverage": seg_rate,
    }
    return result


def print_analysis(a: dict) -> None:
    """Pretty-print analysis results."""
    label = a["label"] or "Operating point"
    print(f"\n{'=' * 70}")
    print(f"  {label}")
    print(f"{'=' * 70}")
    print(f"  seg={a['seg']:.6f}  pose={a['pose']:.6f}  rate={a['rate']:.6f}")
    print(f"  score = {a['score']:.4f}")
    print()
    print(f"  Marginal sensitivities (d_score / d_component):")
    print(f"    d/d(seg)  = {a['d_score_d_seg']:>10.2f}   (constant)")
    print(f"    d/d(pose) = {a['d_score_d_pose']:>10.2f}   (depends on current pose)")
    print(f"    d/d(rate) = {a['d_score_d_rate']:>10.2f}   (constant)")
    print()
    print(f"  Leverage ratios:")
    print(f"    seg/pose  = {a['seg_pose_leverage']:.2f}x  "
          f"({'SegNet wins' if a['seg_pose_leverage'] > 1 else 'PoseNet wins'})")
    print(f"    pose/rate = {a['pose_rate_leverage']:.2f}x  "
          f"({'PoseNet wins' if a['pose_rate_leverage'] > 1 else 'rate wins'})")
    print(f"    seg/rate  = {a['seg_rate_leverage']:.2f}x  "
          f"({'SegNet wins' if a['seg_rate_leverage'] > 1 else 'rate wins'})")
    print()
    print(f"  >>> LARGEST SCORE CONTRIBUTOR: {a['best_dimension'].upper()}")
    print(f"      (accounts for the most score -- biggest reduction target)")
    print(f"  >>> HIGHEST MARGINAL SENSITIVITY: {a['best_marginal_dim'].upper()}")
    print(f"      (most score reduction per absolute unit decrease)")

    # Score contribution breakdown
    seg_contrib = 100.0 * a["seg"]
    pose_contrib = math.sqrt(10.0 * a["pose"])
    rate_contrib = 25.0 * a["rate"]
    total = seg_contrib + pose_contrib + rate_contrib
    print()
    print(f"  Score contribution breakdown:")
    print(f"    100*seg         = {seg_contrib:.4f}  ({100*seg_contrib/total:.1f}%)")
    print(f"    sqrt(10*pose)   = {pose_contrib:.4f}  ({100*pose_contrib/total:.1f}%)")
    print(f"    25*rate         = {rate_contrib:.4f}  ({100*rate_contrib/total:.1f}%)")

    # Crossover points
    # d/d(pose) == d/d(seg)=100  when sqrt(10)/(2*sqrt(pose)) = 100
    # => pose = 10 / (200)^2 = 10/40000 = 0.00025
    pose_seg_crossover = 10.0 / (2.0 * 100.0) ** 2
    # d/d(pose) == d/d(rate)=25  when sqrt(10)/(2*sqrt(pose)) = 25
    # => pose = 10 / (50)^2 = 10/2500 = 0.004
    pose_rate_crossover = 10.0 / (2.0 * 25.0) ** 2
    print()
    print(f"  Crossover points (where marginal returns equalize):")
    print(f"    PoseNet = SegNet  at pose = {pose_seg_crossover:.6f}")
    print(f"    PoseNet = Rate    at pose = {pose_rate_crossover:.6f}")
    print(f"    Current pose      = {a['pose']:.6f}  "
          f"({'above' if a['pose'] > pose_rate_crossover else 'below'} rate crossover, "
          f"{'above' if a['pose'] > pose_seg_crossover else 'below'} seg crossover)")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_optimal_allocation(operating_points: list[dict], save_path: str) -> None:
    """Generate the optimal allocation surface plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    fig.suptitle(
        "Shannon's Optimal Capacity Allocation\n"
        r"score = 100$\cdot$seg + $\sqrt{10\cdot\mathrm{pose}}$ + 25$\cdot$rate",
        fontsize=15, fontweight="bold", y=0.98,
    )

    # --- Panel 1: Marginal sensitivity as function of pose ---
    ax1 = axes[0, 0]
    pose_range = np.logspace(-5, -0.5, 500)
    dp = np.sqrt(10.0) / (2.0 * np.sqrt(pose_range))
    ax1.loglog(pose_range, dp, "b-", linewidth=2, label=r"$\partial$score/$\partial$pose")
    ax1.axhline(100.0, color="r", linestyle="--", linewidth=1.5, label=r"$\partial$score/$\partial$seg = 100")
    ax1.axhline(25.0, color="g", linestyle="--", linewidth=1.5, label=r"$\partial$score/$\partial$rate = 25")
    # Mark crossover points
    pose_seg_xover = 10.0 / (200.0) ** 2
    pose_rate_xover = 10.0 / (50.0) ** 2
    ax1.axvline(pose_seg_xover, color="r", linestyle=":", alpha=0.5)
    ax1.axvline(pose_rate_xover, color="g", linestyle=":", alpha=0.5)
    # Mark operating points
    for op in operating_points:
        dp_val = op["d_score_d_pose"]
        ax1.plot(op["pose"], dp_val, "ko", markersize=8, zorder=5)
        ax1.annotate(
            op["label"], (op["pose"], dp_val),
            textcoords="offset points", xytext=(10, 5), fontsize=8,
        )
    ax1.set_xlabel("PoseNet distortion (raw)", fontsize=10)
    ax1.set_ylabel(r"$\partial$(score) / $\partial$(component)", fontsize=10)
    ax1.set_title("Marginal Returns vs PoseNet", fontsize=12, fontweight="bold")
    ax1.grid(True, alpha=0.3)
    # Shade regions (before legend so labels appear)
    ax1.fill_between(pose_range, 0, dp, where=(dp > 100), alpha=0.1, color="blue",
                     label="PoseNet dominates SegNet")
    ax1.fill_between(pose_range, 0, np.full_like(pose_range, 100), where=(dp < 100) & (dp > 25),
                     alpha=0.1, color="red", label="SegNet dominates, PoseNet > rate")
    ax1.legend(fontsize=7, loc="upper right")

    # --- Panel 2: Score iso-contours in (seg, pose) space ---
    ax2 = axes[0, 1]
    seg_range = np.linspace(0, 0.01, 200)
    pose_range_2d = np.linspace(0.0001, 0.1, 200)
    SEG, POSE = np.meshgrid(seg_range, pose_range_2d)
    # Fix rate at 0.004 (our current rate)
    SCORE = 100.0 * SEG + np.sqrt(10.0 * POSE) + 25.0 * 0.004
    levels = [0.3, 0.5, 0.7, 0.87, 1.0, 1.33, 1.5, 2.0, 2.5]
    cs = ax2.contour(SEG, POSE, SCORE, levels=levels, cmap="RdYlGn_r")
    ax2.clabel(cs, inline=True, fontsize=7, fmt="%.2f")
    for op in operating_points:
        ax2.plot(op["seg"], op["pose"], "ko", markersize=8, zorder=5)
        ax2.annotate(
            op["label"], (op["seg"], op["pose"]),
            textcoords="offset points", xytext=(5, 5), fontsize=7,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8),
        )
    ax2.set_xlabel("SegNet distortion (raw)", fontsize=10)
    ax2.set_ylabel("PoseNet distortion (raw)", fontsize=10)
    ax2.set_title("Score Iso-contours (rate=0.004)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: Score contribution breakdown bar chart ---
    ax3 = axes[1, 0]
    labels = [op["label"] for op in operating_points]
    seg_contribs = [100.0 * op["seg"] for op in operating_points]
    pose_contribs = [math.sqrt(10.0 * op["pose"]) for op in operating_points]
    rate_contribs = [25.0 * op["rate"] for op in operating_points]
    x = np.arange(len(labels))
    width = 0.5
    ax3.bar(x, seg_contribs, width, label="100*seg", color="tab:red", alpha=0.8)
    ax3.bar(x, pose_contribs, width, bottom=seg_contribs, label=r"$\sqrt{10 \cdot pose}$",
            color="tab:blue", alpha=0.8)
    ax3.bar(x, rate_contribs, width,
            bottom=[s + p for s, p in zip(seg_contribs, pose_contribs)],
            label="25*rate", color="tab:green", alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax3.set_ylabel("Score contribution (score points)", fontsize=10)
    ax3.set_title("Score Decomposition by Component", fontsize=12, fontweight="bold")
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3, axis="y")

    # --- Panel 4: Optimal direction gradient field ---
    ax4 = axes[1, 1]
    seg_r = np.linspace(0.0005, 0.008, 30)
    pose_r = np.linspace(0.0005, 0.06, 30)
    SG, PO = np.meshgrid(seg_r, pose_r)
    # Gradient direction: which dimension to reduce
    # Arrow points in direction of steepest score descent
    # In (seg, pose) space, gradient is (100, sqrt(10)/(2*sqrt(pose)))
    GSEG = np.full_like(SG, 100.0)
    GPOSE = np.sqrt(10.0) / (2.0 * np.sqrt(PO))
    # Normalize for display
    MAG = np.sqrt(GSEG**2 + GPOSE**2)
    q = ax4.quiver(SG, PO, -GSEG / MAG, -GPOSE / MAG, MAG, cmap="coolwarm", alpha=0.7)
    plt.colorbar(q, ax=ax4, label=r"$\|\nabla\mathrm{score}\|$", fraction=0.046)
    for op in operating_points:
        ax4.plot(op["seg"], op["pose"], "ko", markersize=10, zorder=5)
        ax4.annotate(
            op["label"], (op["seg"], op["pose"]),
            textcoords="offset points", xytext=(8, 5), fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.8),
        )
    # Mark the crossover line where d/d(pose) = d/d(seg) = 100
    # pose_xover = 10 / (200)^2 = 0.00025
    ax4.axhline(pose_seg_xover, color="red", linestyle="--", alpha=0.5,
                label=f"pose={pose_seg_xover:.5f}: seg=pose marginal")
    ax4.axhline(pose_rate_xover, color="green", linestyle="--", alpha=0.5,
                label=f"pose={pose_rate_xover:.4f}: pose=rate marginal")
    ax4.set_xlabel("SegNet distortion (raw)", fontsize=10)
    ax4.set_ylabel("PoseNet distortion (raw)", fontsize=10)
    ax4.set_title("Steepest Descent Direction (gradient field)", fontsize=12, fontweight="bold")
    ax4.legend(fontsize=7, loc="upper right")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {save_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Shannon's Optimal Capacity Allocation Analysis")
    print("=" * 70)
    print()
    print("Scoring formula: score = 100*seg + sqrt(10*pose) + 25*rate")
    print()

    # Analytical partial derivatives
    print("Partial derivatives:")
    print("  d(score)/d(seg)  = 100                     [constant]")
    print("  d(score)/d(pose) = sqrt(10)/(2*sqrt(pose)) [decreasing]")
    print("  d(score)/d(rate) = 25                      [constant]")
    print()

    # Key crossover points
    pose_seg_xover = 10.0 / (2.0 * 100.0) ** 2
    pose_rate_xover = 10.0 / (2.0 * 25.0) ** 2
    print("Crossover analysis:")
    print(f"  d/d(pose) = d/d(seg)=100  when pose = {pose_seg_xover:.6f}")
    print(f"  d/d(pose) = d/d(rate)=25  when pose = {pose_rate_xover:.6f}")
    print()
    print("  If pose > 0.004000: improving PoseNet beats improving rate")
    print("  If pose > 0.000250: improving PoseNet beats improving SegNet")
    print("  If pose < 0.000250: improving SegNet or rate is more valuable")

    # Operating points to analyze.
    # Component values are raw distortion inputs to the scoring formula:
    #   score = 100*seg + sqrt(10*pose) + 25*rate
    # Only entries with VERIFIED component-to-score consistency are included.
    # v5 best: 100*0.00217 + sqrt(10*0.031) + 25*0.00401 = 0.874 (matches 0.87)
    # Quantizr: 100*0.00264 + sqrt(10*0.000654) + 25*0.01029 = 0.602 (matches 0.60)
    operating_points = [
        # Our current best (v5 renderer) -- verified
        analyze_operating_point(0.00217, 0.031, 0.00401, "v5 best (0.87)"),
        # Quantizr / mask2mask (PR#53) -- verified from PR component breakdown
        analyze_operating_point(0.00264, 0.000654, 0.01029, "Quantizr (0.60)"),
        # Our target operating point
        analyze_operating_point(0.002, 0.005, 0.004, "TARGET (sub-0.50)"),
        # Theoretical floor
        analyze_operating_point(0.002, 0.00066, 0.004, "FLOOR (Quantizr-PoseNet)"),
    ]

    for op in operating_points:
        print_analysis(op)

    # Summary recommendation
    print("\n" + "=" * 70)
    print("  STRATEGIC RECOMMENDATION")
    print("=" * 70)

    our_best = operating_points[0]  # v5 best
    quantizr = operating_points[1]  # mask2mask

    print(f"\n  Our best score:    {our_best['score']:.2f}")
    print(f"  Quantizr's score:  {quantizr['score']:.2f}")
    print(f"  Gap:               {our_best['score'] - quantizr['score']:.2f}")
    print()
    print("  Score gap decomposition (v5 best vs Quantizr):")
    our_seg = 100.0 * our_best["seg"]
    our_pose = math.sqrt(10.0 * our_best["pose"])
    our_rate = 25.0 * our_best["rate"]
    q_seg = 100.0 * quantizr["seg"]
    q_pose = math.sqrt(10.0 * quantizr["pose"])
    q_rate = 25.0 * quantizr["rate"]
    print(f"    SegNet:  {our_seg:.4f} vs {q_seg:.4f}  delta = {our_seg - q_seg:+.4f}")
    print(f"    PoseNet: {our_pose:.4f} vs {q_pose:.4f}  delta = {our_pose - q_pose:+.4f}")
    print(f"    Rate:    {our_rate:.4f} vs {q_rate:.4f}  delta = {our_rate - q_rate:+.4f}")
    print()

    pose_gap = our_pose - q_pose
    seg_gap = our_seg - q_seg
    rate_gap = our_rate - q_rate
    total_gap = pose_gap + seg_gap + rate_gap

    print(f"  PoseNet accounts for {100*pose_gap/total_gap:.1f}% of the gap")
    print(f"  SegNet  accounts for {100*seg_gap/total_gap:.1f}% of the gap")
    print(f"  Rate    accounts for {100*rate_gap/total_gap:.1f}% of the gap")
    print()
    print(f"  At our current pose={our_best['pose']:.4f}:")
    print(f"    d(score)/d(pose) = {our_best['d_score_d_pose']:.2f}  (decreasing -- low at high pose)")
    print(f"    d(score)/d(seg)  = {our_best['d_score_d_seg']:.2f}  (constant)")
    print(f"    d(score)/d(rate) = {our_best['d_score_d_rate']:.2f}  (constant)")
    print()
    print(f"  Marginal ranking: SegNet ({our_best['d_score_d_seg']:.0f}) > "
          f"Rate ({our_best['d_score_d_rate']:.0f}) > "
          f"PoseNet ({our_best['d_score_d_pose']:.1f})")
    print()
    print(f"  BUT: PoseNet accounts for {100*pose_gap/total_gap:.0f}% of the score gap vs Quantizr.")
    print(f"  Our seg ({our_best['seg']:.5f}) already matches Quantizr ({quantizr['seg']:.5f}).")
    print(f"  Our rate ({our_best['rate']:.5f}) already beats Quantizr ({quantizr['rate']:.5f}).")
    print(f"  PoseNet is the ONLY dimension with large headroom.")
    print()
    print("  VERDICT: PoseNet reduction is the dominant strategy because it is the")
    print("  only dimension where we trail Quantizr. Low marginal sensitivity means")
    print("  diminishing returns per unit reduction, but we need a {:.0f}x reduction".format(
        our_best['pose']/quantizr['pose']))
    print("  to close the gap -- and no amount of seg/rate optimization can compensate.")

    # Generate plot
    project_root = Path(__file__).resolve().parent.parent.parent
    save_path = str(project_root / "reports" / "graphs" / "optimal_allocation.png")
    plot_optimal_allocation(operating_points, save_path)


if __name__ == "__main__":
    main()
