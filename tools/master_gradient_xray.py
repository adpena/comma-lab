# SPDX-License-Identifier: MIT
"""Master-gradient xray visualization tool.

Per Cable D D4 (task #797) batched 2026-05-19, lane
`lane_cable_d_master_gradient_extension_batch_20260519`. Sister of the
canonical producer `tools/extract_master_gradient.py` + canonical consumer
catalog `tac.master_gradient_consumers`.

5 plot types per the master-gradient + xray follow-up research (task #874):

  1. per-pair gradient distribution histogram (per-pair |g| L1 distribution
     across all pairs)
  2. per-byte sensitivity heatmap (N_bytes × 3 axes; one row per top-K byte)
  3. cumulative gradient-by-rank curve (rank vs cumulative sensitivity)
  4. cross-substrate gradient correlation matrix (when multiple anchors)
  5. Wyner-Ziv layer-aware gradient flow (per-substrate-section gradient
     breakdown if substrate parser declares sections)

Per CLAUDE.md "no /tmp paths in persisted artifacts": the caller MUST pass an
explicit ``--output``. Per "MPS auth eval is NOISE" + Catalog #192: anchors
with non-authoritative grades render with explicit watermark.

Per CLAUDE.md "Apples-to-apples evidence discipline": every figure carries an
``[axis-tag]`` watermark in the title matching the master-gradient anchor's
``measurement_axis`` field.

Usage::

    .venv/bin/python tools/master_gradient_xray.py \\
        --archive-sha 6bae0201abcd... \\
        --plot all \\
        --output reports/master_gradient_xray/<archive>/

    .venv/bin/python tools/master_gradient_xray.py --plot per_pair_distribution \\
        --archive-sha 6bae0201abcd... --output reports/x/per_pair.png

Plot types accepted by ``--plot``:
  - ``per_pair_distribution`` (plot 1)
  - ``per_byte_heatmap`` (plot 2)
  - ``cumulative_by_rank`` (plot 3)
  - ``cross_substrate_correlation`` (plot 4 — requires ``--archive-sha`` repeated)
  - ``wyner_ziv_flow`` (plot 5)
  - ``all`` (emit all 5 into ``--output`` directory)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np  # noqa: E402

from tac.master_gradient_consumers import (  # noqa: E402
    load_aggregate_gradient_from_anchor,
    load_per_pair_gradient_from_anchor,
)


CANONICAL_PLOTS = (
    "per_pair_distribution",
    "per_byte_heatmap",
    "cumulative_by_rank",
    "cross_substrate_correlation",
    "wyner_ziv_flow",
    "all",
)

AXIS_LABELS = ("seg", "pose", "rate")


def _ensure_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SystemExit(
            f"matplotlib not available ({exc}); install with "
            "`uv pip install matplotlib`"
        )


def _watermark_for_anchor(anchor: dict) -> str:
    """Render canonical [axis-tag] watermark per CLAUDE.md axis-discipline."""
    axis = anchor.get("measurement_axis", "unknown")
    hw = anchor.get("measurement_hardware", "unknown")
    grade = anchor.get("evidence_grade", "")
    # MPS / macOS-CPU advisory tags become explicit watermark per Catalog #192
    if "advisory" in str(grade).lower() or "mps" in str(hw).lower():
        return f"[advisory: {axis} on {hw}]"
    return f"[{axis} on {hw}]"


def _short_sha(sha: str | None) -> str:
    if not isinstance(sha, str) or len(sha) < 8:
        return "unknown"
    return sha[:12]


# ──────────────────────────────────────────────────────────────────────────── #
# Plot 1 — per-pair gradient distribution                                       #
# ──────────────────────────────────────────────────────────────────────────── #


def plot_per_pair_distribution(
    per_pair_gradient: np.ndarray,
    anchor: dict,
    output_path: Path,
) -> None:
    """Histogram per-pair |g| L1 distribution across all pairs (per axis).

    Reveals per-pair leverage spread: a tight distribution = pairs are
    equally important; a long tail = a few pairs dominate the gradient
    signal (canonical "hard-pair" pattern per consumer 5).
    """
    plt = _ensure_matplotlib()
    n_bytes, n_pairs, _ = per_pair_gradient.shape
    # Per-pair L1 magnitude per axis: (n_pairs, 3)
    per_pair_l1 = np.abs(per_pair_gradient).sum(axis=0)  # (n_pairs, 3)
    watermark = _watermark_for_anchor(anchor)
    sha_short = _short_sha(anchor.get("archive_sha256"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis_idx, (ax, label) in enumerate(zip(axes, AXIS_LABELS)):
        values = per_pair_l1[:, axis_idx]
        ax.hist(values, bins=40, color=("tab:blue", "tab:orange", "tab:green")[axis_idx])
        ax.set_title(f"{label}: per-pair |g| L1 distribution")
        ax.set_xlabel("L1 magnitude")
        ax.set_ylabel("pair count")
        ax.axvline(
            float(np.median(values)),
            color="red",
            linestyle="--",
            label=f"median={np.median(values):.3g}",
        )
        ax.legend(loc="upper right", fontsize=8)
    fig.suptitle(
        f"Per-pair gradient distribution {watermark}  "
        f"archive={sha_short}  n_pairs={n_pairs}  n_bytes={n_bytes}",
        fontsize=11,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────── #
# Plot 2 — per-byte sensitivity heatmap                                         #
# ──────────────────────────────────────────────────────────────────────────── #


def plot_per_byte_heatmap(
    aggregate_gradient: np.ndarray,
    anchor: dict,
    output_path: Path,
    top_k: int = 128,
) -> None:
    """Heatmap top-K bytes (rows) × 3 axes (cols) of aggregate sensitivity.

    Identifies the canonical leverage points per axis. The heatmap shows
    where each axis's signal concentrates: a few hot rows = highly
    concentrated leverage; broad heat = distributed sensitivity.
    """
    plt = _ensure_matplotlib()
    n_bytes = aggregate_gradient.shape[0]
    abs_grad = np.abs(aggregate_gradient)
    per_byte_l1 = abs_grad.sum(axis=1)
    top_k_clamped = min(top_k, n_bytes)
    top_indices = np.argsort(-per_byte_l1)[:top_k_clamped]
    heat = abs_grad[top_indices]  # (top_k, 3)
    # Normalize per axis so each axis self-contained
    heat_norm = heat / np.maximum(heat.max(axis=0, keepdims=True), 1e-30)

    watermark = _watermark_for_anchor(anchor)
    sha_short = _short_sha(anchor.get("archive_sha256"))

    fig, ax = plt.subplots(figsize=(6, 12))
    im = ax.imshow(heat_norm, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_xticks(range(3))
    ax.set_xticklabels(AXIS_LABELS)
    ax.set_xlabel("score axis")
    ax.set_ylabel(f"byte rank (top {top_k_clamped})")
    ax.set_title(
        f"Per-byte sensitivity heatmap {watermark}\n"
        f"archive={sha_short}  per-axis normalized  hot=high leverage",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, label="normalized |gradient|")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────── #
# Plot 3 — cumulative gradient-by-rank                                          #
# ──────────────────────────────────────────────────────────────────────────── #


def plot_cumulative_by_rank(
    aggregate_gradient: np.ndarray,
    anchor: dict,
    output_path: Path,
) -> None:
    """Cumulative gradient sensitivity by rank (Pareto leverage curve).

    Shows what fraction of total sensitivity is captured by the top-K
    bytes. Steep early curve = highly concentrated (few bytes dominate);
    flat curve = broadly distributed. Reads off the canonical Pareto
    leverage operating points (top-10%, top-1%, top-0.1%).
    """
    plt = _ensure_matplotlib()
    n_bytes = aggregate_gradient.shape[0]
    abs_grad = np.abs(aggregate_gradient)
    watermark = _watermark_for_anchor(anchor)
    sha_short = _short_sha(anchor.get("archive_sha256"))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for axis_idx, (ax, label) in enumerate(zip(axes, AXIS_LABELS)):
        per_byte_axis = abs_grad[:, axis_idx]
        sorted_desc = np.sort(per_byte_axis)[::-1]
        cumulative = np.cumsum(sorted_desc)
        total = cumulative[-1] if cumulative[-1] > 0 else 1.0
        cumulative_norm = cumulative / total
        ranks = np.arange(1, n_bytes + 1) / n_bytes
        ax.plot(
            ranks,
            cumulative_norm,
            color=("tab:blue", "tab:orange", "tab:green")[axis_idx],
            linewidth=1.5,
        )
        ax.set_xlabel("rank fraction (cumulative)")
        ax.set_ylabel(f"cumulative |g_{label}| fraction")
        ax.set_title(f"{label} cumulative leverage curve")
        ax.grid(True, alpha=0.3)
        # Annotate top-1% and top-10% leverage
        top_1pct_rank = int(n_bytes * 0.01)
        top_10pct_rank = int(n_bytes * 0.10)
        if top_1pct_rank > 0:
            ax.axvline(0.01, color="red", linestyle=":", alpha=0.5)
            ax.text(
                0.01,
                cumulative_norm[top_1pct_rank - 1],
                f" top-1%: {cumulative_norm[top_1pct_rank - 1]:.0%}",
                fontsize=8,
                verticalalignment="bottom",
            )
        if top_10pct_rank > 0:
            ax.axvline(0.10, color="orange", linestyle=":", alpha=0.5)
            ax.text(
                0.10,
                cumulative_norm[top_10pct_rank - 1],
                f" top-10%: {cumulative_norm[top_10pct_rank - 1]:.0%}",
                fontsize=8,
                verticalalignment="bottom",
            )
    fig.suptitle(
        f"Cumulative gradient by rank {watermark}  archive={sha_short}",
        fontsize=11,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────── #
# Plot 4 — cross-substrate gradient correlation                                 #
# ──────────────────────────────────────────────────────────────────────────── #


def plot_cross_substrate_correlation(
    substrate_anchors: list[tuple[str, np.ndarray, dict]],
    output_path: Path,
) -> None:
    """Cross-substrate gradient correlation matrix (when ≥2 anchors).

    Reveals whether different substrates respond similarly to byte mutations:
    high cross-correlation = substrates share underlying sensitivity
    structure; low correlation = substrates have distinct leverage profiles
    (Wyner-Ziv complementary candidates).

    Each cell C[i,j] = cosine similarity between aggregate gradient L1
    magnitude vectors of substrate_i and substrate_j. Off-diagonal ranges
    [-1, 1]; positive = similar leverage profile; negative = opposite.
    """
    plt = _ensure_matplotlib()
    n = len(substrate_anchors)
    if n < 2:
        # Single-substrate degenerate: emit a placeholder with explanation
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5,
            0.5,
            f"Cross-substrate correlation requires ≥2 anchors;\n"
            f"only {n} anchor(s) provided.\n"
            f"Re-run with multiple --archive-sha flags.",
            ha="center",
            va="center",
            fontsize=10,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120)
        plt.close(fig)
        return

    # Build correlation matrix
    profiles = []
    labels = []
    for label, aggregate, anchor in substrate_anchors:
        per_byte_l1 = np.abs(aggregate).sum(axis=1)  # (N_bytes,)
        norm = float(np.linalg.norm(per_byte_l1))
        if norm > 0:
            profiles.append(per_byte_l1 / norm)
        else:
            profiles.append(per_byte_l1)
        labels.append(f"{label}\n{_short_sha(anchor.get('archive_sha256'))}")

    # Align lengths (truncate to min) — different substrates may have
    # different N_bytes
    min_len = min(p.size for p in profiles)
    aligned = np.stack([p[:min_len] for p in profiles])
    correlation = aligned @ aligned.T  # (N_substrates, N_substrates) cosine

    fig, ax = plt.subplots(figsize=(max(6, n * 0.8), max(5, n * 0.8)))
    im = ax.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(n):
        for j in range(n):
            value = correlation[i, j]
            ax.text(
                j,
                i,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color=("white" if abs(value) > 0.5 else "black"),
            )
    ax.set_title(
        f"Cross-substrate gradient correlation matrix ({n} substrates)\n"
        f"cosine similarity on aligned aggregate |g| L1 profiles",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, label="cosine similarity")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────── #
# Plot 5 — Wyner-Ziv layer-aware gradient flow                                  #
# ──────────────────────────────────────────────────────────────────────────── #


def plot_wyner_ziv_flow(
    aggregate_gradient: np.ndarray,
    anchor: dict,
    output_path: Path,
) -> None:
    """Wyner-Ziv layer-aware gradient flow per substrate section.

    Reads the substrate's section manifest from the anchor (when present in
    ``anchor['archive_layout']['sections']``) and emits a stacked bar chart
    of per-section gradient contribution per axis. The Wyner-Ziv canonical
    classification (PAIR_INVARIANT vs PAIR_SPECIFIC vs PAIR_NEUTRAL per
    consumer 1) overlays the section breakdown so the operator sees where
    each Wyner-Ziv class concentrates spatially in the archive.

    When the anchor does NOT carry a section manifest, the plot degrades
    gracefully to a single-section bar showing total per-axis gradient.
    """
    plt = _ensure_matplotlib()
    watermark = _watermark_for_anchor(anchor)
    sha_short = _short_sha(anchor.get("archive_sha256"))

    sections = []
    archive_layout = anchor.get("archive_layout")
    if isinstance(archive_layout, dict):
        section_entries = archive_layout.get("sections")
        if isinstance(section_entries, list):
            for entry in section_entries:
                if not isinstance(entry, dict):
                    continue
                name = entry.get("name") or entry.get("section_name") or "section"
                start = entry.get("offset") or entry.get("start") or 0
                length = entry.get("length") or entry.get("size") or 0
                try:
                    start = int(start)
                    length = int(length)
                except (TypeError, ValueError):
                    continue
                if length > 0:
                    sections.append((str(name)[:30], start, length))

    abs_grad = np.abs(aggregate_gradient)  # (N_bytes, 3)
    if not sections:
        # Degenerate: single-section fallback
        sections = [("(no section manifest)", 0, abs_grad.shape[0])]

    section_names = [s[0] for s in sections]
    per_section_seg = []
    per_section_pose = []
    per_section_rate = []
    for _, start, length in sections:
        end = min(start + length, abs_grad.shape[0])
        if end <= start:
            per_section_seg.append(0.0)
            per_section_pose.append(0.0)
            per_section_rate.append(0.0)
            continue
        slab = abs_grad[start:end]
        per_section_seg.append(float(slab[:, 0].sum()))
        per_section_pose.append(float(slab[:, 1].sum()))
        per_section_rate.append(float(slab[:, 2].sum()))

    n_sections = len(section_names)
    xs = np.arange(n_sections)

    fig, ax = plt.subplots(figsize=(max(8, n_sections * 0.5), 5))
    width = 0.6
    seg_arr = np.asarray(per_section_seg)
    pose_arr = np.asarray(per_section_pose)
    rate_arr = np.asarray(per_section_rate)
    # Normalize to fraction-of-total per section for a stacked bar
    totals = seg_arr + pose_arr + rate_arr
    safe_totals = np.where(totals > 0, totals, 1.0)
    seg_frac = seg_arr / safe_totals
    pose_frac = pose_arr / safe_totals
    rate_frac = rate_arr / safe_totals

    ax.bar(xs, seg_frac, width, label="seg", color="tab:blue")
    ax.bar(xs, pose_frac, width, bottom=seg_frac, label="pose", color="tab:orange")
    ax.bar(
        xs,
        rate_frac,
        width,
        bottom=seg_frac + pose_frac,
        label="rate",
        color="tab:green",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(section_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("gradient fraction per section")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        f"Wyner-Ziv per-section gradient flow {watermark}\n"
        f"archive={sha_short}  n_sections={n_sections}  "
        f"stacked seg+pose+rate fraction per section",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────── #
# Main CLI                                                                      #
# ──────────────────────────────────────────────────────────────────────────── #


def _resolve_anchor_for_plot(
    archive_sha: str,
    *,
    require_per_pair: bool,
) -> tuple[np.ndarray, dict]:
    """Resolve aggregate or per-pair gradient + anchor for one archive sha."""
    if require_per_pair:
        try:
            return load_per_pair_gradient_from_anchor(archive_sha256=archive_sha)
        except FileNotFoundError as exc:
            raise SystemExit(
                f"per-pair anchor required but missing for archive {archive_sha[:16]}; "
                f"run `tools/extract_master_gradient.py --preserve-per-pair` first ({exc})"
            )
    try:
        return load_aggregate_gradient_from_anchor(archive_sha256=archive_sha)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"aggregate anchor required but missing for archive {archive_sha[:16]} ({exc})"
        )


def _emit_plot(
    plot_name: str,
    archive_shas: list[str],
    output: Path,
) -> None:
    """Dispatch the plot to the correct emitter; resolve required anchors."""
    if plot_name == "per_pair_distribution":
        sha = archive_shas[0]
        per_pair, anchor = _resolve_anchor_for_plot(sha, require_per_pair=True)
        plot_per_pair_distribution(per_pair, anchor, output)
    elif plot_name == "per_byte_heatmap":
        sha = archive_shas[0]
        aggregate, anchor = _resolve_anchor_for_plot(sha, require_per_pair=False)
        plot_per_byte_heatmap(aggregate, anchor, output)
    elif plot_name == "cumulative_by_rank":
        sha = archive_shas[0]
        aggregate, anchor = _resolve_anchor_for_plot(sha, require_per_pair=False)
        plot_cumulative_by_rank(aggregate, anchor, output)
    elif plot_name == "cross_substrate_correlation":
        substrate_anchors = []
        for sha in archive_shas:
            aggregate, anchor = _resolve_anchor_for_plot(sha, require_per_pair=False)
            label = anchor.get("substrate_id") or anchor.get("substrate") or sha[:8]
            substrate_anchors.append((str(label), aggregate, anchor))
        plot_cross_substrate_correlation(substrate_anchors, output)
    elif plot_name == "wyner_ziv_flow":
        sha = archive_shas[0]
        aggregate, anchor = _resolve_anchor_for_plot(sha, require_per_pair=False)
        plot_wyner_ziv_flow(aggregate, anchor, output)
    else:
        raise SystemExit(f"unknown plot name: {plot_name!r}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Master-gradient xray visualization tool — 5 canonical plot "
            "types. Per CLAUDE.md no /tmp paths; pass an explicit --output."
        ),
    )
    parser.add_argument(
        "--archive-sha",
        action="append",
        required=False,
        help=(
            "Archive sha256 to plot. May be repeated for "
            "cross_substrate_correlation. For single-archive plots only "
            "the first --archive-sha is used."
        ),
    )
    parser.add_argument(
        "--plot",
        choices=CANONICAL_PLOTS,
        default="all",
        help="Plot type to emit (default: all 5 plots).",
    )
    parser.add_argument(
        "--output",
        required=False,
        help=(
            "Output path (.png for single plot, directory for --plot all). "
            "Caller MUST pass an explicit non-/tmp path per CLAUDE.md."
        ),
    )
    parser.add_argument(
        "--list-plots",
        action="store_true",
        help="List canonical plot names and exit.",
    )
    args = parser.parse_args(argv)

    if args.list_plots:
        print(
            json.dumps(
                {
                    "canonical_plots": list(CANONICAL_PLOTS),
                    "description": "master-gradient xray 5 canonical plot types",
                },
                indent=2,
            )
        )
        return 0

    if not args.output:
        raise SystemExit("--output is required when not using --list-plots")
    if not args.archive_sha:
        raise SystemExit(
            "--archive-sha is required (specify at least one for "
            "single-substrate plots; multiple for cross_substrate_correlation)"
        )
    output_path = Path(args.output).resolve()
    if "/tmp/" in str(output_path) or str(output_path).startswith("/tmp"):
        raise SystemExit(
            "/tmp output paths are FORBIDDEN per CLAUDE.md "
            '"Forbidden /tmp paths in any persisted artifact"; '
            "pass an explicit non-/tmp --output (e.g. reports/master_gradient_xray/<archive>/)"
        )

    if args.plot == "all":
        if not output_path.is_dir() and output_path.suffix == "":
            output_path.mkdir(parents=True, exist_ok=True)
        elif not output_path.is_dir():
            raise SystemExit(
                f"--plot all requires --output to be a directory; got file {output_path}"
            )
        for plot_name in CANONICAL_PLOTS:
            if plot_name == "all":
                continue
            target = output_path / f"{plot_name}.png"
            try:
                _emit_plot(plot_name, args.archive_sha, target)
                print(f"  wrote {target}")
            except SystemExit as exc:
                # Per-plot failures don't abort the batch; report and continue
                print(
                    f"  SKIP {plot_name}: {exc}",
                    file=sys.stderr,
                )
    else:
        if output_path.is_dir():
            output_path = output_path / f"{args.plot}.png"
        _emit_plot(args.plot, args.archive_sha, output_path)
        print(f"wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
