# SPDX-License-Identifier: MIT
"""Master-gradient xray visualization tool.

Per Cable D D4 (task #797) batched 2026-05-19, lane
``lane_cable_d_master_gradient_extension_batch_20260519``; extended
2026-05-19 by ``lane_master_gradient_xray_viz_tool_20260519`` with sister
JSON sidecars + canonical Catalog #323 Provenance + ``--output-dir`` index
HTML mode + ``--anchor-jsonl`` / ``--mps-drift-json`` CLI flags + sample
empirical run. Sister of the canonical producer
``tools/extract_master_gradient.py`` + canonical consumer catalog
``tac.master_gradient_consumers``.

5 canonical plot types per the master-gradient + xray follow-up research
(task #874):

  1. per-pair gradient distribution histogram (per-pair |g| L1 distribution
     across all pairs)
  2. per-byte sensitivity heatmap (N_bytes × 3 axes; one row per top-K byte)
  3. cumulative gradient-by-rank curve (rank vs cumulative sensitivity)
  4. cross-substrate gradient correlation matrix (when multiple anchors)
  5. Wyner-Ziv layer-aware gradient flow (per-substrate-section gradient
     breakdown if substrate parser declares sections)

Per Catalog #305 observability non-negotiable + Catalog #323 canonical
Provenance: every emitted plot carries a sister ``<plot_id>.json`` carrying
the underlying summary statistics + a canonical ``provenance`` sub-object
built via ``tac.provenance.builders.build_provenance_for_predicted`` so
downstream consumers can audit the visualization-vs-measurement boundary.
Per Catalog #287/#323 every emitted statistic is tagged ``[predicted]``
evidence-grade; the visualization is a derived view, NOT a primary
measurement.

``--output-dir`` mode additionally emits an operator-friendly ``index.html``
landing page linking every plot + sister JSON + cross-references.

Per CLAUDE.md "no /tmp paths in persisted artifacts" + Catalog #208: the
caller MUST pass an explicit non-/tmp ``--output`` (or ``--output-dir``).
Per "MPS auth eval is NOISE" + Catalog #192: anchors with non-authoritative
grades render with explicit watermark.

Per CLAUDE.md "Apples-to-apples evidence discipline": every figure carries
an ``[axis-tag]`` watermark in the title matching the master-gradient
anchor's ``measurement_axis`` field.

Usage::

    # All 5 plots into a directory + index.html + sister JSON sidecars:
    .venv/bin/python tools/master_gradient_xray.py \\
        --archive-sha 6bae0201abcd... \\
        --output-dir reports/master_gradient_xray/<archive>/

    # Single plot to a single file:
    .venv/bin/python tools/master_gradient_xray.py \\
        --plot per_pair_distribution --archive-sha 6bae0201abcd... \\
        --output reports/x/per_pair.png

    # Cross-reference MPS-vs-CUDA drift on top of gradient sensitivity:
    .venv/bin/python tools/master_gradient_xray.py \\
        --archive-sha 6bae0201abcd... \\
        --mps-drift-json .omx/state/mps_drift_granular_20260519T122700Z.json \\
        --output-dir reports/master_gradient_xray/<archive>/

Plot types accepted by ``--plot``:
  - ``per_pair_distribution`` (plot 1)
  - ``per_byte_heatmap`` (plot 2)
  - ``cumulative_by_rank`` (plot 3)
  - ``cross_substrate_correlation`` (plot 4 — requires ``--archive-sha`` repeated)
  - ``wyner_ziv_flow`` (plot 5)
  - ``drift_vs_sensitivity_scatter`` (plot 6 — requires ``--mps-drift-json``)
  - ``all`` (emit canonical 5 into ``--output-dir`` directory; plot 6 fires
    when ``--mps-drift-json`` is also provided)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
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
from tac.provenance.builders import (  # noqa: E402
    build_provenance_for_predicted,
)


CANONICAL_PLOTS = (
    "per_pair_distribution",
    "per_byte_heatmap",
    "cumulative_by_rank",
    "cross_substrate_correlation",
    "wyner_ziv_flow",
    "drift_vs_sensitivity_scatter",
    "all",
)

# Canonical 5-plot taxonomy emitted by `--output-dir`/`--plot all` when no
# `--mps-drift-json` is provided. drift_vs_sensitivity_scatter is the
# optional 6th plot that only fires when the cross-reference JSON is
# available.
CANONICAL_OUTPUT_DIR_PLOTS = (
    "per_pair_distribution",
    "per_byte_heatmap",
    "cumulative_by_rank",
    "cross_substrate_correlation",
    "wyner_ziv_flow",
)

AXIS_LABELS = ("seg", "pose", "rate")

# Sister JSON schema for plot summary stats. Bumping invalidates downstream
# observability consumers (autopilot dashboard, audit tools, etc.).
PLOT_SIDECAR_SCHEMA_VERSION = "master_gradient_xray_plot_sidecar_v1_20260519"
INDEX_HTML_SCHEMA_VERSION = "master_gradient_xray_index_v1_20260519"


def _utc_now_iso() -> str:
    """Canonical UTC timestamp for sidecar provenance + index HTML."""
    return datetime.now(UTC).isoformat()


def _sha256_anchor_fingerprint(anchor: dict) -> str:
    """Canonical input fingerprint for Provenance.source_sha256 of derived plots."""
    canonical = json.dumps(
        {
            "archive_sha256": anchor.get("archive_sha256"),
            "gradient_array_path": anchor.get("gradient_array_path"),
            "measurement_axis": anchor.get("measurement_axis"),
            "measurement_hardware": anchor.get("measurement_hardware"),
            "measurement_utc": anchor.get("measurement_utc"),
            "gradient_tensor_kind": anchor.get("gradient_tensor_kind"),
            "n_bytes": anchor.get("n_bytes"),
        },
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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
# Plot 6 — Drift vs sensitivity scatter (cross-reference with MPS drift JSON)   #
# ──────────────────────────────────────────────────────────────────────────── #


def _load_mps_drift_json(path: Path) -> dict:
    """Strict-load the MPS drift granular JSON (sister of Catalog #138)."""
    if not path.is_file():
        raise SystemExit(
            f"--mps-drift-json {path} does not exist; "
            "run `tools/analyze_mps_drift_granular.py` first"
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(
            f"--mps-drift-json {path} failed to parse: {exc}"
        )
    if not isinstance(data, dict):
        raise SystemExit(
            f"--mps-drift-json {path} top-level must be dict; got {type(data).__name__}"
        )
    return data


def plot_drift_vs_sensitivity_scatter(
    per_pair_gradient: np.ndarray | None,
    aggregate_gradient: np.ndarray,
    anchor: dict,
    mps_drift_data: dict,
    output_path: Path,
) -> None:
    """Per-pair drift (MPS-vs-CUDA) vs sensitivity (master-gradient |g_p|) scatter.

    Per `feedback_mps_phase_b_options_b_plus_c_completion_landed_20260519.md`
    + the slot 9 mathematical formalization: the distribution shape IS the
    answer to "is MPS drift in score-relevant subspace or nullspace?".

    Quadrant analysis:
      - top-right (high drift + high sensitivity) = HIGHEST EV for
        engineering correction
      - top-left  (low  drift + high sensitivity) = MPS-VIABLE confirmed
      - bottom-right (high drift + low  sensitivity) = locally-free zone
      - bottom-left  (low  drift + low  sensitivity) = neutral

    Falls back to a per-frame drift overlay when per-pair gradient is not
    available (most aggregate-only anchors). Each point is then one pair-side
    (frame) instead of one pair.
    """
    plt = _ensure_matplotlib()
    watermark = _watermark_for_anchor(anchor)
    sha_short = _short_sha(anchor.get("archive_sha256"))

    per_pair_drift = mps_drift_data.get("per_pair") or []
    per_frame_drift = mps_drift_data.get("per_frame") or []

    # Prefer per-pair drift + per-pair gradient, fall back to per-frame drift
    if per_pair_gradient is not None and per_pair_drift:
        per_pair_l1 = np.abs(per_pair_gradient).sum(axis=(0, 2))  # (n_pairs,)
        drift_values = []
        sensitivity_values = []
        for row in per_pair_drift:
            if not isinstance(row, dict):
                continue
            pair_idx = row.get("pair_index")
            agg = row.get("aggregate")
            if (
                isinstance(pair_idx, int)
                and isinstance(agg, (int, float))
                and 0 <= pair_idx < per_pair_l1.shape[0]
            ):
                drift_values.append(float(agg))
                sensitivity_values.append(float(per_pair_l1[pair_idx]))
        x_label = "per-pair MPS-vs-CUDA aggregate drift"
        point_label = "pair"
    elif per_frame_drift:
        # Aggregate gradient case: use per-byte L1 magnitude broadcast over frames
        per_byte_l1_total = float(np.abs(aggregate_gradient).sum())
        drift_values = []
        sensitivity_values = []
        for row in per_frame_drift:
            if not isinstance(row, dict):
                continue
            agg = row.get("aggregate")
            if isinstance(agg, (int, float)):
                drift_values.append(float(agg))
                # Synthetic per-frame sensitivity proxy: aggregate gradient total
                # (this is a placeholder; per-pair gradient gives true signal)
                sensitivity_values.append(per_byte_l1_total)
        x_label = "per-frame MPS-vs-CUDA aggregate drift"
        point_label = "frame"
    else:
        drift_values = []
        sensitivity_values = []
        x_label = "drift (no MPS data)"
        point_label = "pair"

    fig, ax = plt.subplots(figsize=(9, 6))
    if not drift_values or not sensitivity_values:
        ax.text(
            0.5,
            0.5,
            "drift_vs_sensitivity_scatter requires:\n"
            "  - per-pair drift in mps_drift_json (mps_drift_data['per_pair'])\n"
            "  - OR per-frame drift in mps_drift_data['per_frame']\n"
            "Neither was found.",
            ha="center",
            va="center",
            fontsize=10,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        drift_arr = np.asarray(drift_values, dtype=np.float64)
        sens_arr = np.asarray(sensitivity_values, dtype=np.float64)
        ax.scatter(drift_arr, sens_arr, alpha=0.6, s=24, color="tab:purple")
        ax.set_xlabel(x_label)
        ax.set_ylabel("per-pair gradient sensitivity ‖g_p‖ (L1)")
        ax.grid(True, alpha=0.3)
        # Quadrant lines at medians
        x_med = float(np.median(drift_arr))
        y_med = float(np.median(sens_arr))
        ax.axvline(x_med, color="gray", linestyle="--", alpha=0.5)
        ax.axhline(y_med, color="gray", linestyle="--", alpha=0.5)
        # Linear-fit overlay + R^2
        if drift_arr.size >= 3 and float(np.std(drift_arr)) > 0:
            try:
                slope, intercept = np.polyfit(drift_arr, sens_arr, 1)
                fit_y = slope * drift_arr + intercept
                ss_res = float(np.sum((sens_arr - fit_y) ** 2))
                ss_tot = float(np.sum((sens_arr - float(np.mean(sens_arr))) ** 2))
                r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
                xs_line = np.linspace(drift_arr.min(), drift_arr.max(), 20)
                ax.plot(
                    xs_line,
                    slope * xs_line + intercept,
                    color="tab:red",
                    linewidth=1.5,
                    alpha=0.8,
                    label=f"linear fit  R²={r_squared:.3f}",
                )
                ax.legend(loc="upper right", fontsize=9)
            except (np.linalg.LinAlgError, ValueError):
                pass
        ax.set_title(
            f"Drift vs sensitivity scatter {watermark}\n"
            f"archive={sha_short}  n={len(drift_arr)} {point_label}s  "
            f"quadrant axes at median",
            fontsize=10,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


# ──────────────────────────────────────────────────────────────────────────── #
# Sister JSON sidecar emission — per Catalog #305 observability +              #
# Catalog #323 canonical Provenance                                            #
# ──────────────────────────────────────────────────────────────────────────── #


def _summary_stats_for_aggregate(aggregate_gradient: np.ndarray) -> dict[str, object]:
    """Summary statistics for the aggregate (N_bytes, 3) gradient tensor."""
    abs_grad = np.abs(aggregate_gradient)
    per_byte_l1 = abs_grad.sum(axis=1)
    sorted_desc = np.sort(per_byte_l1)[::-1]
    n_bytes = int(aggregate_gradient.shape[0])
    cumulative = np.cumsum(sorted_desc)
    total = float(cumulative[-1]) if cumulative.size and cumulative[-1] > 0 else 0.0
    top_1pct_rank = max(1, int(n_bytes * 0.01))
    top_10pct_rank = max(1, int(n_bytes * 0.10))
    return {
        "n_bytes": n_bytes,
        "n_axes": int(aggregate_gradient.shape[1]),
        "per_axis_l1_total": [float(abs_grad[:, i].sum()) for i in range(aggregate_gradient.shape[1])],
        "per_axis_max": [float(abs_grad[:, i].max()) for i in range(aggregate_gradient.shape[1])],
        "per_byte_l1_max": float(per_byte_l1.max()) if per_byte_l1.size else 0.0,
        "per_byte_l1_median": float(np.median(per_byte_l1)) if per_byte_l1.size else 0.0,
        "per_byte_l1_total": total,
        "top_1pct_leverage_fraction": (
            float(cumulative[top_1pct_rank - 1]) / total if total > 0 else 0.0
        ),
        "top_10pct_leverage_fraction": (
            float(cumulative[top_10pct_rank - 1]) / total if total > 0 else 0.0
        ),
        "top_10_byte_indices": [int(i) for i in np.argsort(-per_byte_l1)[:10]],
    }


def _summary_stats_for_per_pair(per_pair_gradient: np.ndarray) -> dict[str, object]:
    """Summary statistics for the per-pair (N_bytes, N_pairs, 3) gradient tensor."""
    abs_grad = np.abs(per_pair_gradient)
    per_pair_l1 = abs_grad.sum(axis=(0, 2))  # (n_pairs,)
    per_pair_per_axis = abs_grad.sum(axis=0)  # (n_pairs, 3)
    return {
        "n_bytes": int(per_pair_gradient.shape[0]),
        "n_pairs": int(per_pair_gradient.shape[1]),
        "n_axes": int(per_pair_gradient.shape[2]),
        "per_pair_l1_mean": float(per_pair_l1.mean()) if per_pair_l1.size else 0.0,
        "per_pair_l1_median": float(np.median(per_pair_l1)) if per_pair_l1.size else 0.0,
        "per_pair_l1_max": float(per_pair_l1.max()) if per_pair_l1.size else 0.0,
        "per_pair_per_axis_max": [
            float(per_pair_per_axis[:, i].max()) for i in range(per_pair_gradient.shape[2])
        ],
        "top_10_pair_indices_by_l1": [int(i) for i in np.argsort(-per_pair_l1)[:10]],
    }


def _emit_sidecar_json(
    sidecar_path: Path,
    plot_id: str,
    anchor: dict,
    summary: dict[str, object],
    extra: dict[str, object] | None = None,
) -> None:
    """Emit canonical sister JSON for one plot per Catalog #305 + #323."""
    fingerprint = _sha256_anchor_fingerprint(anchor)
    provenance = build_provenance_for_predicted(
        model_id=f"master_gradient_xray.{plot_id}.v1",
        inputs_sha256=fingerprint,
        measurement_axis="[predicted]",
        hardware_substrate="visualization_derived_view",
        captured_at_utc=_utc_now_iso(),
    )
    payload = {
        "schema_version": PLOT_SIDECAR_SCHEMA_VERSION,
        "plot_id": plot_id,
        "anchor": {
            "archive_sha256": anchor.get("archive_sha256"),
            "measurement_axis": anchor.get("measurement_axis"),
            "measurement_hardware": anchor.get("measurement_hardware"),
            "measurement_utc": anchor.get("measurement_utc"),
            "gradient_tensor_kind": anchor.get("gradient_tensor_kind"),
            "evidence_grade": anchor.get("evidence_grade"),
            "n_bytes": anchor.get("n_bytes"),
            "n_pairs_used": anchor.get("n_pairs_used"),
            "operating_point": anchor.get("operating_point"),
        },
        "summary_statistics": summary,
        "extra": extra or {},
        "provenance": asdict(provenance),
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "notes": (
            "Derived visualization sidecar per Catalog #305 observability + "
            "Catalog #323 canonical Provenance. score_claim=False, "
            "promotion_eligible=False, evidence_grade=[predicted] by "
            "construction; visualization is a derived view, NOT a primary "
            "measurement. Use the underlying anchor's measurement_axis for "
            "any score-axis claim. Per CLAUDE.md 'Apples-to-apples evidence "
            "discipline' the plot is observability infrastructure only."
        ),
    }
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    sidecar_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _emit_index_html(
    output_dir: Path,
    archive_shas: list[str],
    anchors: list[dict],
    emitted_plots: list[dict[str, str]],
    mps_drift_json_path: Path | None,
) -> None:
    """Emit operator-friendly index.html landing page linking every plot.

    Per CLAUDE.md "Max observability — non-negotiable" + Catalog #305
    facet 4 (queryable post-hoc): the index page embeds plots + links to
    sister JSON + cross-references the underlying anchors.
    """
    def _h(s: object) -> str:
        return (
            str(s)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
        )

    archive_rows = []
    for sha, anchor in zip(archive_shas, anchors):
        archive_rows.append(
            f"<tr><td><code>{_h(_short_sha(sha))}</code></td>"
            f"<td>{_h(anchor.get('measurement_axis'))}</td>"
            f"<td>{_h(anchor.get('measurement_hardware'))}</td>"
            f"<td>{_h(anchor.get('gradient_tensor_kind'))}</td>"
            f"<td>{_h(anchor.get('n_bytes'))}</td>"
            f"<td>{_h(anchor.get('evidence_grade') or 'unset')}</td></tr>"
        )

    plot_cards = []
    for entry in emitted_plots:
        plot_id = entry["plot_id"]
        png_rel = entry["png_relative"]
        sidecar_rel = entry["sidecar_relative"]
        plot_cards.append(
            f"<div class=\"plot-card\">"
            f"<h3>{_h(plot_id)}</h3>"
            f"<a href=\"{_h(png_rel)}\"><img src=\"{_h(png_rel)}\" alt=\"{_h(plot_id)}\" /></a>"
            f"<p>Sister JSON: <a href=\"{_h(sidecar_rel)}\"><code>{_h(sidecar_rel)}</code></a></p>"
            f"</div>"
        )

    cross_ref_lines = []
    if mps_drift_json_path is not None:
        try:
            mps_rel = Path(mps_drift_json_path).resolve().relative_to(output_dir.resolve())
        except ValueError:
            mps_rel = Path(mps_drift_json_path).resolve()
        cross_ref_lines.append(
            f"<li>MPS drift cross-reference: <code>{_h(mps_rel)}</code></li>"
        )
    cross_ref_lines.extend(
        [
            "<li>Canonical producer: <code>tools/extract_master_gradient.py</code></li>",
            "<li>Canonical consumer catalog: <code>tac.master_gradient_consumers</code></li>",
            "<li>Catalog #305 observability surface (this tool IS the operator-facing visualization surface)</li>",
            "<li>Catalog #323 canonical Provenance (sister JSON sidecars carry the contract)</li>",
            "<li>Catalog #318 master-gradient raw-byte-authority self-protection</li>",
            "<li>Catalog #327 master-gradient contest-axis authority custody</li>",
        ]
    )

    html = f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
<meta charset=\"utf-8\">
<title>master-gradient xray — {_h(', '.join(_short_sha(s) for s in archive_shas))}</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1200px; margin: 1em auto; padding: 0 1em; color: #1f2937; }}
h1 {{ border-bottom: 2px solid #4b5563; padding-bottom: 0.25em; }}
h2 {{ border-bottom: 1px solid #d1d5db; padding-bottom: 0.15em; margin-top: 1.5em; }}
table {{ border-collapse: collapse; margin: 0.5em 0; }}
td, th {{ border: 1px solid #d1d5db; padding: 4px 8px; font-size: 0.9em; }}
th {{ background: #f3f4f6; text-align: left; }}
.plot-card {{ border: 1px solid #e5e7eb; padding: 12px; margin: 12px 0; border-radius: 6px; background: #fafafa; }}
.plot-card img {{ max-width: 100%; height: auto; border: 1px solid #d1d5db; }}
.plot-card h3 {{ margin-top: 0; color: #1d4ed8; }}
.banner {{ background: #fef3c7; border-left: 4px solid #d97706; padding: 8px 12px; margin: 12px 0; font-size: 0.9em; }}
code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 3px; font-size: 0.85em; }}
ul {{ font-size: 0.9em; }}
</style>
</head>
<body>
<h1>master-gradient xray visualization</h1>
<p>Generated: <code>{_h(_utc_now_iso())}</code> UTC</p>
<p>Schema: <code>{_h(INDEX_HTML_SCHEMA_VERSION)}</code></p>

<div class=\"banner\">
<b>Observability surface, NOT score authority.</b> Every sidecar JSON carries
<code>score_claim=false</code> + <code>promotion_eligible=false</code> +
<code>evidence_grade=[predicted]</code> per Catalog #305 / #323 / #287. The
underlying anchor's <code>measurement_axis</code> determines any score-axis
claim — see the table below.
</div>

<h2>Operator question driving the analysis</h2>
<p>Where is the score-relevant byte leverage concentrated in this archive,
and how does it intersect with MPS-vs-CUDA drift? Inputs: the master-gradient
anchor (per-archive sensitivity tensor) and (optionally) the granular MPS
drift JSON. Outputs: 5 canonical plots that decompose leverage by byte, by
pair, and (when drift is available) by quadrant.</p>

<h2>Archives processed ({_h(len(archive_shas))})</h2>
<table>
<tr><th>sha[:12]</th><th>axis</th><th>hardware</th><th>tensor_kind</th><th>n_bytes</th><th>evidence_grade</th></tr>
{''.join(archive_rows)}
</table>

<h2>Plots ({_h(len(emitted_plots))})</h2>
{''.join(plot_cards) if plot_cards else '<p><em>(no plots emitted)</em></p>'}

<h2>Cross-references</h2>
<ul>
{''.join(cross_ref_lines)}
</ul>

<hr>
<p style=\"color: #6b7280; font-size: 0.85em;\">
Generated by <code>tools/master_gradient_xray.py</code> on lane
<code>lane_master_gradient_xray_viz_tool_20260519</code>. Per the operator
standing directive: this tool IS the observability surface — operator routing
decisions consume the plots but the score authority remains with the
underlying contest-axis anchor.
</p>
</body>
</html>
"""
    index_path = output_dir / "index.html"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(html, encoding="utf-8")


# ──────────────────────────────────────────────────────────────────────────── #
# Main CLI                                                                      #
# ──────────────────────────────────────────────────────────────────────────── #


def _resolve_anchor_for_plot(
    archive_sha: str,
    *,
    require_per_pair: bool,
    soft: bool = False,
) -> tuple[np.ndarray, dict] | None:
    """Resolve aggregate or per-pair gradient + anchor for one archive sha.

    When ``soft=True``, returns None on FileNotFoundError instead of raising.
    Used by ``--output-dir`` mode to keep emitting other plots when one
    underlying anchor type is unavailable.
    """
    if require_per_pair:
        try:
            return load_per_pair_gradient_from_anchor(archive_sha256=archive_sha)
        except FileNotFoundError as exc:
            if soft:
                return None
            raise SystemExit(
                f"per-pair anchor required but missing for archive {archive_sha[:16]}; "
                f"run `tools/extract_master_gradient.py --preserve-per-pair` first ({exc})"
            )
    try:
        return load_aggregate_gradient_from_anchor(archive_sha256=archive_sha)
    except FileNotFoundError as exc:
        if soft:
            return None
        raise SystemExit(
            f"aggregate anchor required but missing for archive {archive_sha[:16]} ({exc})"
        )


def _emit_plot(
    plot_name: str,
    archive_shas: list[str],
    output: Path,
    *,
    mps_drift_data: dict | None = None,
) -> dict[str, object] | None:
    """Dispatch the plot to the correct emitter; resolve required anchors.

    Returns a per-plot summary dict suitable for sister JSON emission, or
    ``None`` when no sidecar is appropriate (e.g. cross-substrate placeholder).
    """
    if plot_name == "per_pair_distribution":
        sha = archive_shas[0]
        per_pair, anchor = _resolve_anchor_for_plot(sha, require_per_pair=True)
        plot_per_pair_distribution(per_pair, anchor, output)
        return {
            "anchor": anchor,
            "summary": _summary_stats_for_per_pair(per_pair),
            "extra": {},
        }
    elif plot_name == "per_byte_heatmap":
        sha = archive_shas[0]
        aggregate, anchor = _resolve_anchor_for_plot(sha, require_per_pair=False)
        plot_per_byte_heatmap(aggregate, anchor, output)
        return {
            "anchor": anchor,
            "summary": _summary_stats_for_aggregate(aggregate),
            "extra": {"plot_top_k": 128},
        }
    elif plot_name == "cumulative_by_rank":
        sha = archive_shas[0]
        aggregate, anchor = _resolve_anchor_for_plot(sha, require_per_pair=False)
        plot_cumulative_by_rank(aggregate, anchor, output)
        return {
            "anchor": anchor,
            "summary": _summary_stats_for_aggregate(aggregate),
            "extra": {},
        }
    elif plot_name == "cross_substrate_correlation":
        substrate_anchors = []
        for sha in archive_shas:
            aggregate, anchor = _resolve_anchor_for_plot(sha, require_per_pair=False)
            label = anchor.get("substrate_id") or anchor.get("substrate") or sha[:8]
            substrate_anchors.append((str(label), aggregate, anchor))
        plot_cross_substrate_correlation(substrate_anchors, output)
        # Cross-substrate sidecar uses first anchor + aggregate stats across all
        first_anchor = substrate_anchors[0][2] if substrate_anchors else {}
        return {
            "anchor": first_anchor,
            "summary": {
                "n_substrates": len(substrate_anchors),
                "archive_shas": [s for s, _, _ in [(a[2].get("archive_sha256"), 0, 0) for a in substrate_anchors] if s],
            },
            "extra": {
                "substrate_labels": [label for label, _, _ in substrate_anchors],
            },
        }
    elif plot_name == "wyner_ziv_flow":
        sha = archive_shas[0]
        aggregate, anchor = _resolve_anchor_for_plot(sha, require_per_pair=False)
        plot_wyner_ziv_flow(aggregate, anchor, output)
        return {
            "anchor": anchor,
            "summary": _summary_stats_for_aggregate(aggregate),
            "extra": {
                "has_section_manifest": bool(
                    isinstance(anchor.get("archive_layout"), dict)
                    and anchor["archive_layout"].get("sections")
                ),
            },
        }
    elif plot_name == "drift_vs_sensitivity_scatter":
        if mps_drift_data is None:
            raise SystemExit(
                f"plot {plot_name!r} requires --mps-drift-json"
            )
        sha = archive_shas[0]
        aggregate, anchor = _resolve_anchor_for_plot(sha, require_per_pair=False)
        per_pair_result = _resolve_anchor_for_plot(sha, require_per_pair=True, soft=True)
        per_pair = per_pair_result[0] if per_pair_result is not None else None
        plot_drift_vs_sensitivity_scatter(
            per_pair, aggregate, anchor, mps_drift_data, output
        )
        return {
            "anchor": anchor,
            "summary": _summary_stats_for_aggregate(aggregate),
            "extra": {
                "mps_drift_axis_tag": mps_drift_data.get("axis_tag"),
                "mps_drift_n_pairs": mps_drift_data.get("n_pairs"),
                "mps_drift_evidence_grade": mps_drift_data.get("evidence_grade"),
                "has_per_pair_gradient": per_pair is not None,
            },
        }
    else:
        raise SystemExit(f"unknown plot name: {plot_name!r}")


def _refuse_tmp_path(p: Path, flag_name: str) -> None:
    """Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact"."""
    text = str(p)
    if "/tmp/" in text or text.startswith("/tmp") or text.startswith("/private/tmp"):
        raise SystemExit(
            f"/tmp output paths are FORBIDDEN per CLAUDE.md "
            f'"Forbidden /tmp paths in any persisted artifact"; '
            f"pass an explicit non-/tmp {flag_name} "
            "(e.g. reports/master_gradient_xray/<archive>/)"
        )


def _emit_output_dir_mode(
    output_dir: Path,
    archive_shas: list[str],
    mps_drift_data: dict | None,
    mps_drift_json_path: Path | None,
) -> tuple[list[dict[str, str]], list[dict]]:
    """Emit all canonical plots + sister JSONs + index.html.

    Per Catalog #305 observability + Catalog #323 Provenance: every plot
    gets a sister JSON; every sister JSON carries the canonical Provenance
    sub-object built via build_provenance_for_predicted (visualization is a
    derived view, NOT a primary measurement).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_to_emit = list(CANONICAL_OUTPUT_DIR_PLOTS)
    if mps_drift_data is not None:
        plots_to_emit.append("drift_vs_sensitivity_scatter")

    emitted: list[dict[str, str]] = []
    anchors_seen: list[dict] = []
    archive_to_anchor: dict[str, dict] = {}

    for plot_name in plots_to_emit:
        png_path = output_dir / f"{plot_name}.png"
        sidecar_path = output_dir / f"{plot_name}.json"
        try:
            result = _emit_plot(
                plot_name, archive_shas, png_path, mps_drift_data=mps_drift_data
            )
        except SystemExit as exc:
            print(f"  SKIP {plot_name}: {exc}", file=sys.stderr)
            continue

        if result is not None:
            sha = result["anchor"].get("archive_sha256")
            if sha and sha not in archive_to_anchor:
                archive_to_anchor[sha] = result["anchor"]
            _emit_sidecar_json(
                sidecar_path,
                plot_id=plot_name,
                anchor=result["anchor"],
                summary=result["summary"],
                extra=result["extra"],
            )
            emitted.append(
                {
                    "plot_id": plot_name,
                    "png_relative": png_path.name,
                    "sidecar_relative": sidecar_path.name,
                }
            )
            print(f"  wrote {png_path}")
            print(f"  wrote {sidecar_path}")
        else:
            print(f"  wrote {png_path} (no sidecar)")

    # Anchors list keeps caller-supplied ordering
    for sha in archive_shas:
        anchor = archive_to_anchor.get(sha)
        if anchor is None:
            # Attempt soft load for index display
            result = _resolve_anchor_for_plot(sha, require_per_pair=False, soft=True)
            anchor = (result[1] if result is not None else {"archive_sha256": sha})
        anchors_seen.append(anchor)

    _emit_index_html(
        output_dir,
        archive_shas=archive_shas,
        anchors=anchors_seen,
        emitted_plots=emitted,
        mps_drift_json_path=mps_drift_json_path,
    )
    print(f"  wrote {output_dir / 'index.html'}")
    return emitted, anchors_seen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Master-gradient xray visualization tool — 5 canonical plot "
            "types + optional drift cross-reference. Per CLAUDE.md no /tmp "
            "paths; pass an explicit --output or --output-dir."
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
        "--anchor-jsonl",
        required=False,
        default=str((REPO_ROOT / ".omx/state/master_gradient_anchors.jsonl")),
        help=(
            "Path to the master-gradient anchor JSONL ledger (default: "
            ".omx/state/master_gradient_anchors.jsonl). The tool resolves "
            "anchors via tac.master_gradient_consumers which reads from "
            "this ledger."
        ),
    )
    parser.add_argument(
        "--mps-drift-json",
        required=False,
        help=(
            "Optional cross-reference: granular MPS-vs-CUDA drift JSON "
            "(e.g. .omx/state/mps_drift_granular_<utc>.json). When provided, "
            "drift_vs_sensitivity_scatter plot is emitted in --output-dir "
            "mode."
        ),
    )
    parser.add_argument(
        "--plot",
        choices=CANONICAL_PLOTS,
        default=None,
        help=(
            "Plot type to emit. With --output-dir, defaults to 'all' "
            "(emits canonical 5 plots + index.html). With --output, a "
            "single plot is required."
        ),
    )
    parser.add_argument(
        "--output",
        required=False,
        help=(
            "Output path for single-plot mode (.png). Caller MUST pass an "
            "explicit non-/tmp path per CLAUDE.md."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=False,
        help=(
            "Output directory mode: emits canonical 5 plots + sister JSON "
            "sidecars (per Catalog #305 + #323) + index.html landing page. "
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
                    "output_dir_canonical_plots": list(CANONICAL_OUTPUT_DIR_PLOTS),
                    "description": (
                        "master-gradient xray 5 canonical plot types + optional "
                        "drift_vs_sensitivity_scatter when --mps-drift-json is "
                        "provided"
                    ),
                    "sidecar_schema_version": PLOT_SIDECAR_SCHEMA_VERSION,
                    "index_html_schema_version": INDEX_HTML_SCHEMA_VERSION,
                },
                indent=2,
            )
        )
        return 0

    if not args.output and not args.output_dir:
        raise SystemExit(
            "--output (single plot file) OR --output-dir (5 plots + index) "
            "is required when not using --list-plots"
        )
    if args.output and args.output_dir:
        raise SystemExit("--output and --output-dir are mutually exclusive")
    if not args.archive_sha:
        raise SystemExit(
            "--archive-sha is required (specify at least one for "
            "single-substrate plots; multiple for cross_substrate_correlation)"
        )

    # MPS drift JSON loaded eagerly so both modes can use it
    mps_drift_data: dict | None = None
    mps_drift_json_path: Path | None = None
    if args.mps_drift_json:
        mps_drift_json_path = Path(args.mps_drift_json).resolve()
        mps_drift_data = _load_mps_drift_json(mps_drift_json_path)

    if args.output_dir:
        output_dir = Path(args.output_dir).resolve()
        _refuse_tmp_path(output_dir, "--output-dir")
        if args.plot and args.plot not in ("all", None):
            print(
                f"WARN: --plot {args.plot!r} ignored in --output-dir mode "
                "(emitting canonical 5 plots + index.html)",
                file=sys.stderr,
            )
        _emit_output_dir_mode(
            output_dir,
            args.archive_sha,
            mps_drift_data=mps_drift_data,
            mps_drift_json_path=mps_drift_json_path,
        )
        return 0

    # --output single-plot mode (legacy entry; preserves Cable D semantics)
    plot_name = args.plot or "all"
    output_path = Path(args.output).resolve()
    _refuse_tmp_path(output_path, "--output")

    if plot_name == "all":
        if not output_path.is_dir() and output_path.suffix == "":
            output_path.mkdir(parents=True, exist_ok=True)
        elif not output_path.is_dir():
            raise SystemExit(
                f"--plot all requires --output to be a directory; "
                f"got file {output_path}. Use --output-dir instead."
            )
        for canonical_plot in CANONICAL_OUTPUT_DIR_PLOTS:
            target = output_path / f"{canonical_plot}.png"
            try:
                _emit_plot(
                    canonical_plot,
                    args.archive_sha,
                    target,
                    mps_drift_data=mps_drift_data,
                )
                print(f"  wrote {target}")
            except SystemExit as exc:
                # Per-plot failures don't abort the batch; report and continue
                print(f"  SKIP {canonical_plot}: {exc}", file=sys.stderr)
        if mps_drift_data is not None:
            target = output_path / "drift_vs_sensitivity_scatter.png"
            try:
                _emit_plot(
                    "drift_vs_sensitivity_scatter",
                    args.archive_sha,
                    target,
                    mps_drift_data=mps_drift_data,
                )
                print(f"  wrote {target}")
            except SystemExit as exc:
                print(
                    f"  SKIP drift_vs_sensitivity_scatter: {exc}", file=sys.stderr
                )
    else:
        if output_path.is_dir():
            output_path = output_path / f"{plot_name}.png"
        _emit_plot(
            plot_name, args.archive_sha, output_path, mps_drift_data=mps_drift_data
        )
        print(f"wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
