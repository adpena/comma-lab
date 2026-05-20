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
from tac.continual_learning import NON_PROMOTABLE_TAGS as _CANONICAL_NON_PROMOTABLE_TAGS  # noqa: E402


# Catalog #341 widening per Slot JJ Path B (2026-05-19): xray tool's
# verdict-matrix Catalog #341 strict check at consumer_verdict_matrix
# accepts ANY axis_tag in the canonical non-promotable set, NOT only the
# narrow "[predicted]" string. The canonical sources of truth are
# ``src/tac/continual_learning.py::NON_PROMOTABLE_TAGS`` (8 entries:
# [macOS-CPU advisory only] / [macOS-CPU calibrated] / [MPS-PROXY] /
# [MPS-research-signal] / [advisory only] / [distortion-proxy:local] /
# [byte-anchor] / [predicted; unified-action; closed-form weighted-sum])
# AND ``src/tac/provenance/contract.py::CANONICAL_MEASUREMENT_AXES``
# (includes [MPS-PROXY]).  The narrow "[predicted]" string is the
# default for consumers that ONLY annotate predicted-helper-availability
# without proxy/diagnostic source; the canonical apparatus mandates the
# 2 MPS consumers (mps_diagnostic_consumer / mps_gap_experiment_consumer)
# use [MPS-PROXY] per CLAUDE.md "MPS auth eval is NOISE" Rule 3 + the
# FORBIDDEN PATTERN "Forbidden MPS-derived strategic decision". Widening
# the strict check to ``axis_tag in CANONICAL_NON_PROMOTABLE_AXES`` brings
# this tool into alignment with both canonical sources. Cross-ref Slot II
# classification memo ``.omx/research/catalog_341_noncompliance_classification_20260519.md``.
CANONICAL_NON_PROMOTABLE_AXES: frozenset[str] = frozenset(
    {"[predicted]"} | set(_CANONICAL_NON_PROMOTABLE_TAGS)
)


CANONICAL_PLOTS = (
    "per_pair_distribution",
    "per_byte_heatmap",
    "cumulative_by_rank",
    "cross_substrate_correlation",
    "wyner_ziv_flow",
    "drift_vs_sensitivity_scatter",
    "cascade_smearing_comparison",
    # Slot EE 2026-05-19 extensions per task #797 operator spec:
    "consumer_verdict_matrix",
    "provenance_audit_timeline",
    "all",
)

# Canonical 5-plot taxonomy emitted by `--output-dir`/`--plot all` when no
# `--mps-drift-json` is provided. drift_vs_sensitivity_scatter is the
# optional 6th plot that only fires when the cross-reference JSON is
# available. cascade_smearing_comparison (7th) fires when --grain
# compare_both AND an archive has BOTH raw-byte AND post-decompress
# anchors (one cascade comparison plot per eligible archive).
CANONICAL_OUTPUT_DIR_PLOTS = (
    "per_pair_distribution",
    "per_byte_heatmap",
    "cumulative_by_rank",
    "cross_substrate_correlation",
    "wyner_ziv_flow",
)

# Canonical grain-filter values for the --grain CLI flag (slot 10
# grain-awareness landing 2026-05-19; sister of slot 6 cathedral consumer
# v1.1 + slot 15 PR101 post-decompress producer + slot 17 5-family
# extension).
GRAIN_FILTER_RAW_BYTE = "raw_byte"
GRAIN_FILTER_POST_DECOMPRESS = "post_decompress"
GRAIN_FILTER_COMPARE_BOTH = "compare_both"
GRAIN_FILTER_ALL = "all"
GRAIN_FILTER_CHOICES = (
    GRAIN_FILTER_RAW_BYTE,
    GRAIN_FILTER_POST_DECOMPRESS,
    GRAIN_FILTER_COMPARE_BOTH,
    GRAIN_FILTER_ALL,
)

AXIS_LABELS = ("seg", "pose", "rate")

# Sister JSON schema for plot summary stats. Bumped to v2 for grain-aware
# routing per slot 6 + slot 10 grain-awareness landing 2026-05-19
# (cascade_smearing_comparison plot + per-grain anchor metadata).
PLOT_SIDECAR_SCHEMA_VERSION = "master_gradient_xray_plot_sidecar_v2_20260519"
INDEX_HTML_SCHEMA_VERSION = "master_gradient_xray_index_v2_20260519"


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
# Plot 7 — Cascade-smearing comparison (raw-byte vs post-decompress)           #
# ──────────────────────────────────────────────────────────────────────────── #


def _compute_cascade_smearing_metrics(
    raw_byte_aggregate: np.ndarray,
    post_decompress_aggregate: np.ndarray,
    top_k: int = 100,
) -> dict[str, object]:
    """Compute cascade-smearing comparison metrics between two grain anchors.

    Returns a dict with:

    * ``top_k_jaccard`` — Jaccard overlap of the top-K byte indices ranked
      by L1-sum-of-abs sensitivity in each grain. 1.0 = identical top-K;
      0.0 = disjoint. LOW Jaccard = HIGH cascade smearing (the raw-byte
      gradient identifies completely different "important" bytes than the
      post-decompress gradient, evidence the raw-byte signal is misleading
      per Catalog #318).
    * ``rank_correlation_spearman`` — Spearman rank correlation between
      per-byte L1 magnitudes (over the SHORTER of the two arrays). 1.0 =
      identical rank order; -1.0 = reversed.
    * ``cascade_smearing_factor`` — derived classifier in [0, 1]:
      ``1.0 - top_k_jaccard``. Bands per the analysis: HIGH >= 0.7,
      MEDIUM 0.3-0.7, LOW < 0.3.
    * ``verdict`` — categorical classification (HIGH / MEDIUM / LOW /
      UNKNOWN_INSUFFICIENT_DATA).

    Note: raw_byte_aggregate and post_decompress_aggregate generally have
    DIFFERENT N_bytes (post-decompress is the inflated stream; raw-byte
    is the compressed stream). The Jaccard is computed on top-K byte
    indices treated as opaque labels (different indexings); rank
    correlation is on min-truncated arrays as a coarse proxy.
    """
    abs_raw = np.abs(raw_byte_aggregate).sum(axis=1)
    abs_post = np.abs(post_decompress_aggregate).sum(axis=1)

    k_effective = min(top_k, abs_raw.shape[0], abs_post.shape[0])
    if k_effective <= 0:
        return {
            "top_k_jaccard": 0.0,
            "rank_correlation_spearman": 0.0,
            "cascade_smearing_factor": 1.0,
            "verdict": "UNKNOWN_INSUFFICIENT_DATA",
            "top_k": 0,
            "n_bytes_raw": int(abs_raw.shape[0]),
            "n_bytes_post": int(abs_post.shape[0]),
        }

    top_raw = set(int(i) for i in np.argsort(-abs_raw)[:k_effective])
    top_post = set(int(i) for i in np.argsort(-abs_post)[:k_effective])
    intersection = len(top_raw & top_post)
    union = len(top_raw | top_post)
    jaccard = intersection / union if union > 0 else 0.0

    # Spearman rank correlation on min-truncated arrays.
    min_len = min(abs_raw.shape[0], abs_post.shape[0])
    if min_len >= 2:
        ranks_raw = np.argsort(np.argsort(abs_raw[:min_len])).astype(np.float64)
        ranks_post = np.argsort(np.argsort(abs_post[:min_len])).astype(np.float64)
        rm = ranks_raw - ranks_raw.mean()
        pm = ranks_post - ranks_post.mean()
        denom = float(np.sqrt((rm**2).sum() * (pm**2).sum()))
        spearman = float((rm * pm).sum() / denom) if denom > 0 else 0.0
    else:
        spearman = 0.0

    cascade_factor = 1.0 - jaccard
    if cascade_factor >= 0.7:
        verdict = "HIGH"
    elif cascade_factor >= 0.3:
        verdict = "MEDIUM"
    else:
        verdict = "LOW"

    return {
        "top_k_jaccard": float(jaccard),
        "rank_correlation_spearman": spearman,
        "cascade_smearing_factor": float(cascade_factor),
        "verdict": verdict,
        "top_k": int(k_effective),
        "n_bytes_raw": int(abs_raw.shape[0]),
        "n_bytes_post": int(abs_post.shape[0]),
        "intersection_count": int(intersection),
        "union_count": int(union),
    }


def plot_cascade_smearing_comparison(
    raw_byte_aggregate: np.ndarray,
    raw_byte_anchor: dict,
    post_decompress_aggregate: np.ndarray,
    post_decompress_anchor: dict,
    output_path: Path,
    top_k: int = 128,
) -> dict[str, object]:
    """Side-by-side cascade-smearing comparison plot (Catalog #318 anchor).

    Emits a 2-panel heatmap (raw-byte LEFT, post-decompress RIGHT) of the
    top-K bytes by L1-sum-of-abs sensitivity in each grain. Watermarks
    annotate the per-archive verdict + top-K-overlap Jaccard +
    cascade-smearing factor + categorical band (HIGH / MEDIUM / LOW).

    Returns the metrics dict from `_compute_cascade_smearing_metrics` so
    the caller can persist them in the sister JSON sidecar.

    Per CLAUDE.md "Apples-to-apples evidence discipline": the side-by-side
    layout IS the empirical disambiguator — operators can see the
    entropy-cascade-smeared raw-byte heatmap diverging from the
    locality-correct post-decompress heatmap at a glance.
    """
    plt = _ensure_matplotlib()
    metrics = _compute_cascade_smearing_metrics(
        raw_byte_aggregate, post_decompress_aggregate, top_k=top_k
    )

    raw_sha = _short_sha(raw_byte_anchor.get("archive_sha256"))
    post_sha = _short_sha(post_decompress_anchor.get("archive_sha256"))
    raw_watermark = _watermark_for_anchor(raw_byte_anchor)
    post_watermark = _watermark_for_anchor(post_decompress_anchor)
    raw_grain = raw_byte_anchor.get("gradient_byte_domain", "unknown")
    post_grain = post_decompress_anchor.get("gradient_byte_domain", "unknown")

    fig, axes = plt.subplots(1, 2, figsize=(13, 11))

    for ax, aggregate, watermark, grain_label, sha in (
        (axes[0], raw_byte_aggregate, raw_watermark, raw_grain, raw_sha),
        (
            axes[1],
            post_decompress_aggregate,
            post_watermark,
            post_grain,
            post_sha,
        ),
    ):
        n_bytes_local = aggregate.shape[0]
        abs_grad = np.abs(aggregate)
        per_byte_l1 = abs_grad.sum(axis=1)
        top_k_local = min(top_k, n_bytes_local)
        top_indices = np.argsort(-per_byte_l1)[:top_k_local]
        heat = abs_grad[top_indices]
        heat_norm = heat / np.maximum(heat.max(axis=0, keepdims=True), 1e-30)
        im = ax.imshow(
            heat_norm, aspect="auto", cmap="hot", interpolation="nearest"
        )
        ax.set_xticks(range(3))
        ax.set_xticklabels(AXIS_LABELS)
        ax.set_xlabel("score axis")
        ax.set_ylabel(f"byte rank (top {top_k_local})")
        ax.set_title(
            f"{grain_label}\n{watermark}\narchive={sha}  n_bytes={n_bytes_local}",
            fontsize=9,
        )
        fig.colorbar(im, ax=ax, label="normalized |gradient|", fraction=0.04)

    verdict = metrics["verdict"]
    fig.suptitle(
        f"Cascade-smearing comparison — verdict={verdict}\n"
        f"top-{metrics['top_k']} Jaccard={metrics['top_k_jaccard']:.3f}  "
        f"cascade_smearing_factor={metrics['cascade_smearing_factor']:.3f}  "
        f"rank_corr={metrics['rank_correlation_spearman']:.3f}",
        fontsize=11,
    )
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)
    return metrics


# ──────────────────────────────────────────────────────────────────────────── #
# Plot 8 — cathedral consumer verdict matrix                                   #
# (Slot EE 2026-05-19 extension per task #797 operator spec — sister of the   #
# 35-consumer Cable D auto-discovery surface; per Catalog #341 routing        #
# markers + Catalog #335 canonical contract)                                  #
# ──────────────────────────────────────────────────────────────────────────── #


def _collect_consumer_verdicts(
    candidate: dict,
    *,
    consumer_modules: list | None = None,
) -> list[dict[str, object]]:
    """Invoke `consume_candidate` on every auto-discovered consumer.

    Returns a list of per-consumer verdict dicts with structured fields for
    matrix rendering. Pure observability per Catalog #341 + #335: NEVER
    promotes a routing recommendation into a score signal; markers
    (`predicted_delta_adjustment=0.0` / `promotable=False` /
    `axis_tag="[predicted]"`) are surfaced verbatim from the consumer's own
    return value so the matrix is faithful to the canonical contract.
    """
    if consumer_modules is None:
        # `tools/` is not a package on sys.path; load the canonical
        # cathedral_autopilot_autonomous_loop module by file path per the
        # same pattern other operator-facing tools use. Register in
        # sys.modules BEFORE exec to satisfy dataclass machinery.
        import importlib.util as _ilu  # noqa: E402
        _mod_name = "cathedral_autopilot_autonomous_loop_xray_loader"
        if _mod_name in sys.modules:
            _cathedral_mod = sys.modules[_mod_name]
        else:
            _cathedral_loop_path = REPO_ROOT / "tools" / "cathedral_autopilot_autonomous_loop.py"
            _spec = _ilu.spec_from_file_location(
                _mod_name, _cathedral_loop_path
            )
            if _spec is None or _spec.loader is None:
                raise SystemExit(
                    f"plot consumer_verdict_matrix: cannot load "
                    f"{_cathedral_loop_path}; verify file exists per Catalog #335 + #336."
                )
            _cathedral_mod = _ilu.module_from_spec(_spec)
            sys.modules[_mod_name] = _cathedral_mod  # register BEFORE exec
            _spec.loader.exec_module(_cathedral_mod)
        consumer_modules = _cathedral_mod.discover_compliant_consumer_modules()
    verdicts: list[dict[str, object]] = []
    for mod in consumer_modules:
        consumer_name = getattr(mod, "CONSUMER_NAME", mod.__name__.rsplit(".", 1)[-1])
        consumer_version = getattr(mod, "CONSUMER_VERSION", "unknown")
        hook_numbers = getattr(mod, "CONSUMER_HOOK_NUMBERS", ())
        verdict_row: dict[str, object] = {
            "consumer_name": consumer_name,
            "consumer_version": str(consumer_version),
            "hook_numbers": [int(h) for h in hook_numbers],
            "error": None,
        }
        try:
            result = mod.consume_candidate(candidate)
            if isinstance(result, dict):
                verdict_row.update({
                    "predicted_delta_adjustment": float(
                        result.get("predicted_delta_adjustment", 0.0)
                    ),
                    "promotable": bool(result.get("promotable", False)),
                    "axis_tag": str(result.get("axis_tag", "[predicted]")),
                    "confidence": float(result.get("confidence", 0.0)),
                    "rationale": str(result.get("rationale", "")),
                })
                # Catalog #341 marker compliance check (3 canonical markers).
                # Slot JJ Path B (2026-05-19): widened from narrow
                # ``axis_tag == "[predicted]"`` to ``axis_tag in
                # CANONICAL_NON_PROMOTABLE_AXES`` per Slot II classification
                # memo + canonical Provenance/NON_PROMOTABLE_TAGS surfaces.
                markers_compliant = (
                    verdict_row["predicted_delta_adjustment"] == 0.0
                    and verdict_row["promotable"] is False
                    and verdict_row["axis_tag"] in CANONICAL_NON_PROMOTABLE_AXES
                )
                verdict_row["catalog_341_markers_compliant"] = markers_compliant
                verdict_row["non_vacuous"] = bool(
                    verdict_row["rationale"]
                    and verdict_row["rationale"] not in ("", "no anchor lookup attempted")
                )
            else:
                verdict_row.update({
                    "error": f"non-dict return: {type(result).__name__}",
                    "catalog_341_markers_compliant": False,
                    "non_vacuous": False,
                })
        except (FileNotFoundError, ValueError, OSError, KeyError, AttributeError) as exc:
            verdict_row.update({
                "error": f"{type(exc).__name__}: {exc}",
                "catalog_341_markers_compliant": False,
                "non_vacuous": False,
            })
        verdicts.append(verdict_row)
    return verdicts


def plot_consumer_verdict_matrix(
    candidate: dict,
    output_path: Path,
    *,
    consumer_modules: list | None = None,
) -> dict[str, object]:
    """Render the per-consumer × per-candidate verdict matrix.

    The matrix shows, for one candidate, which of the auto-discovered
    cathedral consumers fire NON-VACUOUS verdicts (a real rationale; not
    just "no anchor lookup attempted") AND which carry the Catalog #341
    canonical 3 routing markers (predicted_delta_adjustment=0.0 /
    promotable=False / axis_tag="[predicted]").

    Per Catalog #335 the consumer surface is auto-discovered; this plot
    operationalizes the live count empirically per CLAUDE.md
    "Max observability — non-negotiable" + Catalog #305 6-facet
    observability.

    Per Catalog #318 raw-byte-authority guard: this plot NEVER surfaces
    raw byte tensors; the matrix encodes consumer-level verdict booleans
    + hook coverage only.

    Returns the metrics dict (consumer_count + non_vacuous_count + markers
    compliance rate + per-hook-coverage histogram) for the sister JSON
    sidecar.
    """
    plt = _ensure_matplotlib()
    verdicts = _collect_consumer_verdicts(candidate, consumer_modules=consumer_modules)
    n_consumers = len(verdicts)
    if n_consumers == 0:
        raise SystemExit(
            "plot consumer_verdict_matrix: 0 compliant consumers discovered; "
            "verify src/tac/cathedral_consumers/* + Catalog #335 contract."
        )

    # Build matrix: rows = consumers (sorted by hook count desc then name)
    # cols = (non_vacuous, catalog_341_compliant, promotable)
    verdicts_sorted = sorted(
        verdicts,
        key=lambda v: (-len(v.get("hook_numbers", [])), str(v.get("consumer_name", ""))),
    )
    matrix_rows: list[list[float]] = []
    consumer_labels: list[str] = []
    for v in verdicts_sorted:
        row = [
            1.0 if v.get("non_vacuous") else 0.0,
            1.0 if v.get("catalog_341_markers_compliant") else 0.0,
            1.0 if v.get("promotable") else 0.0,  # always 0 for canonical consumers
            1.0 if v.get("error") is None else 0.0,
        ]
        matrix_rows.append(row)
        hooks_str = ",".join(str(h) for h in v.get("hook_numbers", [])) or "—"
        consumer_labels.append(f"{v.get('consumer_name', '?')[:42]} [h:{hooks_str}]")

    matrix = np.array(matrix_rows, dtype=np.float32)
    col_labels = ["non-vacuous", "Catalog #341 markers", "promotable (should=0)", "no-error"]
    candidate_id = str(candidate.get("candidate_id", "(unnamed)"))[:32]
    archive_sha = candidate.get("archive_sha256", "")
    archive_short = _short_sha(archive_sha) if archive_sha else "no-archive-sha"

    fig_height = max(8.0, 0.32 * n_consumers + 2.0)
    fig, ax = plt.subplots(figsize=(11, fig_height))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0.0, vmax=1.0,
                   interpolation="nearest")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right", fontsize=8)
    ax.set_yticks(range(n_consumers))
    ax.set_yticklabels(consumer_labels, fontsize=6)
    ax.set_xlabel("verdict dimension")
    ax.set_title(
        f"Cathedral consumer verdict matrix — candidate={candidate_id}\n"
        f"archive={archive_short}  n_consumers={n_consumers}  "
        f"per Catalog #335 + #341 [predicted]",
        fontsize=10,
    )
    fig.colorbar(im, ax=ax, label="verdict (0=red / 1=green)", fraction=0.025)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

    # Sister-summary metrics
    non_vacuous_count = sum(1 for v in verdicts if v.get("non_vacuous"))
    markers_compliant_count = sum(1 for v in verdicts if v.get("catalog_341_markers_compliant"))
    promotable_violation_count = sum(1 for v in verdicts if v.get("promotable"))
    error_count = sum(1 for v in verdicts if v.get("error") is not None)
    hook_histogram: dict[str, int] = {}
    for v in verdicts:
        for h in v.get("hook_numbers", []):
            hook_histogram[str(int(h))] = hook_histogram.get(str(int(h)), 0) + 1
    return {
        "n_consumers": n_consumers,
        "non_vacuous_count": non_vacuous_count,
        "non_vacuous_fraction": (
            float(non_vacuous_count) / n_consumers if n_consumers else 0.0
        ),
        "catalog_341_markers_compliant_count": markers_compliant_count,
        "catalog_341_markers_compliant_fraction": (
            float(markers_compliant_count) / n_consumers if n_consumers else 0.0
        ),
        "promotable_violation_count": promotable_violation_count,
        "error_count": error_count,
        "hook_coverage_histogram": hook_histogram,
        "candidate_id": candidate_id,
        "archive_sha256": archive_sha,
        "per_consumer_verdicts": [
            {
                "consumer_name": v["consumer_name"],
                "non_vacuous": v.get("non_vacuous"),
                "catalog_341_markers_compliant": v.get("catalog_341_markers_compliant"),
                "promotable": v.get("promotable"),
                "axis_tag": v.get("axis_tag"),
                "error": v.get("error"),
            }
            for v in verdicts_sorted
        ],
    }


# ──────────────────────────────────────────────────────────────────────────── #
# Plot 9 — provenance audit timeline                                           #
# (Slot EE 2026-05-19 extension per task #797 operator spec — sister of       #
# Catalog #323 canonical Provenance umbrella + Catalog #287/#327 axis +       #
# hardware + custody discipline; surfaces canonical 6-tuple completeness     #
# per anchor row in chronological order)                                     #
# ──────────────────────────────────────────────────────────────────────────── #


CANONICAL_PROVENANCE_KEYS = (
    "archive_sha256",
    "measurement_axis",
    "measurement_hardware",
    "measurement_utc",
    "measurement_call_id",
    "measurement_method",
)


def _classify_provenance_for_anchor(anchor: dict) -> dict[str, object]:
    """Classify one anchor row by canonical Provenance completeness.

    Returns dict with `complete` bool + `missing_keys` list + `axis_tag` +
    `is_authoritative` + `reject_reason` (None if accepted). Mirrors the
    Catalog #287/#323 sister contract pattern.

    Per Catalog #318 we do NOT touch raw byte tensors here; the audit
    operates over the anchor METADATA only.
    """
    missing_keys: list[str] = []
    for key in CANONICAL_PROVENANCE_KEYS:
        val = anchor.get(key)
        if val is None or (isinstance(val, str) and not val.strip()):
            missing_keys.append(key)
    # Importing here defers the dependency for unit-testability.
    try:
        from tac.master_gradient import (  # noqa: E402
            contest_axis_authority_violation_reason,
            is_authoritative_axis_anchor,
        )
        is_auth = bool(is_authoritative_axis_anchor(anchor))
        reject_reason = contest_axis_authority_violation_reason(anchor)
    except ImportError:
        is_auth = False
        reject_reason = "tac.master_gradient unavailable"
    return {
        "complete": not missing_keys,
        "missing_keys": missing_keys,
        "axis_tag": str(anchor.get("measurement_axis", "unknown")),
        "is_authoritative": is_auth,
        "reject_reason": reject_reason,
    }


def plot_provenance_audit_timeline(
    anchors: list[dict],
    output_path: Path,
    *,
    since_utc: str | None = None,
) -> dict[str, object]:
    """Chronological timeline of per-row Provenance completeness.

    Each anchor row is plotted as a vertical bar at its `measurement_utc`,
    colored by (a) Provenance completeness (all 6 canonical keys present)
    AND (b) authoritative-axis status per `is_authoritative_axis_anchor`.

    Color taxonomy:
    - GREEN: complete + authoritative
    - YELLOW: complete + non-authoritative (advisory / proxy)
    - RED: incomplete (REJECT verdict per Catalog #323 validate_provenance)

    Per Catalog #305 6-facet observability: this plot is the canonical
    operator-facing audit surface for ledger Provenance health over time.
    Per Catalog #110/#113: ledger rows are APPEND-ONLY; the timeline is
    historical truth that NEVER mutates.

    Per Catalog #287 placeholder-rationale rejection: rows with empty or
    "<rationale>" / "<reason>" measurement_method are surfaced as REJECT.

    Returns sister-summary metrics dict.
    """
    plt = _ensure_matplotlib()
    # Filter by since_utc if provided.
    if since_utc:
        anchors = [
            a for a in anchors
            if str(a.get("measurement_utc", "")) >= since_utc
            or str(a.get("written_at_utc", "")) >= since_utc
        ]
    if not anchors:
        raise SystemExit(
            "plot provenance_audit_timeline: 0 anchors in scope; "
            f"check ledger or relax --since-utc {since_utc!r}"
        )

    # Sort by measurement_utc ascending (fallback to written_at_utc).
    def _ts(a: dict) -> str:
        return str(a.get("measurement_utc") or a.get("written_at_utc") or "")
    anchors_sorted = sorted(anchors, key=_ts)
    n_anchors = len(anchors_sorted)

    classifications = [_classify_provenance_for_anchor(a) for a in anchors_sorted]

    # Build bar plot: x = index, y = boolean stacks
    x_positions = list(range(n_anchors))
    colors = []
    bar_heights = []
    for cls in classifications:
        if not cls["complete"]:
            colors.append("tab:red")
            bar_heights.append(0.4)  # short red bar = REJECT
        elif cls["is_authoritative"]:
            colors.append("tab:green")
            bar_heights.append(1.0)  # full green bar = ACCEPT-AUTHORITATIVE
        else:
            colors.append("gold")
            bar_heights.append(0.7)  # medium yellow = ACCEPT-ADVISORY

    fig, ax = plt.subplots(figsize=(max(12.0, 0.45 * n_anchors), 6))
    ax.bar(x_positions, bar_heights, color=colors, edgecolor="black", linewidth=0.4)

    # Tick labels = sha[:8] + measurement_utc[:19]
    labels = []
    for a in anchors_sorted:
        sha = _short_sha(a.get("archive_sha256"))
        utc = str(a.get("measurement_utc") or a.get("written_at_utc") or "")[:19]
        labels.append(f"{sha}\n{utc}")
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, rotation=70, ha="right", fontsize=6)
    ax.set_ylabel("Provenance verdict (1=auth / 0.7=advisory / 0.4=REJECT)")
    ax.set_ylim(0, 1.15)
    ax.set_title(
        f"Provenance audit timeline — n_anchors={n_anchors}  "
        f"green=auth  gold=advisory  red=incomplete\n"
        f"per Catalog #287/#323 canonical Provenance + #327 contest-axis custody",
        fontsize=10,
    )

    # Legend
    from matplotlib.patches import Patch  # noqa: E402
    legend_handles = [
        Patch(color="tab:green", label="complete + authoritative"),
        Patch(color="gold", label="complete + advisory (non-promotable)"),
        Patch(color="tab:red", label="incomplete (REJECT per Catalog #323)"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)

    # Sister-summary metrics
    complete_count = sum(1 for c in classifications if c["complete"])
    authoritative_count = sum(1 for c in classifications if c["is_authoritative"])
    reject_count = sum(1 for c in classifications if not c["complete"])
    # Categorize REJECTs
    reject_categories: dict[str, int] = {}
    for c, a in zip(classifications, anchors_sorted):
        if not c["complete"]:
            cat = "missing_keys:" + ",".join(c["missing_keys"][:3])
            reject_categories[cat] = reject_categories.get(cat, 0) + 1
        elif c.get("reject_reason"):
            cat = f"axis_violation:{c['reject_reason'][:48]}"
            reject_categories[cat] = reject_categories.get(cat, 0) + 1
    # Axis breakdown
    axis_histogram: dict[str, int] = {}
    for c in classifications:
        ax_tag = c["axis_tag"]
        axis_histogram[ax_tag] = axis_histogram.get(ax_tag, 0) + 1
    return {
        "n_anchors": n_anchors,
        "complete_count": complete_count,
        "complete_fraction": (
            float(complete_count) / n_anchors if n_anchors else 0.0
        ),
        "authoritative_count": authoritative_count,
        "authoritative_fraction": (
            float(authoritative_count) / n_anchors if n_anchors else 0.0
        ),
        "reject_count": reject_count,
        "reject_categories": reject_categories,
        "axis_histogram": axis_histogram,
        "since_utc": since_utc,
        "first_anchor_utc": str(
            anchors_sorted[0].get("measurement_utc")
            or anchors_sorted[0].get("written_at_utc")
            or ""
        )[:19],
        "last_anchor_utc": str(
            anchors_sorted[-1].get("measurement_utc")
            or anchors_sorted[-1].get("written_at_utc")
            or ""
        )[:19],
    }


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
    grain: str | None = None,
    cascade_comparison_eligible: list[str] | None = None,
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

    # Grain-aware: surface the grain inventory per archive so the operator
    # can see which archives have BOTH raw-byte + post-decompress anchors.
    try:
        from tac.master_gradient_per_byte_consumer import (
            available_grains_for_archive,
        )
    except ImportError:
        available_grains_for_archive = None  # type: ignore[assignment]

    archive_rows = []
    cascade_eligible_set = set(cascade_comparison_eligible or [])
    for sha, anchor in zip(archive_shas, anchors):
        grain_inv_str = "—"
        if available_grains_for_archive is not None:
            inv = available_grains_for_archive(sha)
            parts = []
            if inv["post_decompress"]:
                parts.append(f"post×{len(inv['post_decompress'])}")
            if inv["raw_byte"]:
                parts.append(f"raw×{len(inv['raw_byte'])}")
            if inv["other"]:
                parts.append(f"other×{len(inv['other'])}")
            grain_inv_str = " + ".join(parts) if parts else "—"
        cascade_flag = "✓" if sha in cascade_eligible_set else ""
        archive_rows.append(
            f"<tr><td><code>{_h(_short_sha(sha))}</code></td>"
            f"<td>{_h(anchor.get('measurement_axis'))}</td>"
            f"<td>{_h(anchor.get('measurement_hardware'))}</td>"
            f"<td>{_h(anchor.get('gradient_tensor_kind'))}</td>"
            f"<td>{_h(anchor.get('gradient_byte_domain') or 'unset')}</td>"
            f"<td>{_h(anchor.get('n_bytes'))}</td>"
            f"<td>{_h(anchor.get('evidence_grade') or 'unset')}</td>"
            f"<td>{_h(grain_inv_str)}</td>"
            f"<td>{cascade_flag}</td></tr>"
        )

    # Group plot cards: cascade comparison plots get prominence at top.
    plot_cards: list[str] = []
    cascade_cards: list[str] = []
    standard_cards: list[str] = []
    for entry in emitted_plots:
        plot_id = entry["plot_id"]
        png_rel = entry["png_relative"]
        sidecar_rel = entry["sidecar_relative"]
        if "cascade_smearing_comparison" in plot_id:
            cascade_cards.append(
                f"<div class=\"plot-card cascade-comparison\">"
                f"<h3>{_h(plot_id)}</h3>"
                f"<p class=\"cascade-note\"><b>Cascade-smearing comparison:</b> "
                f"side-by-side per-byte sensitivity heatmaps. RAW-BYTE (LEFT) is "
                f"entropy-cascade-smeared per Catalog #318 + codex op7 finding; "
                f"POST-DECOMPRESS (RIGHT) is the locality-correct reference. "
                f"Top-K Jaccard + cascade_smearing_factor in sister JSON.</p>"
                f"<a href=\"{_h(png_rel)}\"><img src=\"{_h(png_rel)}\" alt=\"{_h(plot_id)}\" /></a>"
                f"<p>Sister JSON: <a href=\"{_h(sidecar_rel)}\"><code>{_h(sidecar_rel)}</code></a></p>"
                f"</div>"
            )
        else:
            standard_cards.append(
                f"<div class=\"plot-card\">"
                f"<h3>{_h(plot_id)}</h3>"
                f"<a href=\"{_h(png_rel)}\"><img src=\"{_h(png_rel)}\" alt=\"{_h(plot_id)}\" /></a>"
                f"<p>Sister JSON: <a href=\"{_h(sidecar_rel)}\"><code>{_h(sidecar_rel)}</code></a></p>"
                f"</div>"
            )
    plot_cards = cascade_cards + standard_cards

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
.plot-card.cascade-comparison {{ background: #fffbeb; border: 2px solid #d97706; }}
.plot-card.cascade-comparison h3 {{ color: #b45309; }}
.cascade-note {{ background: #fef3c7; padding: 6px 10px; border-radius: 4px; font-size: 0.88em; }}
.banner {{ background: #fef3c7; border-left: 4px solid #d97706; padding: 8px 12px; margin: 12px 0; font-size: 0.9em; }}
.grain-banner {{ background: #e0e7ff; border-left: 4px solid #4338ca; padding: 8px 12px; margin: 12px 0; font-size: 0.9em; }}
code {{ background: #f3f4f6; padding: 1px 4px; border-radius: 3px; font-size: 0.85em; }}
ul {{ font-size: 0.9em; }}
</style>
</head>
<body>
<h1>master-gradient xray visualization</h1>
<p>Generated: <code>{_h(_utc_now_iso())}</code> UTC</p>
<p>Schema: <code>{_h(INDEX_HTML_SCHEMA_VERSION)}</code></p>
<p>Grain filter: <code>{_h(grain or "all (latest-by-utc; pre-grain behavior)")}</code></p>

<div class=\"banner\">
<b>Observability surface, NOT score authority.</b> Every sidecar JSON carries
<code>score_claim=false</code> + <code>promotion_eligible=false</code> +
<code>evidence_grade=[predicted]</code> per Catalog #305 / #323 / #287. The
underlying anchor's <code>measurement_axis</code> determines any score-axis
claim — see the table below.
</div>

<div class=\"grain-banner\">
<b>Grain awareness (slot 6 + slot 10, 2026-05-19):</b> per-byte sensitivity
is now ROUTED by grain. Raw-byte grains
(<code>scored_archive_bytes</code> / <code>zip_inner_member_payload</code>)
are entropy-cascade-smeared per Catalog #318 + codex op7 finding (one
raw-byte flip invalidates the entire downstream entropy stream).
Post-decompress grains
(<code>post_brotli_decompress_decoder_weight_bytes</code> +
<code>post_arithmetic_*</code> / <code>post_decompress_*</code>) are the
CORRECT locality basis. Archives with BOTH grains get a
<b>cascade_smearing_comparison</b> plot showing the divergence empirically.
</div>

<h2>Operator question driving the analysis</h2>
<p>Where is the score-relevant byte leverage concentrated in this archive,
how does it intersect with MPS-vs-CUDA drift, and how much cascade smearing
does the raw-byte gradient exhibit vs the post-decompress reference?
Inputs: the master-gradient anchor (per-archive sensitivity tensor),
optionally the granular MPS drift JSON, and (when available) BOTH the
raw-byte AND post-decompress anchors. Outputs: 5 canonical plots that
decompose leverage by byte/pair/quadrant + the 7th cascade-smearing
comparison plot.</p>

<h2>Archives processed ({_h(len(archive_shas))})</h2>
<table>
<tr><th>sha[:12]</th><th>axis</th><th>hardware</th><th>tensor_kind</th><th>grain</th><th>n_bytes</th><th>evidence_grade</th><th>grain inventory</th><th>compare?</th></tr>
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
    grain_filter: str | None = None,
) -> tuple[np.ndarray, dict] | None:
    """Resolve aggregate or per-pair gradient + anchor for one archive sha.

    When ``soft=True``, returns None on FileNotFoundError instead of raising.
    Used by ``--output-dir`` mode to keep emitting other plots when one
    underlying anchor type is unavailable.

    When ``grain_filter`` is set (one of GRAIN_FILTER_RAW_BYTE /
    GRAIN_FILTER_POST_DECOMPRESS), filters anchors by grain class via
    :func:`_load_aggregate_by_grain`. The default (None) preserves the
    pre-grain-aware behavior (latest-by-utc across all grains).
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
    if grain_filter in (GRAIN_FILTER_RAW_BYTE, GRAIN_FILTER_POST_DECOMPRESS):
        result = _load_aggregate_by_grain(archive_sha, grain_filter=grain_filter)
        if result is None:
            if soft:
                return None
            raise SystemExit(
                f"aggregate anchor for grain={grain_filter} missing for archive "
                f"{archive_sha[:16]}; consider extracting via "
                "tools/extract_master_gradient.py (raw_byte) OR slot 15/17 "
                "post-decompress extractors "
                "(tac.master_gradient_post_brotli_decompress / "
                "tac.master_gradient_post_decompress_multi_archive)"
            )
        return result
    try:
        return load_aggregate_gradient_from_anchor(archive_sha256=archive_sha)
    except FileNotFoundError as exc:
        if soft:
            return None
        raise SystemExit(
            f"aggregate anchor required but missing for archive {archive_sha[:16]} ({exc})"
        )


def _load_aggregate_by_grain(
    archive_sha: str, *, grain_filter: str
) -> tuple[np.ndarray, dict] | None:
    """Load an aggregate gradient for a specific grain class.

    Routes through ``tac.master_gradient.query_anchors_by_archive``
    filtering on ``gradient_byte_domain in POST_DECOMPRESS_GRAINS`` (for
    ``grain_filter="post_decompress"``) or ``gradient_byte_domain in
    RAW_BYTE_GRAINS`` (for ``grain_filter="raw_byte"``). Picks the most
    recent matching anchor by ``measurement_utc`` and loads its .npy
    array. Returns None when no matching anchor exists.

    Sister of ``tac.master_gradient_per_byte_consumer.
    load_per_byte_sensitivity_for_archive(prefer_grain=...)``.
    """
    from tac.master_gradient import query_anchors_by_archive
    from tac.master_gradient_per_byte_consumer import (
        POST_DECOMPRESS_GRAINS,
        RAW_BYTE_GRAINS,
    )

    if grain_filter == GRAIN_FILTER_POST_DECOMPRESS:
        allowed = POST_DECOMPRESS_GRAINS
    elif grain_filter == GRAIN_FILTER_RAW_BYTE:
        allowed = RAW_BYTE_GRAINS
    else:
        return None

    rows = query_anchors_by_archive(archive_sha)
    rows = [
        r
        for r in rows
        if r.get("gradient_tensor_kind", "aggregate_per_byte_v1")
        == "aggregate_per_byte_v1"
        and r.get("gradient_array_path")
        and str(
            r.get("gradient_byte_domain") or "scored_archive_bytes"
        ) in allowed
    ]
    if not rows:
        return None
    latest = max(rows, key=lambda r: str(r.get("measurement_utc", "")))
    array_path = Path(str(latest.get("gradient_array_path") or ""))
    if not array_path.is_absolute():
        array_path = Path.cwd() / array_path
    if not array_path.is_file():
        return None
    arr = np.load(array_path)
    return arr, dict(latest)


def _emit_plot(
    plot_name: str,
    archive_shas: list[str],
    output: Path,
    *,
    mps_drift_data: dict | None = None,
    grain: str | None = None,
) -> dict[str, object] | None:
    """Dispatch the plot to the correct emitter; resolve required anchors.

    Returns a per-plot summary dict suitable for sister JSON emission, or
    ``None`` when no sidecar is appropriate (e.g. cross-substrate
    placeholder).

    ``grain`` is one of GRAIN_FILTER_RAW_BYTE / GRAIN_FILTER_POST_DECOMPRESS /
    GRAIN_FILTER_ALL / None (latest-by-utc across all grains; pre-grain
    behavior). For grain_filter in {raw_byte, post_decompress}, the anchor
    resolver filters by ``gradient_byte_domain`` class. The
    ``cascade_smearing_comparison`` plot ignores ``grain`` and ALWAYS
    requires BOTH a raw-byte AND a post-decompress anchor.
    """
    # Map grain values to the anchor-resolver's expected vocabulary.
    if grain in (GRAIN_FILTER_RAW_BYTE, GRAIN_FILTER_POST_DECOMPRESS):
        resolver_grain = grain
    else:
        resolver_grain = None
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
        aggregate, anchor = _resolve_anchor_for_plot(
            sha, require_per_pair=False, grain_filter=resolver_grain
        )
        plot_per_byte_heatmap(aggregate, anchor, output)
        return {
            "anchor": anchor,
            "summary": _summary_stats_for_aggregate(aggregate),
            "extra": {"plot_top_k": 128, "grain_filter": grain or "any"},
        }
    elif plot_name == "cumulative_by_rank":
        sha = archive_shas[0]
        aggregate, anchor = _resolve_anchor_for_plot(
            sha, require_per_pair=False, grain_filter=resolver_grain
        )
        plot_cumulative_by_rank(aggregate, anchor, output)
        return {
            "anchor": anchor,
            "summary": _summary_stats_for_aggregate(aggregate),
            "extra": {"grain_filter": grain or "any"},
        }
    elif plot_name == "cross_substrate_correlation":
        substrate_anchors = []
        for sha in archive_shas:
            aggregate, anchor = _resolve_anchor_for_plot(
                sha, require_per_pair=False, grain_filter=resolver_grain
            )
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
                "grain_filter": grain or "any",
            },
        }
    elif plot_name == "wyner_ziv_flow":
        sha = archive_shas[0]
        aggregate, anchor = _resolve_anchor_for_plot(
            sha, require_per_pair=False, grain_filter=resolver_grain
        )
        plot_wyner_ziv_flow(aggregate, anchor, output)
        return {
            "anchor": anchor,
            "summary": _summary_stats_for_aggregate(aggregate),
            "extra": {
                "has_section_manifest": bool(
                    isinstance(anchor.get("archive_layout"), dict)
                    and anchor["archive_layout"].get("sections")
                ),
                "grain_filter": grain or "any",
            },
        }
    elif plot_name == "drift_vs_sensitivity_scatter":
        if mps_drift_data is None:
            raise SystemExit(
                f"plot {plot_name!r} requires --mps-drift-json"
            )
        sha = archive_shas[0]
        aggregate, anchor = _resolve_anchor_for_plot(
            sha, require_per_pair=False, grain_filter=resolver_grain
        )
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
                "grain_filter": grain or "any",
            },
        }
    elif plot_name == "cascade_smearing_comparison":
        sha = archive_shas[0]
        # Hard requirement: BOTH grains must exist for this plot.
        raw_result = _resolve_anchor_for_plot(
            sha,
            require_per_pair=False,
            soft=True,
            grain_filter=GRAIN_FILTER_RAW_BYTE,
        )
        post_result = _resolve_anchor_for_plot(
            sha,
            require_per_pair=False,
            soft=True,
            grain_filter=GRAIN_FILTER_POST_DECOMPRESS,
        )
        if raw_result is None or post_result is None:
            raise SystemExit(
                f"plot {plot_name!r} requires BOTH raw-byte and post-decompress "
                f"anchors for archive {sha[:16]}; "
                f"raw_byte_anchor={'present' if raw_result else 'MISSING'} "
                f"post_decompress_anchor={'present' if post_result else 'MISSING'}. "
                "Extract raw-byte via `tools/extract_master_gradient.py` and "
                "post-decompress via slot 15/17 helpers "
                "(tac.master_gradient_post_brotli_decompress / "
                "tac.master_gradient_post_decompress_multi_archive)."
            )
        raw_arr, raw_anchor = raw_result
        post_arr, post_anchor = post_result
        metrics = plot_cascade_smearing_comparison(
            raw_arr, raw_anchor, post_arr, post_anchor, output, top_k=128
        )
        # Sister JSON anchor = post_decompress (the CORRECT-locality reference)
        return {
            "anchor": post_anchor,
            "summary": {
                "raw_byte_anchor": {
                    "archive_sha256": raw_anchor.get("archive_sha256"),
                    "gradient_byte_domain": raw_anchor.get("gradient_byte_domain"),
                    "n_bytes": raw_anchor.get("n_bytes"),
                    "measurement_utc": raw_anchor.get("measurement_utc"),
                    "measurement_axis": raw_anchor.get("measurement_axis"),
                    "measurement_hardware": raw_anchor.get("measurement_hardware"),
                },
                "post_decompress_anchor": {
                    "archive_sha256": post_anchor.get("archive_sha256"),
                    "gradient_byte_domain": post_anchor.get("gradient_byte_domain"),
                    "n_bytes": post_anchor.get("n_bytes"),
                    "measurement_utc": post_anchor.get("measurement_utc"),
                    "measurement_axis": post_anchor.get("measurement_axis"),
                    "measurement_hardware": post_anchor.get("measurement_hardware"),
                },
                "cascade_smearing_metrics": metrics,
            },
            "extra": {"plot_top_k": 128, "grain_filter": "compare_both"},
        }
    elif plot_name == "consumer_verdict_matrix":
        # Slot EE 2026-05-19 plot type 3 per task #797 operator spec.
        # Per Catalog #335 + #341: invoke every auto-discovered consumer
        # for ONE candidate and render the per-consumer × per-candidate
        # verdict matrix. The candidate is synthesized from the first
        # archive_sha if no full candidate dict is available at the CLI
        # surface (richer candidate routing is operator-facing via direct
        # call to `plot_consumer_verdict_matrix`).
        sha = archive_shas[0]
        # Try to fetch a real per-pair anchor first; fall back to aggregate
        anchor: dict | None = None
        soft_per_pair = _resolve_anchor_for_plot(sha, require_per_pair=True, soft=True)
        if soft_per_pair is not None:
            anchor = soft_per_pair[1]
        else:
            soft_agg = _resolve_anchor_for_plot(
                sha, require_per_pair=False, soft=True, grain_filter=resolver_grain
            )
            if soft_agg is not None:
                anchor = soft_agg[1]
        candidate = {
            "candidate_id": f"matrix_candidate_{sha[:8]}",
            "archive_sha256": sha,
            "predicted_delta": -0.005,
            "family": "ad_hoc_xray_visualization",
            "literature_anchor": (
                anchor.get("substrate_id") if isinstance(anchor, dict) else None
            ),
            "pareto_scope": "rate_seg_pose",
        }
        metrics = plot_consumer_verdict_matrix(candidate, output)
        return {
            "anchor": anchor or {"archive_sha256": sha},
            "summary": metrics,
            "extra": {
                "candidate_id": candidate["candidate_id"],
                "grain_filter": grain or "any",
            },
        }
    elif plot_name == "provenance_audit_timeline":
        # Slot EE 2026-05-19 plot type 5 per task #797 operator spec.
        # Operates over the FULL canonical ledger (not per-archive); the
        # CLI's `--archive-sha` filter is honored to scope by sha.
        from tac.master_gradient import load_anchors_lenient  # noqa: E402

        anchors = load_anchors_lenient()
        if archive_shas:
            anchors = [
                a for a in anchors
                if any(
                    str(a.get("archive_sha256", "")).startswith(sha)
                    for sha in archive_shas
                )
            ]
        metrics = plot_provenance_audit_timeline(anchors, output, since_utc=None)
        # Sister-JSON anchor synthesizes the audit-level metadata
        fake_anchor = {
            "archive_sha256": archive_shas[0] if archive_shas else None,
            "measurement_axis": "[predicted]",
            "measurement_hardware": "visualization_derived_view",
            "measurement_utc": _utc_now_iso(),
            "evidence_grade": "[predicted; provenance audit derived view]",
            "n_bytes": metrics["n_anchors"],
        }
        return {
            "anchor": fake_anchor,
            "summary": metrics,
            "extra": {"plot_type": "provenance_audit_timeline"},
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
    grain: str | None = None,
) -> tuple[list[dict[str, str]], list[dict]]:
    """Emit all canonical plots + sister JSONs + index.html.

    Per Catalog #305 observability + Catalog #323 Provenance: every plot
    gets a sister JSON; every sister JSON carries the canonical Provenance
    sub-object built via build_provenance_for_predicted (visualization is
    a derived view, NOT a primary measurement).

    ``grain`` (slot 6 + slot 10 grain awareness landing 2026-05-19):

    * ``None`` or ``"all"`` — emit canonical 5 plots (latest-by-utc
      across grains; pre-grain behavior) + optional drift scatter +
      optional cascade comparison when BOTH grains exist per archive.
    * ``"raw_byte"`` / ``"post_decompress"`` — emit canonical 5 plots
      filtered to that grain only. Cascade comparison is skipped.
    * ``"compare_both"`` — emit canonical 5 plots in latest-by-utc mode
      PLUS one cascade_smearing_comparison plot per archive that has
      BOTH raw-byte AND post-decompress anchors.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    plots_to_emit = list(CANONICAL_OUTPUT_DIR_PLOTS)
    if mps_drift_data is not None:
        plots_to_emit.append("drift_vs_sensitivity_scatter")

    # Cascade-smearing comparison: one entry per archive that has BOTH
    # raw-byte AND post-decompress anchors. Entry filename suffixed with
    # short-sha so multiple archives in same output dir get distinct
    # artifacts.
    cascade_comparison_eligible: list[str] = []
    if grain in (GRAIN_FILTER_COMPARE_BOTH, GRAIN_FILTER_ALL, None):
        from tac.master_gradient_per_byte_consumer import (
            available_grains_for_archive,
        )

        for sha in archive_shas:
            inv = available_grains_for_archive(sha)
            if inv["raw_byte"] and inv["post_decompress"]:
                cascade_comparison_eligible.append(sha)

    emitted: list[dict[str, str]] = []
    anchors_seen: list[dict] = []
    archive_to_anchor: dict[str, dict] = {}

    # Pass 1: canonical 5 plots (+ optional drift) — grain filter honored
    # by the resolver. For raw_byte / post_decompress filters, only
    # anchors of that grain are picked.
    if grain in (GRAIN_FILTER_RAW_BYTE, GRAIN_FILTER_POST_DECOMPRESS):
        grain_for_resolver: str | None = grain
    else:
        grain_for_resolver = None

    for plot_name in plots_to_emit:
        png_path = output_dir / f"{plot_name}.png"
        sidecar_path = output_dir / f"{plot_name}.json"
        try:
            result = _emit_plot(
                plot_name,
                archive_shas,
                png_path,
                mps_drift_data=mps_drift_data,
                grain=grain_for_resolver,
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

    # Pass 2: cascade-smearing comparison (one per eligible archive).
    for sha in cascade_comparison_eligible:
        short = sha[:12]
        plot_id = f"cascade_smearing_comparison_{short}"
        png_path = output_dir / f"{plot_id}.png"
        sidecar_path = output_dir / f"{plot_id}.json"
        try:
            result = _emit_plot(
                "cascade_smearing_comparison",
                [sha],
                png_path,
                mps_drift_data=None,
                grain=GRAIN_FILTER_COMPARE_BOTH,
            )
        except SystemExit as exc:
            print(f"  SKIP {plot_id}: {exc}", file=sys.stderr)
            continue
        if result is not None:
            _emit_sidecar_json(
                sidecar_path,
                plot_id=plot_id,
                anchor=result["anchor"],
                summary=result["summary"],
                extra=result["extra"],
            )
            emitted.append(
                {
                    "plot_id": plot_id,
                    "png_relative": png_path.name,
                    "sidecar_relative": sidecar_path.name,
                }
            )
            print(f"  wrote {png_path}")
            print(f"  wrote {sidecar_path}")

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
        grain=grain,
        cascade_comparison_eligible=cascade_comparison_eligible,
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
        "--grain",
        choices=list(GRAIN_FILTER_CHOICES),
        default=None,
        help=(
            "Per-byte grain filter (slot 6 + slot 10 grain awareness, "
            "2026-05-19). Default: None (latest-by-utc across grains; "
            "pre-grain behavior). raw_byte / post_decompress: filter to "
            "ONE grain class only (per Catalog #318 + codex op7 finding: "
            "post-decompress is the CORRECT locality; raw-byte is "
            "entropy-cascade-smeared). compare_both: emit canonical plots "
            "in latest-by-utc mode PLUS a side-by-side "
            "cascade_smearing_comparison plot for every archive that has "
            "BOTH raw-byte AND post-decompress anchors. all: emit "
            "canonical plots in latest-by-utc mode (alias for default)."
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
            grain=args.grain,
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
                    grain=args.grain,
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
                    grain=args.grain,
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
            plot_name,
            args.archive_sha,
            output_path,
            mps_drift_data=mps_drift_data,
            grain=args.grain,
        )
        print(f"wrote {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
