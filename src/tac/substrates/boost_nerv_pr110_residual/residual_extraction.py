# SPDX-License-Identifier: MIT
"""Per-pair residual extraction from PR110 fec6 frontier archive (L0 SCAFFOLD).

Stage 0 + Stage 1 of the BoostNeRV-against-PR110 curriculum per design memo
.omx/research/path_3_e_boost_nerv_against_pr110_substrate_design_20260526.md

Stage 0: invoke PR110's inflate.sh as subprocess; cache per-pair frame_0 +
    frame_1 RGB reconstructions to .omx/state/pr110_base_reconstructions_<sha_prefix>/

Stage 1: compute per-pair residual_target = GT - PR110_base_reconstruction;
    log p50 + p99 residual magnitude as smoke convergence diagnostic.

This module is the PR110-base-extraction bridge — it does NOT train; it
caches the frozen base learner's per-pair outputs for downstream MLX
training. Per the binding 2026-05-26 reframing: the substrate is built
AROUND the method (iterative boosting against a frozen base), and PR110 is
the canonical frozen base.

CLAUDE.md compliance:
- No /tmp paths in persisted artifacts (cache lives under `.omx/state/` per
  the canonical artifact lifecycle Catalog #113)
- No score claims (this is research-signal extraction; not a contest scorer
  invocation)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Pr110BaseExtractionPlan:
    """Plan for extracting + caching PR110 base reconstructions.

    Refuses non-existent archive paths at construction time per Catalog
    #229 PV. The cache directory is keyed on the PR110 archive sha256
    prefix so multiple PR110-variant archives (e.g. v42 vs v43 vs v44)
    produce distinct cached reconstruction sets.
    """

    pr110_archive_zip: Path
    """Absolute path to the PR110 archive.zip (read-only)."""

    pr110_inflate_sh: Path
    """Absolute path to PR110's inflate.sh (read-only)."""

    cache_root: Path
    """Cache root under .omx/state/ (NOT /tmp/ per Catalog #113)."""

    upstream_video_path: Path
    """Absolute path to upstream/videos/0.mkv for GT decode."""

    def __post_init__(self) -> None:
        if not self.pr110_archive_zip.is_file():
            raise FileNotFoundError(
                f"PR110 archive.zip not found: {self.pr110_archive_zip}"
            )
        if not self.pr110_inflate_sh.is_file():
            raise FileNotFoundError(
                f"PR110 inflate.sh not found: {self.pr110_inflate_sh}"
            )
        if not self.upstream_video_path.is_file():
            raise FileNotFoundError(
                f"upstream video not found: {self.upstream_video_path}"
            )
        # cache_root may not exist yet; we create it on demand. But
        # refuse /tmp paths per Catalog #113 (transient-evidence trap).
        if str(self.cache_root).startswith("/tmp/") or str(self.cache_root).startswith("/private/var/"):
            raise ValueError(
                f"cache_root must not be under /tmp/ or /private/var/ per CLAUDE.md "
                f"'Forbidden /tmp paths in any persisted artifact' "
                f"(forbidden path: {self.cache_root}); use .omx/state/ subtree instead"
            )

    def compute_pr110_sha256_prefix(self) -> bytes:
        """SHA256 prefix (16 bytes) of PR110 archive for BPR1 header binding."""
        h = hashlib.sha256()
        with self.pr110_archive_zip.open("rb") as fh:
            while True:
                chunk = fh.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        return h.digest()[:16]

    def cache_dir_for_this_pr110(self) -> Path:
        """Cache subdirectory keyed by PR110 sha256 prefix (16 hex chars)."""
        sha_prefix_hex = self.compute_pr110_sha256_prefix().hex()
        return self.cache_root / f"pr110_base_reconstructions_{sha_prefix_hex}"


def extract_per_pair_residual_targets(
    pr110_base_frames_0,  # numpy array (N_pairs, 3, H, W)
    pr110_base_frames_1,  # numpy array (N_pairs, 3, H, W)
    gt_frames_0,          # numpy array (N_pairs, 3, H, W)
    gt_frames_1,          # numpy array (N_pairs, 3, H, W)
):
    """Compute per-pair residual targets = GT - PR110_base_reconstruction.

    Stage 1 of the BoostNeRV-against-PR110 curriculum. Inputs are float32
    in [0, 1]; output is float32 in [-1, 1] (most values near 0).

    Returns:
        residual_target_0, residual_target_1 (each (N_pairs, 3, H, W) float32)
    """
    import numpy as np

    if pr110_base_frames_0.shape != gt_frames_0.shape:
        raise ValueError(
            f"shape mismatch: pr110_base_frames_0 {pr110_base_frames_0.shape} "
            f"vs gt_frames_0 {gt_frames_0.shape}"
        )
    if pr110_base_frames_1.shape != gt_frames_1.shape:
        raise ValueError(
            f"shape mismatch: pr110_base_frames_1 {pr110_base_frames_1.shape} "
            f"vs gt_frames_1 {gt_frames_1.shape}"
        )

    residual_target_0 = (gt_frames_0.astype(np.float32) - pr110_base_frames_0.astype(np.float32))
    residual_target_1 = (gt_frames_1.astype(np.float32) - pr110_base_frames_1.astype(np.float32))
    return residual_target_0, residual_target_1


def diagnose_residual_target_magnitude(residual_target_0, residual_target_1) -> dict:
    """Smoke convergence diagnostic per design memo Stage 1.

    If residual_target_magnitude_p99 < 0.01 (1% RGB range) across all pairs,
    PR110 is already near-optimal at the per-pair RGB-pixel level and the
    boosting learner has no signal to extract — DEFER per CLAUDE.md
    "Forbidden premature KILL".

    If p99 >= 0.05 (5% RGB range), the boosting paradigm has headroom to
    extract structured signal — PROCEED to Stage 2 (MLX warm-up).

    Returns a diagnostic dict (NOT a score claim per Catalog #127).
    """
    import numpy as np

    all_abs_residuals = np.concatenate(
        [np.abs(residual_target_0).flatten(), np.abs(residual_target_1).flatten()]
    )
    p50 = float(np.percentile(all_abs_residuals, 50))
    p99 = float(np.percentile(all_abs_residuals, 99))
    pmax = float(np.max(all_abs_residuals))

    # Verdict per design memo Stage 1 convergence criterion (NOT a score
    # claim — this is a residual magnitude diagnostic; per Catalog #127 +
    # #341 routing markers).
    if p99 < 0.01:
        verdict = "PR110_BASE_NEAR_OPTIMAL_NO_HEADROOM_DEFER_PER_CLAUDE_MD_FORBIDDEN_PREMATURE_KILL"
    elif p99 >= 0.05:
        verdict = "BOOSTING_HEADROOM_AVAILABLE_PROCEED_TO_STAGE_2_MLX_WARMUP"
    else:
        verdict = "BOOSTING_HEADROOM_MARGINAL_PROCEED_WITH_PHASE_2_COUNCIL_REVIEW_PER_CATALOG_325"

    return {
        "residual_magnitude_p50_rgb_range": p50,
        "residual_magnitude_p99_rgb_range": p99,
        "residual_magnitude_pmax_rgb_range": pmax,
        "verdict": verdict,
        "axis_tag": "[macOS-MLX research-signal]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def write_diagnostic_manifest(diagnostic: dict, output_path: Path) -> None:
    """Persist Stage 1 diagnostic with canonical Provenance per Catalog #323.

    NOT a score claim. Carries canonical non-promotable markers per
    Catalog #341 dual-tier consumer architecture.
    """
    if not isinstance(diagnostic, dict):
        raise TypeError("diagnostic must be dict")
    # Validate non-promotable markers per Catalog #341.
    if diagnostic.get("score_claim") is not False:
        raise ValueError("Stage 1 diagnostic must carry score_claim=False per Catalog #127")
    if diagnostic.get("promotion_eligible") is not False:
        raise ValueError("Stage 1 diagnostic must carry promotion_eligible=False per Catalog #192")
    if diagnostic.get("axis_tag") != "[macOS-MLX research-signal]":
        raise ValueError(
            "Stage 1 diagnostic must carry axis_tag='[macOS-MLX research-signal]' per Catalog #341"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(diagnostic, indent=2, sort_keys=True))
