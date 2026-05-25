#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe 9c: per-level wavelet-basis selection disambiguator.

PURPOSE
=======
Per Probe 9 Tier-2 dispatch symposium 2026-05-25 Mallat binding revision #3
(BLOCKS DISPATCH): the optimal wavelet basis is CONTENT-DEPENDENT (Mallat 1989
*A theory for multiresolution signal decomposition* + Daubechies 1988 orthogonal
compactly-supported wavelets). Dashcam video at 384x512 carries SPECIFIC
spatial-frequency content (vehicles / road surface / vanishing-point geometry)
that MAY have OPTIMAL wavelet basis different from generic db8. Per the Mallat
seat verbatim binding revision: per-level wavelet-basis selection table required
with empirical anchor BEFORE archive grammar finalization.

Probe 9b (commit 686b07f31, N=100, db8) established:
- segment_textured_avg_weight_mean = 0.3915 (95% CI [0.3805, 0.4030]; bootstrap 5000-iter)
- segment_textured_avg_weight_min = 0.0932 (sharpest inversion)
- valid_segment_count = 537; below_threshold_fraction = 0.8007 (80.1% < 0.5)

Probe 9c extends Probe 9b by holding ALL parameters identical EXCEPT wavelet
basis: db4 / db8 / db16 / bior4.4. Same N=100 pairs / same source video / same
3-level decomposition / same per-instance segmentation / same UNIWARD weighting /
same MIN_SEGMENT_PIXELS / same SIGMA_FRIDRICH. Pure basis ablation.

CANONICAL WAVELET BASIS FAMILY
================================
- db4 (Daubechies-4, 4-tap orthogonal): 2 vanishing moments; sparser basis;
  canonical sparse-coefficient basis for texture; smaller support = sharper
  localization at edges.
- db8 (Daubechies-8, 8-tap orthogonal): 4 vanishing moments; Probe 9 / 9b
  anchor; canonical compromise sparsity vs locality.
- db16 (Daubechies-16, 16-tap orthogonal): 8 vanishing moments; larger support
  = smoother basis; canonical for smooth-texture basis.
- bior4.4 (biorthogonal 4.4, 9/7 filter pair): symmetric-phase-no-distortion;
  canonical JPEG2000 lossless transform.

FALSIFIABLE PREDICATE (Carmack MVP-first step 2)
===================================================
- NULL hypothesis: db8 is canonical-optimal; sister bases produce
  indistinguishable mean within 2sigma tolerance.
- ALTERNATIVE 1: db4 dominates (sparser basis -> stronger UNIWARD signal at
  texture boundaries).
- ALTERNATIVE 2: db16 dominates (smoother basis -> reduced noise floor).
- ALTERNATIVE 3: bior4.4 dominates (symmetric-phase preserves true edge response).
- REJECT NULL at any sister basis z-score > 2sigma from db8 baseline.

VERDICT TAXONOMY
==================
- db8_CONFIRMED_OPTIMAL : NULL preserved; no sister basis crosses 2sigma threshold.
- SISTER_BASIS_DOMINATES_<basis_name> : sister z-score > 2sigma AND mean is
  lower (lower = sharper inversion = better signal per UNIWARD inverse-cost form).
  Per Catalog #307 IMPLEMENTATION-LEVEL falsification: paradigm INTACT; Probe 9
  substrate recipe canonical update to use the dominant basis.
- INSUFFICIENT_DISCRIMINATION : variance dominates across bases (all bases'
  bootstrap CIs overlap by > 50%); queue Catalog #308 alternative reducer
  (e.g., 2-level vs 3-level vs 4-level decomposition; OR per-region vs
  per-instance UNIWARD weighting; OR alternative steganographic distortion
  measure HILL or S-UNIWARD).

DISCIPLINE
===========
- $0 macOS-CPU advisory ONLY (Catalog #1 + #192 + CLAUDE.md "MPS auth eval is NOISE")
- canonical Provenance per Catalog #287 + #323 + #341:
    evidence_grade = "macOS-CPU-advisory"
    axis_tag = "[macOS-CPU advisory]"
    promotable = False
    score_claim = False
    ready_for_exact_eval_dispatch = False
- canonical PyTorch CPU forward (NOT MPS per Catalog #1 silent-fallback ban)
- canonical PR 9b helper REUSE for apples-to-apples basis comparison
  (per Mallat seat: same downstream pipeline; ONLY wavelet basis varies)

CANONICAL EQUATION
===================
Sister candidate: ``uniward_per_instance_multi_scale_wavelet_combined_v1``
(FORMALIZATION_PENDING per Catalog #344). N=100 anchor STRENGTHENS evidence
count if db8_CONFIRMED_OPTIMAL; queues NEW sister candidate
``uniward_per_instance_multi_scale_wavelet_basis_optimal_<dominant_basis>_v1``
if SISTER_BASIS_DOMINATES verdict.

USAGE
=====
    .venv/bin/python tools/probe_9c_per_level_wavelet_basis_selection_disambiguator.py \\
        --wavelet-basis-family db4,db8,db16,bior4.4 \\
        --n-pairs 100 \\
        --source-video upstream/videos/0.mkv \\
        --output-json .omx/research/tier_1_distortion_axis_probes_20260521/probe_9c_per_level_wavelet_basis_<utc>.json \\
        --seed 42

CROSS-REFERENCES
=================
- Probe 9b landing (db8 N=100 baseline): ``.omx/research/probe_9b_100_pair_disambiguator_landed_20260525.md``
- Probe 9b source tool: ``tools/probe_9b_100_pair_uniward_per_instance_multi_scale_wavelet_combined_disambiguator.py``
- Probe 9b verdict JSON: ``.omx/research/tier_1_distortion_axis_probes_20260521/probe_9b_100_pair_20260525T164153Z.json``
- Symposium (Mallat revision #3): ``.omx/research/per_substrate_symposium_uniward_per_instance_multi_scale_wavelet_segnet_20260525.md`` L101-122 + L210-217
- Discipline: CLAUDE.md "Carmack MVP-first" + "Forbidden premature KILL" + Catalog #1/#192/#287/#290/#307/#308/#313/#323/#341/#344
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import socket
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "upstream"))
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pywt  # noqa: E402
import scipy.ndimage  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from safetensors.torch import load_file  # noqa: E402

from upstream.modules import SegNet  # noqa: E402

# ---------------------------------------------------------------------------
# Canonical constants (identical to Probe 9b for apples-to-apples)
# ---------------------------------------------------------------------------

WAVELET_LEVELS = 3
SIGMA_FRIDRICH = 2.0**-6
MIN_SEGMENT_PIXELS = 200
MIN_TEXTURED_PIXELS = 50
SEGNET_EVAL_H = 384
SEGNET_EVAL_W = 512

# Canonical wavelet basis family per Mallat revision #3
CANONICAL_WAVELET_BASIS_FAMILY = ("db4", "db8", "db16", "bior4.4")
BASELINE_BASIS = "db8"  # Probe 9 / 9b anchor

# Probe 9b N=100 db8 BASELINE empirical anchor (commit 686b07f31; UNIQUE
# WORKING-TREE source of truth for cross-comparison z-score)
PROBE_9B_N100_DB8_BASELINE = {
    "segment_textured_avg_weight_mean": 0.39151591753493475,
    "segment_textured_avg_weight_min": 0.09315050393342972,
    "segment_textured_avg_weight_max": 0.7116744518280029,
    "segment_textured_avg_weight_median": 0.4198218882083893,
    "segment_textured_avg_weight_stdev": 0.13180527138677492,
    "segment_textured_below_threshold_count": 430,
    "segment_textured_below_threshold_fraction": 0.8007448789571695,
    "valid_segment_count": 537,
    "bootstrap_ci_mean_lower": 0.38051205494196555,
    "bootstrap_ci_mean_upper": 0.40295172989146444,
    "bootstrap_ci_halfwidth": 0.011219837474749444,
    "wavelet_name": "db8",
    "wavelet_levels": 3,
    "evidence_path": ".omx/research/tier_1_distortion_axis_probes_20260521/probe_9b_100_pair_20260525T164153Z.json",
}

# Falsification thresholds (Carmack MVP-first step 2)
SISTER_BASIS_DOMINATES_Z_THRESHOLD = 2.0  # z > 2sigma rejects NULL
INSUFFICIENT_CI_HALFWIDTH = 0.05  # bootstrap CI half-width above this -> insufficient
CI_OVERLAP_FRACTION_INSUFFICIENT = 0.5  # > 50% CI overlap across all bases

# Bootstrap CI
BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_CI_PERCENTILE_LOWER = 2.5
BOOTSTRAP_CI_PERCENTILE_UPPER = 97.5

# Canonical Provenance (Catalog #287 + #323 + #341)
PROVENANCE_AXIS_TAG = "[macOS-CPU advisory]"
PROVENANCE_HARDWARE_SUBSTRATE = "darwin_arm64_m5_max_macos_cpu_advisory"
PROVENANCE_EVIDENCE_GRADE = "macOS-CPU-advisory"
SCHEMA_VERSION = "probe_9c_per_level_wavelet_basis_selection_disambiguator_v1"


# ---------------------------------------------------------------------------
# Helpers (CANONICAL Probe 9b functions reused verbatim for apples-to-apples)
# ---------------------------------------------------------------------------


def _decode_first_n_frames(video_path: Path, n: int) -> tuple[torch.Tensor, str]:
    """Decode first N frames from contest video via pyav; return tensor + sha256."""
    import av  # type: ignore

    container = av.open(str(video_path))
    frames: list[np.ndarray] = []
    for i, frame in enumerate(container.decode(video=0)):
        if i >= n:
            break
        arr = frame.to_ndarray(format="rgb24")
        frames.append(arr)
    container.close()

    if len(frames) < n:
        raise RuntimeError(
            f"Video has only {len(frames)} frames; requested {n}; "
            f"need >={n} for {n - 1}-pair sample."
        )

    stacked = np.stack(frames, axis=0)
    tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).float() / 255.0
    sha = hashlib.sha256(video_path.read_bytes()).hexdigest()
    return tensor, sha


def _luma(frames: torch.Tensor) -> torch.Tensor:
    """ITU-R BT.601 luma; matches CCC probe 3 baseline."""
    return 0.2989 * frames[:, 0] + 0.5870 * frames[:, 1] + 0.1140 * frames[:, 2]


def _multi_scale_wavelet_detail_magnitudes(
    luma: torch.Tensor,
    wavelet_name: str,
    levels: int = WAVELET_LEVELS,
) -> torch.Tensor:
    """Canonical Holub-Fridrich 2014 multi-level wavelet detail-subband sum.

    Per-basis ablation: ONLY wavelet_name varies; all other parameters held
    identical to Probe 9b for apples-to-apples comparison.

    Per Mallat 1989 + Daubechies 1988 + Holub-Fridrich-Denemark 2014.
    """
    luma_np = luma.cpu().numpy()
    n, h, w = luma_np.shape
    detail_sum = np.zeros((n, h, w), dtype=np.float32)

    for i in range(n):
        coeffs_multi = pywt.wavedec2(
            luma_np[i], wavelet_name, mode="symmetric", level=levels
        )
        for level_coeffs in coeffs_multi[1:]:  # skip approximation
            cH, cV, cD = level_coeffs
            detail_lo = np.abs(cH) + np.abs(cV) + np.abs(cD)
            detail_t = torch.from_numpy(detail_lo).unsqueeze(0).unsqueeze(0)
            upsampled = F.interpolate(detail_t, size=(h, w), mode="nearest")
            detail_sum[i] += upsampled.squeeze(0).squeeze(0).numpy()

    return torch.from_numpy(detail_sum)


# ---------------------------------------------------------------------------
# Bootstrap CI (canonical Probe 9b helper reused verbatim)
# ---------------------------------------------------------------------------


def _bootstrap_mean_ci(
    samples: np.ndarray,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    ci_lower_pct: float = BOOTSTRAP_CI_PERCENTILE_LOWER,
    ci_upper_pct: float = BOOTSTRAP_CI_PERCENTILE_UPPER,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Bootstrap-resample the mean and emit 95% CI."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(samples)
    if n == 0:
        return {
            "mean": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "ci_halfwidth": float("nan"),
            "iterations": n_iterations,
            "stdev_bootstrap_means": float("nan"),
        }
    boot_means = np.empty(n_iterations, dtype=np.float64)
    samples_arr = np.asarray(samples, dtype=np.float64)
    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = samples_arr[idx].mean()
    ci_lower = float(np.percentile(boot_means, ci_lower_pct))
    ci_upper = float(np.percentile(boot_means, ci_upper_pct))
    mean_val = float(samples_arr.mean())
    return {
        "mean": mean_val,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_halfwidth": (ci_upper - ci_lower) / 2.0,
        "iterations": n_iterations,
        "stdev_bootstrap_means": float(boot_means.std(ddof=1)),
    }


# ---------------------------------------------------------------------------
# Per-basis probe runner (single basis; called 4x via outer loop)
# ---------------------------------------------------------------------------


def _run_basis_probe(
    *,
    wavelet_name: str,
    pairs: torch.Tensor,
    seg_mask: torch.Tensor,
    frames: torch.Tensor,
    n_pairs: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Run per-instance + multi-scale UNIWARD pipeline with a specific wavelet basis.

    All inputs (pairs / seg_mask / frames) are PRECOMPUTED once outside this
    function for cross-basis apples-to-apples: ONLY the wavelet decomposition
    branch varies per basis.

    Returns canonical per-basis summary dict.
    """
    t_start = time.time()

    # === Eval frames per upstream/modules.py x[:, -1] convention ===
    eval_frame_indices = list(range(1, n_pairs + 1))
    eval_frames = frames[eval_frame_indices]  # (n_pairs, 3, H, W)
    luma_eval = _luma(eval_frames)  # (n_pairs, H, W)

    # === Per-basis MULTI-LEVEL wavelet detail-subband sum ===
    detail_sum_orig_res = _multi_scale_wavelet_detail_magnitudes(
        luma_eval, wavelet_name=wavelet_name, levels=WAVELET_LEVELS
    )  # (n_pairs, H, W)

    # Resize to SegNet's eval resolution
    detail_sum = F.interpolate(
        detail_sum_orig_res.unsqueeze(1),
        size=(SEGNET_EVAL_H, SEGNET_EVAL_W),
        mode="bilinear",
        align_corners=False,
    ).squeeze(1)  # (n_pairs, 384, 512)

    # UNIWARD weighting (per Fridrich 2014 canonical inverse cost)
    uniward_weights = 1.0 / (detail_sum + SIGMA_FRIDRICH)
    uniward_weights_norm = uniward_weights / uniward_weights.mean()

    # === Per-instance + multi-scale combined decomposition ===
    seg_mask_np = seg_mask.cpu().numpy()
    detail_sum_np = detail_sum.cpu().numpy()
    uniward_weights_norm_np = uniward_weights_norm.cpu().numpy()

    per_segment_textured_weights: list[float] = []
    per_class_segment_count: dict[int, int] = dict.fromkeys(range(5), 0)

    structure_4conn = np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8
    )

    for p in range(n_pairs):
        for c in range(5):
            class_mask_p = seg_mask_np[p] == c
            labeled, num_segments = scipy.ndimage.label(
                class_mask_p, structure=structure_4conn
            )
            if num_segments == 0:
                continue

            for instance_id in range(1, num_segments + 1):
                instance_mask = labeled == instance_id
                instance_pixel_count = int(instance_mask.sum())
                if instance_pixel_count < MIN_SEGMENT_PIXELS:
                    continue
                per_class_segment_count[c] += 1

                instance_details = detail_sum_np[p][instance_mask]
                instance_detail_q75 = float(np.quantile(instance_details, 0.75))
                instance_textured_mask = instance_mask & (
                    detail_sum_np[p] > instance_detail_q75
                )
                instance_textured_count = int(instance_textured_mask.sum())

                if instance_textured_count < MIN_TEXTURED_PIXELS:
                    continue

                instance_textured_avg_weight = float(
                    uniward_weights_norm_np[p][instance_textured_mask].mean()
                )
                per_segment_textured_weights.append(instance_textured_avg_weight)

    valid_segment_count = len(per_segment_textured_weights)

    # === Summary statistics ===
    if valid_segment_count > 0:
        weights_arr = np.asarray(per_segment_textured_weights, dtype=np.float64)
        basis_mean = float(weights_arr.mean())
        basis_min = float(weights_arr.min())
        basis_max = float(weights_arr.max())
        basis_median = float(np.median(weights_arr))
        basis_stdev = float(weights_arr.std(ddof=1)) if valid_segment_count > 1 else 0.0
        basis_below_threshold_count = int((weights_arr < 0.5).sum())
        basis_below_threshold_fraction = float(basis_below_threshold_count) / valid_segment_count
        hist_bins = np.linspace(0.0, 1.0, 11)
        hist_counts, _ = np.histogram(weights_arr, bins=hist_bins)
        boot_ci_mean = _bootstrap_mean_ci(weights_arr, rng=rng)
    else:
        basis_mean = basis_min = basis_max = basis_median = basis_stdev = float("nan")
        basis_below_threshold_count = 0
        basis_below_threshold_fraction = float("nan")
        hist_counts = np.zeros(10, dtype=np.int64)
        hist_bins = np.linspace(0.0, 1.0, 11)
        boot_ci_mean = {
            "mean": float("nan"),
            "ci_lower": float("nan"),
            "ci_upper": float("nan"),
            "ci_halfwidth": float("nan"),
            "iterations": BOOTSTRAP_ITERATIONS,
            "stdev_bootstrap_means": float("nan"),
        }

    elapsed_seconds = time.time() - t_start

    return {
        "wavelet_basis": wavelet_name,
        "wavelet_levels": WAVELET_LEVELS,
        "valid_segment_count": valid_segment_count,
        "per_class_segment_count": per_class_segment_count,
        "segment_textured_avg_weight_mean": basis_mean,
        "segment_textured_avg_weight_min": basis_min,
        "segment_textured_avg_weight_max": basis_max,
        "segment_textured_avg_weight_median": basis_median,
        "segment_textured_avg_weight_stdev": basis_stdev,
        "segment_textured_below_threshold_count": basis_below_threshold_count,
        "segment_textured_below_threshold_fraction": basis_below_threshold_fraction,
        "bootstrap_ci_mean": boot_ci_mean,
        "distribution_histogram": {
            "bin_edges": [float(x) for x in hist_bins.tolist()],
            "bin_counts": [int(c) for c in hist_counts.tolist()],
        },
        "elapsed_seconds": elapsed_seconds,
    }


# ---------------------------------------------------------------------------
# Cross-basis aggregation + verdict
# ---------------------------------------------------------------------------


def _compute_pairwise_ci_overlap_fraction(
    ci_a: tuple[float, float], ci_b: tuple[float, float]
) -> float:
    """Fraction of overlap between two confidence intervals (0.0 = disjoint;
    1.0 = identical). Used for CI_OVERLAP_FRACTION_INSUFFICIENT threshold."""
    lo_a, hi_a = ci_a
    lo_b, hi_b = ci_b
    width_a = hi_a - lo_a
    width_b = hi_b - lo_b
    if width_a <= 0 or width_b <= 0:
        return float("nan")
    overlap_lo = max(lo_a, lo_b)
    overlap_hi = min(hi_a, hi_b)
    overlap = max(0.0, overlap_hi - overlap_lo)
    min_width = min(width_a, width_b)
    return float(overlap / min_width)


def _classify_verdict(
    per_basis_results: dict[str, dict[str, Any]],
    baseline_basis: str = BASELINE_BASIS,
) -> dict[str, Any]:
    """Apply falsifiable predicate per Carmack MVP-first step 2.

    Returns (verdict, dominant_basis, per_basis_z_score_vs_db8, operator_signal).

    Decision cascade:
    1. INSUFFICIENT_DISCRIMINATION: all 4 basis-pair CI overlaps > 50% threshold
       (variance dominates; no basis discriminates).
    2. SISTER_BASIS_DOMINATES_<basis>: ANY sister basis (not baseline) has
       |z| > 2sigma vs baseline AND mean is LOWER than baseline (lower = sharper
       inversion = better signal per UNIWARD inverse-cost form). Dominant basis
       is the one with the LOWEST mean among all sister bases that cross
       2sigma.
    3. db8_CONFIRMED_OPTIMAL: NULL hypothesis preserved (no sister basis crosses
       2sigma in the favorable direction).
    """
    baseline = per_basis_results.get(baseline_basis)
    if baseline is None or baseline.get("valid_segment_count", 0) == 0:
        return {
            "verdict": "INSUFFICIENT_DISCRIMINATION",
            "dominant_basis": "baseline_db8_invalid_no_baseline_data",
            "per_basis_z_score_vs_db8_baseline": {},
            "ci_overlap_matrix": {},
            "operator_signal": "INSUFFICIENT_DISCRIMINATION",
            "rationale": (
                "baseline db8 produced zero valid segments; cannot compute "
                "discrimination."
            ),
        }

    baseline_mean = baseline["segment_textured_avg_weight_mean"]
    baseline_stdev = baseline["segment_textured_avg_weight_stdev"]
    baseline_n = baseline["valid_segment_count"]
    baseline_sem = baseline_stdev / np.sqrt(float(baseline_n)) if baseline_n > 1 else float("inf")
    baseline_ci = (
        baseline["bootstrap_ci_mean"]["ci_lower"],
        baseline["bootstrap_ci_mean"]["ci_upper"],
    )

    per_basis_z: dict[str, float] = {}
    per_basis_delta: dict[str, float] = {}
    per_basis_relative_delta_pct: dict[str, float] = {}
    ci_overlap_matrix: dict[str, dict[str, float]] = {}
    sister_dominates_candidates: list[tuple[str, float, float]] = []  # (basis, z, mean)

    for basis_name, basis_result in per_basis_results.items():
        if basis_result.get("valid_segment_count", 0) == 0:
            per_basis_z[basis_name] = float("nan")
            per_basis_delta[basis_name] = float("nan")
            per_basis_relative_delta_pct[basis_name] = float("nan")
            continue

        b_mean = basis_result["segment_textured_avg_weight_mean"]
        b_stdev = basis_result["segment_textured_avg_weight_stdev"]
        b_n = basis_result["valid_segment_count"]
        b_sem = b_stdev / np.sqrt(float(b_n)) if b_n > 1 else float("inf")
        combined_sem = float(np.sqrt(baseline_sem**2 + b_sem**2))
        absolute_delta = b_mean - baseline_mean
        relative_delta_pct = (
            100.0 * absolute_delta / baseline_mean if baseline_mean != 0 else float("nan")
        )
        z = absolute_delta / combined_sem if combined_sem > 0 else float("inf")

        per_basis_z[basis_name] = z
        per_basis_delta[basis_name] = absolute_delta
        per_basis_relative_delta_pct[basis_name] = relative_delta_pct

        # SISTER_BASIS_DOMINATES: |z| > 2sigma AND mean is LOWER than baseline
        # (UNIWARD inverse-cost form: lower weight = sharper inversion = better signal)
        if basis_name != baseline_basis and abs(z) > SISTER_BASIS_DOMINATES_Z_THRESHOLD and b_mean < baseline_mean:
            sister_dominates_candidates.append((basis_name, z, b_mean))

        # CI overlap matrix
        b_ci = (
            basis_result["bootstrap_ci_mean"]["ci_lower"],
            basis_result["bootstrap_ci_mean"]["ci_upper"],
        )
        ci_overlap_matrix[basis_name] = {}
        for other_name, other_result in per_basis_results.items():
            if other_result.get("valid_segment_count", 0) == 0:
                ci_overlap_matrix[basis_name][other_name] = float("nan")
                continue
            other_ci = (
                other_result["bootstrap_ci_mean"]["ci_lower"],
                other_result["bootstrap_ci_mean"]["ci_upper"],
            )
            ci_overlap_matrix[basis_name][other_name] = _compute_pairwise_ci_overlap_fraction(
                b_ci, other_ci
            )

    # INSUFFICIENT_DISCRIMINATION: check ALL non-self pairs overlap > threshold
    non_self_overlaps = []
    basis_names_list = list(per_basis_results.keys())
    for i, name_a in enumerate(basis_names_list):
        for name_b in basis_names_list[i + 1:]:
            ov = ci_overlap_matrix.get(name_a, {}).get(name_b, float("nan"))
            if not np.isnan(ov):
                non_self_overlaps.append(ov)
    all_overlaps_exceed_threshold = (
        len(non_self_overlaps) > 0
        and all(ov > CI_OVERLAP_FRACTION_INSUFFICIENT for ov in non_self_overlaps)
    )

    # VERDICT decision cascade
    if sister_dominates_candidates:
        # SISTER_BASIS_DOMINATES: pick the sister with the LOWEST mean
        sister_dominates_candidates.sort(key=lambda x: x[2])  # sort by mean ascending
        dominant_basis_name = sister_dominates_candidates[0][0]
        dominant_z = sister_dominates_candidates[0][1]
        dominant_mean = sister_dominates_candidates[0][2]
        verdict = f"SISTER_BASIS_DOMINATES_{dominant_basis_name}"
        operator_signal = f"SISTER_BASIS_DOMINATES_{dominant_basis_name}"
        rationale = (
            f"Sister basis {dominant_basis_name} (mean {dominant_mean:.4f}) is "
            f"significantly LOWER than baseline db8 (mean {baseline_mean:.4f}); "
            f"z-score {dominant_z:.3f}σ (|z| > {SISTER_BASIS_DOMINATES_Z_THRESHOLD:.1f}σ). "
            f"Per Catalog #307 IMPLEMENTATION-LEVEL falsification: paradigm INTACT; "
            f"Probe 9 substrate recipe canonical update required to use {dominant_basis_name}. "
            f"Per CLAUDE.md 'Forbidden premature KILL': substrate paradigm preserved."
        )
    elif all_overlaps_exceed_threshold:
        verdict = "INSUFFICIENT_DISCRIMINATION"
        operator_signal = "INSUFFICIENT_DISCRIMINATION"
        dominant_basis_name = "indistinguishable_variance_dominates"
        rationale = (
            f"All basis-pair CIs overlap by > {CI_OVERLAP_FRACTION_INSUFFICIENT*100:.0f}% "
            f"(non-self overlap fractions: {[f'{ov:.2f}' for ov in non_self_overlaps]}). "
            f"Variance dominates; no basis discriminates at the 2sigma level. "
            f"Per Catalog #308 alternative reducer cascade: queue 2-level vs 3-level vs "
            f"4-level decomposition OR per-region vs per-instance UNIWARD OR HILL/S-UNIWARD."
        )
    else:
        verdict = "db8_CONFIRMED_OPTIMAL"
        operator_signal = "db8_CONFIRMED_OPTIMAL"
        dominant_basis_name = "indistinguishable_db8_confirmed"
        # Find max |z| among sister bases for rationale
        sister_zs = [(name, z) for name, z in per_basis_z.items() if name != baseline_basis and not np.isnan(z)]
        max_z = max(sister_zs, key=lambda x: abs(x[1])) if sister_zs else (None, float("nan"))
        rationale = (
            f"NULL hypothesis PRESERVED: no sister basis crosses 2sigma threshold in the "
            f"favorable direction (lower mean = sharper UNIWARD inversion). Max sister |z| = "
            f"{abs(max_z[1]):.3f}σ ({max_z[0]}). Baseline db8 (Probe 9 / 9b anchor) is "
            f"canonical-optimal at the per-level wavelet-basis selection surface. Mallat "
            f"binding revision #3 CLEARED."
        )

    return {
        "verdict": verdict,
        "dominant_basis": dominant_basis_name,
        "per_basis_z_score_vs_db8_baseline": per_basis_z,
        "per_basis_absolute_delta_vs_db8_baseline": per_basis_delta,
        "per_basis_relative_delta_pct_vs_db8_baseline": per_basis_relative_delta_pct,
        "ci_overlap_matrix": ci_overlap_matrix,
        "operator_signal": operator_signal,
        "rationale": rationale,
        "non_self_ci_overlaps": non_self_overlaps,
        "all_overlaps_exceed_insufficient_threshold": all_overlaps_exceed_threshold,
        "sister_dominates_candidates_sorted_by_mean_ascending": [
            {"basis": name, "z_score_vs_db8": z, "mean": mean}
            for name, z, mean in sister_dominates_candidates
        ],
    }


# ---------------------------------------------------------------------------
# Top-level probe runner
# ---------------------------------------------------------------------------


def _run_probe(
    *,
    wavelet_basis_family: tuple[str, ...],
    n_pairs: int,
    source_video: Path,
    seed: int,
) -> dict[str, Any]:
    """Run Probe 9c per-level wavelet-basis selection disambiguator."""
    t_start = time.time()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")  # NEVER MPS per Catalog #1 silent-fallback ban

    # === Load canonical SegNet (verbatim Probe 9b pattern) ===
    segnet = SegNet()
    segnet_sd_path = REPO_ROOT / "upstream" / "models" / "segnet.safetensors"
    sd = load_file(str(segnet_sd_path))
    missing, unexpected = segnet.load_state_dict(sd, strict=False)
    segnet = segnet.eval().to(device)

    # === Decode N+1 frames -> N pairs ===
    n_frames_required = n_pairs + 1
    frames, source_video_sha = _decode_first_n_frames(source_video, n=n_frames_required)
    n_frames_decoded, _, H, W = frames.shape

    # Build per-pair tensors: (n_pairs, 2, 3, H, W)
    pairs = torch.stack(
        [torch.stack([frames[i], frames[i + 1]], dim=0) for i in range(n_pairs)],
        dim=0,
    ).to(device)

    # === SegNet forward across all pairs (batched per memory-safety) ===
    BATCH = 8
    seg_mask_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_start in range(0, n_pairs, BATCH):
            batch_end = min(batch_start + BATCH, n_pairs)
            batch = pairs[batch_start:batch_end]
            seg_input = segnet.preprocess_input(batch)
            seg_logits = segnet(seg_input)
            seg_mask_chunks.append(seg_logits.argmax(dim=1))
    seg_mask = torch.cat(seg_mask_chunks, dim=0)

    # === Iterate over canonical wavelet basis family ===
    per_basis_results: dict[str, dict[str, Any]] = {}
    for basis_name in wavelet_basis_family:
        print(f"[probe_9c] Running per-basis ablation for wavelet '{basis_name}'...", flush=True)
        basis_result = _run_basis_probe(
            wavelet_name=basis_name,
            pairs=pairs,
            seg_mask=seg_mask,
            frames=frames,
            n_pairs=n_pairs,
            rng=rng,
        )
        per_basis_results[basis_name] = basis_result
        print(
            f"[probe_9c]   '{basis_name}': "
            f"mean={basis_result['segment_textured_avg_weight_mean']:.4f} "
            f"valid_segments={basis_result['valid_segment_count']} "
            f"below_threshold={basis_result['segment_textured_below_threshold_fraction']:.4f} "
            f"elapsed={basis_result['elapsed_seconds']:.2f}s",
            flush=True,
        )

    # === Cross-basis verdict + dominant basis ===
    verdict_data = _classify_verdict(per_basis_results, baseline_basis=BASELINE_BASIS)

    elapsed_seconds = time.time() - t_start

    # === Next-action message per Carmack MVP-first step 5 ===
    if verdict_data["verdict"] == "db8_CONFIRMED_OPTIMAL":
        next_action = (
            f"UNBLOCK_2_OF_3_DISPATCH_BLOCKING: db8 (Probe 9 / 9b anchor) is empirically "
            f"canonical-optimal across canonical wavelet basis family {wavelet_basis_family}. "
            f"Mallat binding revision #3 of 6 SATISFIED. Of 3 dispatch-blocking revisions: "
            f"#1 (Contrarian sister-probe 100-pair) CLEARED; #3 (Mallat per-level basis) CLEARED. "
            f"REMAINING: paired CPU+CUDA empirical anchor (emergent from revisions 4-5-6 + sister "
            f"subagent `_full_main` build). Per Carmack MVP-first cascade: queue paired CPU+CUDA "
            f"dispatch authorization OR sister subagent `_full_main` implementation per Daubechies / "
            f"Quantizr / Selfcomp binding revisions 4-5-6."
        )
    elif verdict_data["verdict"].startswith("SISTER_BASIS_DOMINATES_"):
        dominant_basis = verdict_data["dominant_basis"]
        next_action = (
            f"DEFER_PER_CATALOG_307_IMPLEMENTATION_LEVEL_FALSIFICATION: sister basis "
            f"{dominant_basis} empirically dominates db8 baseline. Per Catalog #307: paradigm "
            f"INTACT (per-instance + multi-scale wavelet UNIWARD-weighted SegNet loss); db8-specific "
            f"implementation FALSIFIED at per-level basis surface. Per CLAUDE.md 'Forbidden premature "
            f"KILL': substrate recipe canonical update required to use {dominant_basis} as new "
            f"baseline; queue NEW sister canonical equation candidate "
            f"`uniward_per_instance_multi_scale_wavelet_basis_optimal_{dominant_basis}_v1`. "
            f"Probe 9b sister rerun at {dominant_basis} recommended for substrate-recipe "
            f"canonical-fix evidence; sister subagent `_full_main` build should use {dominant_basis}."
        )
    elif verdict_data["verdict"] == "INSUFFICIENT_DISCRIMINATION":
        next_action = (
            f"INSUFFICIENT_DISCRIMINATION_QUEUE_ALTERNATIVE_REDUCER: variance dominates across "
            f"canonical wavelet basis family {wavelet_basis_family}; no basis discriminates at "
            f"2sigma. Per Carmack MVP-first step 5 + Catalog #308 alternative-reducer cascade: "
            f"queue (a) 2-level vs 3-level vs 4-level wavelet decomposition sweep; OR (b) per-region "
            f"UNIWARD vs per-instance UNIWARD ablation; OR (c) alternative steganographic distortion "
            f"measure (HILL Holub-Fridrich 2014 OR S-UNIWARD Holub-Fridrich-Denemark 2013); OR "
            f"(d) Probe 9c rerun at N=600 full-contest-video sample for higher statistical "
            f"confidence (~1-2 hr macOS-CPU, $0)."
        )
    else:
        next_action = "UNKNOWN_VERDICT — operator-routable review required"

    # === Canonical Provenance per Catalog #287 + #323 + #341 ===
    canonical_provenance = {
        "kind": "macos_cpu_advisory",
        "axis_tag": PROVENANCE_AXIS_TAG,
        "hardware_substrate": PROVENANCE_HARDWARE_SUBSTRATE,
        "evidence_grade": PROVENANCE_EVIDENCE_GRADE,
        "score_claim_valid": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "source_video_path": str(source_video),
        "source_video_sha256": source_video_sha,
        "source_video_n_frames_decoded": n_frames_decoded,
        "device": "cpu",
        "wavelet_basis_family": list(wavelet_basis_family),
        "wavelet_levels": WAVELET_LEVELS,
        "sigma_fridrich": SIGMA_FRIDRICH,
        "min_segment_pixels_threshold": MIN_SEGMENT_PIXELS,
        "min_textured_pixels_threshold": MIN_TEXTURED_PIXELS,
        "segnet_eval_resolution_HxW": [SEGNET_EVAL_H, SEGNET_EVAL_W],
        "segnet_missing_keys": len(missing),
        "segnet_unexpected_keys": len(unexpected),
        "rng_seed": seed,
        "bootstrap_iterations": BOOTSTRAP_ITERATIONS,
        "fridrich_canonical_reference": (
            "Holub-Fridrich-Denemark 2014 'Universal Distortion Function for "
            "Steganography in an Arbitrary Domain' (UNIWARD) multi-level wavelet "
            "extension; CLAUDE.md 'Fridrich inverse steganalysis' section."
        ),
        "mallat_canonical_reference": (
            "Mallat 1989 'A theory for multiresolution signal decomposition: "
            "the wavelet representation' + Daubechies 1988 'Orthonormal bases "
            "of compactly supported wavelets'."
        ),
        "predecessor_probe": (
            "tools/probe_9b_100_pair_uniward_per_instance_multi_scale_wavelet_combined_disambiguator.py "
            "(N=100 db8 baseline; commit 686b07f31; verdict REPLICATES_WITHIN_BAND)"
        ),
        "sister_canonical_equation_candidate": (
            "uniward_per_instance_multi_scale_wavelet_combined_v1 "
            "(Catalog #344 FORMALIZATION_PENDING; RATIFY-N evidence count update if db8_CONFIRMED_OPTIMAL)"
        ),
    }

    # === Assemble final verdict JSON ===
    verdict_record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "probe_id": "probe_9c_per_level_wavelet_basis_selection_disambiguator_20260525",
        "lane_id": "lane_probe_9c_per_level_wavelet_basis_disambiguator_20260525",
        "probe_name": (
            "Probe 9c per-level wavelet-basis selection disambiguator "
            "(extends Probe 9b N=100 db8 baseline to wavelet basis family "
            "db4 / db8 / db16 / bior4.4 per Mallat binding revision #3)"
        ),
        "evidence_grade": PROVENANCE_EVIDENCE_GRADE,
        "axis_tag": PROVENANCE_AXIS_TAG,
        "promotable": False,
        "score_claim": False,
        "ready_for_exact_eval_dispatch": False,
        "device": "cpu",
        "hardware_substrate": PROVENANCE_HARDWARE_SUBSTRATE,
        "elapsed_seconds": elapsed_seconds,
        "n_pairs": n_pairs,
        "source_video_sha256": source_video_sha,
        "wavelet_basis_family": list(wavelet_basis_family),
        "baseline_basis": BASELINE_BASIS,
        "baseline_db8_n100_anchor": PROBE_9B_N100_DB8_BASELINE,
        "predicted_signature": {
            "db8_CONFIRMED_OPTIMAL": (
                "NULL hypothesis preserved: no sister basis crosses 2sigma threshold "
                "in the favorable direction (lower mean = sharper UNIWARD inversion)"
            ),
            "SISTER_BASIS_DOMINATES_<basis>": (
                f"|z-score(sister vs db8 baseline)| > {SISTER_BASIS_DOMINATES_Z_THRESHOLD:.1f}σ "
                f"AND sister mean is LOWER than baseline (per Catalog #307 IMPLEMENTATION-LEVEL "
                f"falsification; paradigm intact)"
            ),
            "INSUFFICIENT_DISCRIMINATION": (
                f"all basis-pair CIs overlap by > {CI_OVERLAP_FRACTION_INSUFFICIENT*100:.0f}% "
                f"(variance dominates; queue alternative reducer per Catalog #308)"
            ),
        },
        "per_basis_results": per_basis_results,
        "cross_basis_classification": verdict_data,
        "verdict": verdict_data["verdict"],
        "dominant_basis": verdict_data["dominant_basis"],
        "next_action": next_action,
        "operator_signal": verdict_data["operator_signal"],
        "canonical_provenance": canonical_provenance,
        "canonical_equation_reference": (
            "sister candidate uniward_per_instance_multi_scale_wavelet_combined_v1 "
            "(Catalog #344 FORMALIZATION_PENDING; RATIFY-N evidence count update via this "
            "anchor if db8_CONFIRMED_OPTIMAL; new sister candidate queued if SISTER_BASIS_DOMINATES)"
        ),
        "catalog_references": [
            "#1",
            "#192",
            "#287",
            "#290",
            "#307",
            "#308",
            "#313",
            "#323",
            "#341",
            "#344",
        ],
        "predecessor_probe_anchor": {
            "predecessor_probe_id": "probe_9b_100_pair_uniward_per_instance_multi_scale_wavelet_combined_disambiguator_20260525",
            "predecessor_commit": "686b07f31",
            "predecessor_verdict": "REPLICATES_WITHIN_BAND",
            "predecessor_basis": "db8",
            "predecessor_mean": PROBE_9B_N100_DB8_BASELINE["segment_textured_avg_weight_mean"],
            "predecessor_valid_segments": PROBE_9B_N100_DB8_BASELINE["valid_segment_count"],
            "predecessor_evidence_path": PROBE_9B_N100_DB8_BASELINE["evidence_path"],
        },
        "produced_at_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "produced_pid": int(__import__("os").getpid()),
        "produced_host": socket.gethostname(),
    }

    return verdict_record


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_basis_family(arg: str) -> tuple[str, ...]:
    """Parse comma-separated wavelet basis list; validate against pywt."""
    candidates = tuple(s.strip() for s in arg.split(",") if s.strip())
    available = set(pywt.wavelist())
    invalid = [b for b in candidates if b not in available]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid wavelet basis names not in pywt.wavelist(): {invalid}. "
            f"Canonical family per Mallat revision #3: db4, db8, db16, bior4.4"
        )
    if not candidates:
        raise argparse.ArgumentTypeError("--wavelet-basis-family must be non-empty")
    return candidates


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe 9c: per-level wavelet-basis selection disambiguator. Extends "
            "Probe 9b N=100 db8 baseline (commit 686b07f31) to canonical wavelet "
            "basis family per Mallat binding revision #3 of 6 (BLOCKS DISPATCH). "
            "Tests db4 / db8 / db16 / bior4.4 at $0 macOS-CPU advisory."
        )
    )
    parser.add_argument(
        "--wavelet-basis-family",
        type=_parse_basis_family,
        default=CANONICAL_WAVELET_BASIS_FAMILY,
        help=(
            "Comma-separated canonical wavelet basis family. Default per Mallat "
            "revision #3: db4,db8,db16,bior4.4"
        ),
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=100,
        help="Number of pairs to probe (default 100; identical to Probe 9b baseline).",
    )
    parser.add_argument(
        "--source-video",
        type=Path,
        default=REPO_ROOT / "upstream" / "videos" / "0.mkv",
        help="Path to contest video (default upstream/videos/0.mkv).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Default: "
            ".omx/research/tier_1_distortion_axis_probes_20260521/"
            "probe_9c_per_level_wavelet_basis_<utc>.json"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for bootstrap (default 42; identical to Probe 9b baseline).",
    )
    args = parser.parse_args()

    if args.n_pairs < 2:
        print(f"[probe_9c] FATAL: --n-pairs must be >= 2; got {args.n_pairs}", file=sys.stderr)
        return 2
    if not args.source_video.exists():
        print(
            f"[probe_9c] FATAL: source video not found at {args.source_video}",
            file=sys.stderr,
        )
        return 2

    if args.output_json is None:
        utc = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        out_dir = REPO_ROOT / ".omx" / "research" / "tier_1_distortion_axis_probes_20260521"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / f"probe_9c_per_level_wavelet_basis_{utc}.json"
    else:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[probe_9c] Running per-basis ablation across "
        f"{args.wavelet_basis_family} on {args.source_video} (n_pairs={args.n_pairs}; "
        f"seed={args.seed})..."
    )

    verdict_record = _run_probe(
        wavelet_basis_family=args.wavelet_basis_family,
        n_pairs=args.n_pairs,
        source_video=args.source_video,
        seed=args.seed,
    )

    args.output_json.write_text(
        json.dumps(verdict_record, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(
        f"[probe_9c] verdict={verdict_record['verdict']} "
        f"operator_signal={verdict_record['operator_signal']} "
        f"dominant_basis={verdict_record['dominant_basis']} "
        f"elapsed={verdict_record['elapsed_seconds']:.2f}s"
    )
    print(f"[probe_9c] wrote {args.output_json}")

    # Exit code: 0 on db8_CONFIRMED_OPTIMAL (NULL preserved; clears Mallat revision);
    # 1 on SISTER_BASIS_DOMINATES (operator-routable recipe canonical update);
    # 2 on INSUFFICIENT_DISCRIMINATION (queue alternative reducer)
    if verdict_record["verdict"] == "db8_CONFIRMED_OPTIMAL":
        return 0
    if verdict_record["verdict"].startswith("SISTER_BASIS_DOMINATES_"):
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
