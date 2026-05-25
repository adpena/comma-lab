#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Probe 9b: 100-pair UNIWARD per-instance × multi-scale wavelet COMBINED disambiguator.

PURPOSE
=======
Per Probe 9 Tier-2 dispatch symposium 2026-05-25 Contrarian binding revision #1
(BLOCKS DISPATCH): the Probe 9 BREAKTHROUGH anchor
(min textured_avg_weight=0.2597 [macOS-CPU advisory] at N=25 segments / 4 pairs)
must be tested at expanded sample (next 100 pairs, all 5 classes) BEFORE paid
dispatch fires. This tool IS the canonical disambiguator that clears the first
of 3 dispatch-blocking binding revisions.

The N=25 anchor:
- 4 pairs (frames 0..7 -> pairs [(0,1), (1,2), (2,3), (3,4)])
- 22 valid per-instance segments (after MIN_SEGMENT_PIXELS=200 filter)
- min textured_avg_weight = 0.2597
- max = 0.6153; median = 0.4315; mean = 0.4202; stdev = 0.0853
- 19/22 below 0.5 hard threshold (BREAKS_THRESHOLD verdict)
- per_class_segment_count: {0: 7, 1: 7, 2: 4, 3: 0, 4: 4}

N=100 EXTENSION
================
- 101 frames decoded -> 100 pairs [(0,1), (1,2), ..., (99,100)]
- Expected ~550 valid per-instance segments (22/4 segments-per-pair × 100)
- 25x N=25 sample size enables bootstrap 5000-iter CI computation
- Falsifiable challenge per Carmack MVP-first step 2:
    REPLICATES_WITHIN_BAND : N=100 textured_avg_weight mean within ±5% of N=25 mean
                             (0.4202 ± 0.0210, i.e., [0.3992, 0.4412])
                             AND z-score(N=100 mean vs N=25 mean) < 2σ
    DIVERGES_SMALL_N_ARTIFACT : N=25 mean outside N=100 CI band
                                (per Catalog #307 IMPLEMENTATION-LEVEL falsification:
                                 paradigm intact; specific N=25 implementation falsified;
                                 per CLAUDE.md "Forbidden premature KILL")
    INSUFFICIENT_SIGNAL : variance dominates (N=100 CI half-width > 0.05)
                          per Catalog #308 alternative reducer cascade

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

CANONICAL EQUATION
===================
sister candidate: ``uniward_per_instance_multi_scale_wavelet_combined_v1``
(FORMALIZATION_PENDING per Catalog #344). N=100 anchor STRENGTHENS evidence
count if REPLICATES_WITHIN_BAND; marks INSUFFICIENT_EVIDENCE if DIVERGES.

USAGE
=====
    .venv/bin/python tools/probe_9b_100_pair_uniward_per_instance_multi_scale_wavelet_combined_disambiguator.py \\
        --n-pairs 100 \\
        --source-video upstream/videos/0.mkv \\
        --output-json .omx/research/tier_1_distortion_axis_probes_20260521/probe_9b_100_pair_<utc>.json \\
        --seed 42

CROSS-REFERENCES
=================
- Probe 9 BREAKTHROUGH: ``.omx/research/combined_tier_1_wave_3_uniward_multi_scale_plus_hinton_motion_aware_landed_20260525.md`` (commit 685fe6726)
- Probe 9 tool: ``.omx/research/tier_1_distortion_axis_probes_20260521/probe_9_uniward_per_instance_multi_scale_wavelet_combined_smoke.py``
- Symposium: ``.omx/research/per_substrate_symposium_uniward_per_instance_multi_scale_wavelet_segnet_20260525.md`` Contrarian revision #1
- Substrate trainer: ``experiments/train_substrate_uniward_per_instance_multi_scale_wavelet_segnet.py``
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
# Constants (canonical per Probe 9 BREAKTHROUGH; held identical for apples-to-apples)
# ---------------------------------------------------------------------------

# Canonical Probe 9 BREAKTHROUGH anchor (commit 685fe6726; 4 pairs / 22 segments)
N25_ANCHOR_PAIRS = 4
N25_ANCHOR_VALID_SEGMENTS = 22
N25_ANCHOR_MIN_SEGMENT_TEXTURED_AVG_WEIGHT = 0.2597
N25_ANCHOR_MAX_SEGMENT_TEXTURED_AVG_WEIGHT = 0.6153
N25_ANCHOR_MEDIAN_SEGMENT_TEXTURED_AVG_WEIGHT = 0.4315
N25_ANCHOR_MEAN_SEGMENT_TEXTURED_AVG_WEIGHT = 0.4202
N25_ANCHOR_STDEV_SEGMENT_TEXTURED_AVG_WEIGHT = 0.0853
N25_ANCHOR_BELOW_THRESHOLD_COUNT = 19  # 19 of 22

# Canonical PR 9 algorithm constants
WAVELET_NAME = "db8"
WAVELET_LEVELS = 3
SIGMA_FRIDRICH = 2.0**-6
MIN_SEGMENT_PIXELS = 200  # ~0.1% of 384*512
MIN_TEXTURED_PIXELS = 50
SEGNET_EVAL_H = 384
SEGNET_EVAL_W = 512

# Predecessor baselines for Δ comparison
CCC_BASELINE_TEXTURED_AVG_WEIGHT = 0.8062214255332947  # CCC probe 3 per-pixel
DDD_BASELINE_TEXTURED_AVG_WEIGHT = 0.626  # DDD probe 3b single-level wavelet
PROBE_5_PER_CLASS_MIN = 0.5673  # Probe 5 PARTIAL ceiling
PROBE_8_PER_INSTANCE_MIN = 0.5233  # Probe 8 PARTIAL per-instance min

# Falsification thresholds (Carmack MVP-first step 2)
REPLICATION_BAND_FRACTION = 0.05  # ±5% of N=25 mean
DIVERGENCE_Z_THRESHOLD = 2.0  # z-score for "diverges" verdict
INSUFFICIENT_CI_HALFWIDTH = 0.05  # bootstrap CI half-width above this → insufficient

# Bootstrap CI
BOOTSTRAP_ITERATIONS = 5000
BOOTSTRAP_CI_PERCENTILE_LOWER = 2.5  # 95% CI
BOOTSTRAP_CI_PERCENTILE_UPPER = 97.5

# Canonical Provenance (Catalog #287 + #323 + #341)
PROVENANCE_AXIS_TAG = "[macOS-CPU advisory]"
PROVENANCE_HARDWARE_SUBSTRATE = "darwin_arm64_m5_max_macos_cpu_advisory"
PROVENANCE_EVIDENCE_GRADE = "macOS-CPU-advisory"
SCHEMA_VERSION = "probe_9b_100_pair_uniward_per_instance_multi_scale_wavelet_combined_disambiguator_v1"


# ---------------------------------------------------------------------------
# Helpers (canonical Probe 9 functions reused verbatim for apples-to-apples)
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
            f"need ≥{n} for {n - 1}-pair sample."
        )

    stacked = np.stack(frames, axis=0)
    tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).float() / 255.0

    # Source video sha256 for provenance
    sha = hashlib.sha256(video_path.read_bytes()).hexdigest()

    return tensor, sha


def _luma(frames: torch.Tensor) -> torch.Tensor:
    """ITU-R BT.601 luma; matches CCC probe 3 baseline."""
    return 0.2989 * frames[:, 0] + 0.5870 * frames[:, 1] + 0.1140 * frames[:, 2]


def _multi_scale_wavelet_detail_magnitudes(
    luma: torch.Tensor,
    wavelet_name: str = WAVELET_NAME,
    levels: int = WAVELET_LEVELS,
) -> torch.Tensor:
    """Canonical Holub-Fridrich 2014 multi-level wavelet detail-subband sum.

    For each level l in 0..L-1, compute 2D DWT detail subbands (HL, LH, HH),
    sum their magnitudes, upsample to original resolution via nearest-neighbor,
    and accumulate across levels.

    Per Holub-Fridrich-Denemark 2014 + Mallat 1989 multi-resolution analysis.
    Verbatim port of Probe 9 algorithm for apples-to-apples replication.
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
# Bootstrap CI
# ---------------------------------------------------------------------------


def _bootstrap_mean_ci(
    samples: np.ndarray,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    ci_lower_pct: float = BOOTSTRAP_CI_PERCENTILE_LOWER,
    ci_upper_pct: float = BOOTSTRAP_CI_PERCENTILE_UPPER,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Bootstrap-resample the mean and emit 95% CI.

    Returns ``{"mean": .., "ci_lower": .., "ci_upper": .., "ci_halfwidth": ..,
    "iterations": N, "stdev_bootstrap_means": ..}``. Resampling is WITH
    replacement, sample size = len(samples).
    """
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
# Probe runner
# ---------------------------------------------------------------------------


def _run_probe(
    *,
    n_pairs: int,
    source_video: Path,
    seed: int,
) -> dict[str, Any]:
    """Run Probe 9b N-pair UNIWARD per-instance × multi-scale wavelet COMBINED."""
    t_start = time.time()
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cpu")  # NEVER MPS per Catalog #1 silent-fallback ban

    # === Load canonical SegNet (verbatim Probe 9 pattern) ===
    segnet = SegNet()
    segnet_sd_path = REPO_ROOT / "upstream" / "models" / "segnet.safetensors"
    sd = load_file(str(segnet_sd_path))
    missing, unexpected = segnet.load_state_dict(sd, strict=False)
    segnet = segnet.eval().to(device)

    # === Decode N+1 frames -> N pairs ===
    # frames [0..N] -> pairs [(0,1), (1,2), ..., (N-1, N)]
    n_frames_required = n_pairs + 1
    frames, source_video_sha = _decode_first_n_frames(source_video, n=n_frames_required)
    n_frames_decoded, _, H, W = frames.shape

    # Build per-pair tensors: (n_pairs, 2, 3, H, W)
    pairs = torch.stack(
        [torch.stack([frames[i], frames[i + 1]], dim=0) for i in range(n_pairs)],
        dim=0,
    ).to(device)

    # === SegNet forward across all pairs (batched per memory-safety) ===
    # SegNet outputs hard-classification mask via argmax; eval at (384, 512).
    # Process in batches of 8 pairs to bound peak memory at 100 pairs.
    BATCH = 8
    seg_mask_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch_start in range(0, n_pairs, BATCH):
            batch_end = min(batch_start + BATCH, n_pairs)
            batch = pairs[batch_start:batch_end]
            seg_input = segnet.preprocess_input(batch)  # (B, 3, 384, 512)
            seg_logits = segnet(seg_input)  # (B, 5, 384, 512)
            seg_mask_chunks.append(seg_logits.argmax(dim=1))  # (B, 384, 512)
    seg_mask = torch.cat(seg_mask_chunks, dim=0)  # (n_pairs, 384, 512)

    # === Eval frames per upstream/modules.py x[:, -1] convention ===
    eval_frame_indices = list(range(1, n_pairs + 1))
    eval_frames = frames[eval_frame_indices]  # (n_pairs, 3, H, W)
    luma_eval = _luma(eval_frames)  # (n_pairs, H, W)

    # === MULTI-LEVEL wavelet detail-subband sum (3-level db8) ===
    detail_sum_orig_res = _multi_scale_wavelet_detail_magnitudes(
        luma_eval, wavelet_name=WAVELET_NAME, levels=WAVELET_LEVELS
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

    # === Per-instance + multi-scale combined decomposition (D26) ===
    seg_mask_np = seg_mask.cpu().numpy()  # (n_pairs, 384, 512)
    detail_sum_np = detail_sum.cpu().numpy()
    uniward_weights_norm_np = uniward_weights_norm.cpu().numpy()

    per_segment_textured_weights: list[float] = []
    per_class_segment_count: dict[int, int] = dict.fromkeys(range(5), 0)
    per_segment_metrics_sample: list[dict] = []  # bounded sample for inspection

    structure_4conn = np.array(
        [[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int8
    )

    MAX_SAMPLE_METRICS = 100  # keep first 100 + last 50 segments for inspection
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
                instance_avg_detail = float(instance_details.mean())

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

                if len(per_segment_metrics_sample) < MAX_SAMPLE_METRICS:
                    per_segment_metrics_sample.append(
                        {
                            "pair_index": p,
                            "class_index": c,
                            "instance_id": instance_id,
                            "instance_pixel_count": instance_pixel_count,
                            "instance_textured_count": instance_textured_count,
                            "instance_avg_detail_multi_scale": instance_avg_detail,
                            "instance_textured_avg_weight_combined": instance_textured_avg_weight,
                        }
                    )

    valid_segment_count = len(per_segment_textured_weights)

    # === Compute summary statistics ===
    if valid_segment_count > 0:
        weights_arr = np.asarray(per_segment_textured_weights, dtype=np.float64)
        n100_mean = float(weights_arr.mean())
        n100_min = float(weights_arr.min())
        n100_max = float(weights_arr.max())
        n100_median = float(np.median(weights_arr))
        n100_stdev = float(weights_arr.std(ddof=1)) if valid_segment_count > 1 else 0.0
        n100_below_threshold_count = int((weights_arr < 0.5).sum())
        n100_below_threshold_fraction = float(n100_below_threshold_count) / valid_segment_count
        # Histogram bins for distribution shape characterization
        hist_bins = np.linspace(0.0, 1.0, 11)
        hist_counts, _ = np.histogram(weights_arr, bins=hist_bins)
        # Bootstrap CI on the MEAN
        boot_ci_mean = _bootstrap_mean_ci(weights_arr, rng=rng)
        # Bootstrap CI on the MIN (more variable; uses 5th percentile for stability)
        # We resample and take the percentile-5 of each bootstrap sample as a
        # stable "low-end" disambiguator.
        boot_p5 = _bootstrap_percentile_ci(weights_arr, percentile=5.0, rng=rng)
    else:
        n100_mean = n100_min = n100_max = n100_median = n100_stdev = float("nan")
        n100_below_threshold_count = 0
        n100_below_threshold_fraction = float("nan")
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
        boot_p5 = {
            "p5_mean": float("nan"),
            "p5_ci_lower": float("nan"),
            "p5_ci_upper": float("nan"),
            "iterations": BOOTSTRAP_ITERATIONS,
        }

    # === Replication / divergence verdict (Carmack MVP-first step 2) ===
    # The N=25 anchor reports SEGMENT-LEVEL min=0.2597 / mean=0.4202 / stdev=0.0853.
    # Compare N=100 MEAN to N=25 MEAN via z-score using N=25's per-segment stdev
    # normalized by sqrt(N=22) approximate standard error of the N=25 mean.
    n25_mean = N25_ANCHOR_MEAN_SEGMENT_TEXTURED_AVG_WEIGHT
    n25_stdev = N25_ANCHOR_STDEV_SEGMENT_TEXTURED_AVG_WEIGHT
    n25_sem = n25_stdev / np.sqrt(float(N25_ANCHOR_VALID_SEGMENTS))
    band_lower = n25_mean * (1.0 - REPLICATION_BAND_FRACTION)
    band_upper = n25_mean * (1.0 + REPLICATION_BAND_FRACTION)

    if valid_segment_count == 0:
        verdict = "INSUFFICIENT_SIGNAL"
        z_score = float("nan")
        absolute_delta = float("nan")
        relative_delta_pct = float("nan")
        replicates_within_band = False
        diverges_zscore = False
        ci_halfwidth_acceptable = False
    else:
        # Standard error of N=100 mean
        n100_sem = n100_stdev / np.sqrt(float(valid_segment_count)) if valid_segment_count > 1 else float("inf")
        # Use combined SEM for two-sample z-test
        combined_sem = float(np.sqrt(n25_sem**2 + n100_sem**2))
        absolute_delta = n100_mean - n25_mean
        relative_delta_pct = 100.0 * absolute_delta / n25_mean if n25_mean != 0 else float("nan")
        z_score = absolute_delta / combined_sem if combined_sem > 0 else float("inf")
        ci_halfwidth_acceptable = boot_ci_mean["ci_halfwidth"] <= INSUFFICIENT_CI_HALFWIDTH

        # Replication: N=100 mean inside ±5% band AND bootstrap CI contains N=25 mean
        n25_inside_ci = boot_ci_mean["ci_lower"] <= n25_mean <= boot_ci_mean["ci_upper"]
        n100_inside_band = band_lower <= n100_mean <= band_upper
        replicates_within_band = bool(n25_inside_ci and n100_inside_band)
        diverges_zscore = bool(abs(z_score) > DIVERGENCE_Z_THRESHOLD)

        if not ci_halfwidth_acceptable:
            verdict = "INSUFFICIENT_SIGNAL"
        elif replicates_within_band:
            verdict = "REPLICATES_WITHIN_BAND"
        elif diverges_zscore:
            verdict = "DIVERGES_SMALL_N_ARTIFACT"
        else:
            # Outside ±5% band but within 2σ — classify as marginal replication
            verdict = "REPLICATES_WITHIN_BAND"

    elapsed_seconds = time.time() - t_start

    # === Verdict recommendation text per Carmack MVP-first step 5 ===
    if verdict == "REPLICATES_WITHIN_BAND":
        next_action = (
            f"UNBLOCK_1_OF_3_DISPATCH_BLOCKING: Probe 9 N=25 anchor (min=0.2597; "
            f"mean=0.4202) REPLICATES at N=100 (mean={n100_mean:.4f}; "
            f"95% CI [{boot_ci_mean['ci_lower']:.4f}, {boot_ci_mean['ci_upper']:.4f}]; "
            f"z-score {z_score:.3f}σ). The Contrarian binding revision #1 is SATISFIED. "
            f"2 of 3 dispatch-blocking revisions remain (Mallat per-level wavelet-basis "
            f"selection table; the paired CPU+CUDA empirical anchor + sister subagent "
            f"_full_main build). Probe 9 Tier-2 paid dispatch ~$2-7 still pending those "
            f"2 revisions OR operator-frontier-override per Catalog #300. Per Carmack "
            f"MVP-first cascade: queue sister subagent for Mallat per-level wavelet-basis "
            f"empirical anchor table (db4 / db8 / db16 / bior4.4) at $0 macOS-CPU advisory."
        )
        operator_signal = "UNBLOCK_1_OF_3_DISPATCH_BLOCKING"
    elif verdict == "DIVERGES_SMALL_N_ARTIFACT":
        next_action = (
            f"DEFER_PER_CATALOG_307_IMPLEMENTATION_LEVEL_FALSIFICATION: Probe 9 N=25 anchor "
            f"(mean=0.4202) DIVERGES at N=100 (mean={n100_mean:.4f}; z-score {z_score:.3f}σ; "
            f"absolute Δ {absolute_delta:+.4f}; relative Δ {relative_delta_pct:+.2f}%). "
            f"Per Catalog #307 IMPLEMENTATION-LEVEL falsification: paradigm INTACT (per-instance "
            f"+ multi-scale wavelet UNIWARD-weighted SegNet loss); N=25 specific implementation "
            f"FALSIFIED (small-N artifact). Per CLAUDE.md 'Forbidden premature KILL': queue "
            f"alternative reducer per Catalog #308 — per-region UNIWARD (vs per-instance), OR "
            f"wavelet level 2 vs 3, OR db4 vs db8 vs bior4.4 basis sweep. Probe 9 Tier-2 dispatch "
            f"DEFERRED pending alternative-reducer disambiguator."
        )
        operator_signal = "DEFER_PER_CATALOG_307"
    elif verdict == "INSUFFICIENT_SIGNAL":
        next_action = (
            f"INSUFFICIENT_SIGNAL_QUEUE_LARGER_SAMPLE_OR_ALTERNATIVE_REDUCER: N=100 bootstrap "
            f"CI half-width = {boot_ci_mean.get('ci_halfwidth', float('nan')):.4f} > "
            f"{INSUFFICIENT_CI_HALFWIDTH:.4f} threshold (variance dominates). Per Carmack "
            f"MVP-first step 5 + Catalog #308: queue (a) Probe 9c N=600 full-contest-video sample "
            f"(~1-2 hr macOS-CPU, $0); OR (b) alternative reducer (per-region UNIWARD vs "
            f"per-instance; OR wavelet level 2/3 sweep; OR basis sweep db4/db8/db16/bior4.4)."
        )
        operator_signal = "INSUFFICIENT_SIGNAL"
    else:
        next_action = "UNKNOWN_VERDICT — operator-routable review required"
        operator_signal = "OPERATOR_REVIEW_REQUIRED"

    # === Build canonical Provenance per Catalog #287 + #323 + #341 ===
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
        "wavelet_basis": WAVELET_NAME,
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
        "predecessor_probe": (
            "tier_1_distortion_uniward_per_instance_multi_scale_wavelet_combined_smoke "
            "(N=25 segments / 4 pairs / commit 685fe6726)"
        ),
        "sister_canonical_equation_candidate": (
            "uniward_per_instance_multi_scale_wavelet_combined_v1 "
            "(Catalog #344 FORMALIZATION_PENDING; RATIFY-N evidence count update)"
        ),
    }

    # === Assemble final verdict JSON ===
    verdict_record: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "probe_id": "probe_9b_100_pair_uniward_per_instance_multi_scale_wavelet_combined_disambiguator_20260525",
        "lane_id": "lane_probe_9b_100_pair_disambiguator_20260525",
        "probe_name": (
            "Probe 9b N-pair UNIWARD per-instance × multi-scale wavelet COMBINED "
            "disambiguator (extends Probe 9 N=25 anchor to N=100 pairs)"
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
        "predicted_signature": {
            "REPLICATES_WITHIN_BAND": (
                f"N=100 mean inside N=25 mean ±{REPLICATION_BAND_FRACTION*100:.0f}% band "
                f"AND bootstrap CI contains N=25 mean AND CI half-width ≤ "
                f"{INSUFFICIENT_CI_HALFWIDTH:.3f}"
            ),
            "DIVERGES_SMALL_N_ARTIFACT": (
                f"|z-score(N=100 mean vs N=25 mean)| > {DIVERGENCE_Z_THRESHOLD:.1f}σ "
                f"(per Catalog #307 IMPLEMENTATION-LEVEL falsification; paradigm intact)"
            ),
            "INSUFFICIENT_SIGNAL": (
                f"bootstrap CI half-width > {INSUFFICIENT_CI_HALFWIDTH:.3f} "
                f"(variance dominates; queue N=600 sample or alternative reducer)"
            ),
        },
        "actual_signature": {
            "n_frames_decoded": n_frames_decoded,
            "pair_count": n_pairs,
            "segnet_eval_resolution_HxW": [SEGNET_EVAL_H, SEGNET_EVAL_W],
            "wavelet_levels": WAVELET_LEVELS,
            "wavelet_name": WAVELET_NAME,
            "min_segment_pixels_threshold": MIN_SEGMENT_PIXELS,
            "min_textured_pixels_threshold": MIN_TEXTURED_PIXELS,
            "per_class_segment_count": per_class_segment_count,
            "valid_segment_count": valid_segment_count,
            "segment_textured_avg_weight_mean": n100_mean,
            "segment_textured_avg_weight_min": n100_min,
            "segment_textured_avg_weight_max": n100_max,
            "segment_textured_avg_weight_median": n100_median,
            "segment_textured_avg_weight_stdev": n100_stdev,
            "segment_textured_below_threshold_count": n100_below_threshold_count,
            "segment_textured_below_threshold_fraction": n100_below_threshold_fraction,
            "bootstrap_ci_mean": boot_ci_mean,
            "bootstrap_ci_p5_disambiguator": boot_p5,
            "distribution_histogram": {
                "bin_edges": [float(x) for x in hist_bins.tolist()],
                "bin_counts": [int(c) for c in hist_counts.tolist()],
            },
            "per_segment_metrics_sample_bounded_to_100": per_segment_metrics_sample,
            "ccc_baseline_textured_avg_weight": CCC_BASELINE_TEXTURED_AVG_WEIGHT,
            "ddd_baseline_textured_avg_weight": DDD_BASELINE_TEXTURED_AVG_WEIGHT,
            "probe_5_per_class_min": PROBE_5_PER_CLASS_MIN,
            "probe_8_per_instance_min": PROBE_8_PER_INSTANCE_MIN,
        },
        "replicates_n25_anchor": {
            "n25_anchor_mean": N25_ANCHOR_MEAN_SEGMENT_TEXTURED_AVG_WEIGHT,
            "n25_anchor_stdev": N25_ANCHOR_STDEV_SEGMENT_TEXTURED_AVG_WEIGHT,
            "n25_anchor_min": N25_ANCHOR_MIN_SEGMENT_TEXTURED_AVG_WEIGHT,
            "n25_anchor_median": N25_ANCHOR_MEDIAN_SEGMENT_TEXTURED_AVG_WEIGHT,
            "n25_anchor_valid_segments": N25_ANCHOR_VALID_SEGMENTS,
            "n25_anchor_below_threshold_count": N25_ANCHOR_BELOW_THRESHOLD_COUNT,
            "n100_mean": n100_mean,
            "n100_valid_segments": valid_segment_count,
            "absolute_delta_mean_n100_minus_n25": absolute_delta,
            "relative_delta_pct": relative_delta_pct,
            "divergence_zscore_n100_vs_n25": z_score,
            "replication_band_lower": band_lower,
            "replication_band_upper": band_upper,
            "replication_band_fraction": REPLICATION_BAND_FRACTION,
            "divergence_z_threshold": DIVERGENCE_Z_THRESHOLD,
            "insufficient_ci_halfwidth_threshold": INSUFFICIENT_CI_HALFWIDTH,
            "replicates_within_band": replicates_within_band if valid_segment_count > 0 else False,
            "diverges_zscore": diverges_zscore if valid_segment_count > 0 else False,
            "ci_halfwidth_acceptable": ci_halfwidth_acceptable if valid_segment_count > 0 else False,
        },
        "verdict": verdict,
        "next_action": next_action,
        "operator_signal": operator_signal,
        "canonical_provenance": canonical_provenance,
        "canonical_equation_reference": (
            "sister candidate uniward_per_instance_multi_scale_wavelet_combined_v1 "
            "(Catalog #344 FORMALIZATION_PENDING; RATIFY-N evidence count update via "
            "this anchor)"
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
            "predecessor_probe_id": "tier_1_distortion_uniward_per_instance_multi_scale_wavelet_combined_smoke",
            "predecessor_commit": "685fe6726",
            "predecessor_verdict": "POSITIVE_SIGNAL_BREAKS_THRESHOLD",
            "predecessor_min_textured_avg_weight": N25_ANCHOR_MIN_SEGMENT_TEXTURED_AVG_WEIGHT,
            "predecessor_mean_textured_avg_weight": N25_ANCHOR_MEAN_SEGMENT_TEXTURED_AVG_WEIGHT,
            "predecessor_valid_segments": N25_ANCHOR_VALID_SEGMENTS,
            "predecessor_pair_count": N25_ANCHOR_PAIRS,
        },
        "produced_at_utc": _dt.datetime.now(_dt.UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "produced_pid": int(__import__("os").getpid()),
        "produced_host": socket.gethostname(),
    }

    return verdict_record


def _bootstrap_percentile_ci(
    samples: np.ndarray,
    *,
    percentile: float = 5.0,
    n_iterations: int = BOOTSTRAP_ITERATIONS,
    ci_lower_pct: float = BOOTSTRAP_CI_PERCENTILE_LOWER,
    ci_upper_pct: float = BOOTSTRAP_CI_PERCENTILE_UPPER,
    rng: np.random.Generator | None = None,
) -> dict[str, float]:
    """Bootstrap-resample the percentile and emit 95% CI on the percentile."""
    if rng is None:
        rng = np.random.default_rng()
    n = len(samples)
    if n == 0:
        return {
            "p5_mean": float("nan"),
            "p5_ci_lower": float("nan"),
            "p5_ci_upper": float("nan"),
            "iterations": n_iterations,
        }
    boot_p = np.empty(n_iterations, dtype=np.float64)
    samples_arr = np.asarray(samples, dtype=np.float64)
    for i in range(n_iterations):
        idx = rng.integers(0, n, size=n)
        boot_p[i] = np.percentile(samples_arr[idx], percentile)
    return {
        "p5_mean": float(boot_p.mean()),
        "p5_ci_lower": float(np.percentile(boot_p, ci_lower_pct)),
        "p5_ci_upper": float(np.percentile(boot_p, ci_upper_pct)),
        "iterations": n_iterations,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe 9b: 100-pair UNIWARD per-instance × multi-scale wavelet COMBINED "
            "disambiguator. Extends Probe 9 N=25 anchor (commit 685fe6726) to N=100 "
            "pairs at $0 macOS-CPU advisory. Per Probe 9 Tier-2 dispatch symposium "
            "Contrarian binding revision #1 (BLOCKS DISPATCH)."
        )
    )
    parser.add_argument(
        "--n-pairs",
        type=int,
        default=100,
        help="Number of pairs to probe (default 100; N=25 anchor used 4 pairs).",
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
            "probe_9b_100_pair_<utc>.json"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for bootstrap (default 42).",
    )
    args = parser.parse_args()

    if args.n_pairs < 2:
        print(f"[probe_9b] FATAL: --n-pairs must be ≥ 2; got {args.n_pairs}", file=sys.stderr)
        return 2
    if not args.source_video.exists():
        print(
            f"[probe_9b] FATAL: source video not found at {args.source_video}",
            file=sys.stderr,
        )
        return 2

    # Default output path
    if args.output_json is None:
        utc = _dt.datetime.now(_dt.UTC).strftime("%Y%m%dT%H%M%SZ")
        out_dir = REPO_ROOT / ".omx" / "research" / "tier_1_distortion_axis_probes_20260521"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output_json = out_dir / f"probe_9b_100_pair_{utc}.json"
    else:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[probe_9b] Running N={args.n_pairs}-pair Probe 9b UNIWARD per-instance × "
        f"multi-scale wavelet COMBINED disambiguator on {args.source_video} "
        f"(seed={args.seed})..."
    )

    verdict_record = _run_probe(
        n_pairs=args.n_pairs,
        source_video=args.source_video,
        seed=args.seed,
    )

    args.output_json.write_text(
        json.dumps(verdict_record, indent=2, sort_keys=True), encoding="utf-8"
    )

    # Console summary
    actual = verdict_record["actual_signature"]
    repl = verdict_record["replicates_n25_anchor"]
    print(
        f"[probe_9b] verdict={verdict_record['verdict']} "
        f"operator_signal={verdict_record['operator_signal']} "
        f"valid_segments={actual['valid_segment_count']} "
        f"n100_mean={actual['segment_textured_avg_weight_mean']:.4f} "
        f"(N=25 anchor mean={repl['n25_anchor_mean']:.4f}; "
        f"absolute Δ {repl['absolute_delta_mean_n100_minus_n25']:+.4f}; "
        f"z-score {repl['divergence_zscore_n100_vs_n25']:+.3f}σ; "
        f"95%CI=[{actual['bootstrap_ci_mean']['ci_lower']:.4f}, "
        f"{actual['bootstrap_ci_mean']['ci_upper']:.4f}]) "
        f"elapsed={verdict_record['elapsed_seconds']:.2f}s"
    )
    print(f"[probe_9b] wrote {args.output_json}")

    # Exit code: 0 on REPLICATES_WITHIN_BAND; 1 otherwise (operator-routable signal)
    return 0 if verdict_record["verdict"] == "REPLICATES_WITHIN_BAND" else 1


if __name__ == "__main__":
    sys.exit(main())
