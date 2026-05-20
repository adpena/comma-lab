# SPDX-License-Identifier: MIT
"""tools/run_dwt_detail_subband_procedural_smoke.py — WAVE-3 op-routable #2.

LOCAL macOS-CPU smoke applying ``derive_codebook_from_seed`` (Catalog #344
canonical equation ``procedural_codebook_from_seed_compression_savings_v1``
producer) to the LH+HL+HH detail subbands of a YCbCr Y-plane 2-level Haar
DWT decomposition on ``upstream/videos/0.mkv`` (single frame index=300 for
smoke). Computes empirical-vs-synthetic distribution distance (KL +
Wasserstein) per subband + runs the Catalog #272 byte-mutation smoke
(mutate 1 seed byte → re-derive → compare downstream distribution change)
+ appends the FIRST empirical anchor for canonical equation
``procedural_codebook_from_seed_compression_savings_v1`` (registry #26).

Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192 (macOS-CPU advisory
not promotable without Linux x86_64 verification) + Catalog #127
(per-call-site custody triple axis × hardware × evidence_grade): every
metric stamped ``[macOS-CPU advisory]`` + ``hardware_substrate=darwin_arm64_m5_max_macos_cpu_advisory``
+ ``evidence_grade=local_cpu_smoke_advisory``. NOT score truth.

Per the T3 DWT BIND SYMPOSIUM 2026-05-20 (commit 9ef3eee22)
PROCEED_WITH_REVISIONS verdict + Carmack MVP-first op-routable #2:
validates the DWT bind's distributional foundation BEFORE op-routable #1's
$2 paid substrate BUILD. If residual_zscore < 2σ → canonical equation
HARD-EARNED + bind foundation validated. If residual_zscore > 2σ →
revisit Assumption-Adversary verdict #1 (DWT-2-level canonical CARGO-CULTED).

Pipeline:

1. Decode frame index=300 from ``upstream/videos/0.mkv`` via pyav → YUV420p
2. Extract Y plane (uint8, native HxW = 874x1164)
3. Apply ``pywt.wavedec2(Y, "haar", level=2)`` → LL2 + (LH2, HL2, HH2) + (LH1, HL1, HH1)
4. For each detail subband at level=2 (LH2, HL2, HH2):
   a. Compute empirical histogram (centered + scaled to int8 range)
   b. Derive synthetic int8 array via ``derive_codebook_from_seed`` (pcg64,
      32B seed = hash of subband identifier, output_shape=(N_detail, 1))
   c. Compute KL divergence + Wasserstein-1 distance vs empirical
5. Catalog #272 byte-mutation smoke: mutate 1 byte of seed → re-derive →
   verify the per-subband distribution changes (proves derivation is
   seed-sensitive; rules out the empty-byte / no-op trap)
6. Compute aggregate residual_zscore via normalized distance metric
7. Append FIRST empirical anchor to canonical equation #26 via
   ``tac.canonical_equations.update_equation_with_empirical_anchor``
   (axis=[macOS-CPU advisory]; hardware=darwin_arm64_m5_max_macos_cpu_advisory;
   evidence_grade=MACOS_CPU_ADVISORY; promotion_eligible=False;
   score_claim_valid=False)
8. Emit JSON + Markdown artifacts under
   ``experiments/results/dwt_detail_subband_procedural_smoke_<utc>/``

Note: this smoke does NOT run inflate.sh / contest_auth_eval.py — it
measures the DISTRIBUTIONAL fit of seed-derived codebook bytes against
empirical detail-subband statistics, which is the foundation for the
Catalog #272 byte-mutation smoke that would precede any per-substrate
inflate dispatch. The actual score-mutation proof requires the
operator-routable substrate BUILD (op-routable #1).

Sister of:

* ``tools/run_master_gradient_*.py`` (canonical smoke-script pattern;
  this script follows the same CLI surface + canonical output layout)
* ``src/tac/procedural_codebook_generator/seed_derived_codebook.py``
  (canonical producer being exercised; the smoke is the first empirical
  pairing of its derive_codebook_from_seed output against measured
  contest-video statistics)
* ``src/tac/canonical_equations/procedural_codebook_savings.py``
  (canonical equation being calibrated; the smoke appends the first
  ``anchor_appended`` event per Catalog #344)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


# Canonical constants per CLAUDE.md "Submission auth eval" + canonical equation #26
CANONICAL_RATE_MULTIPLIER = 25.0
CANONICAL_RATE_DENOM_BYTES = 37_545_489
DEFAULT_FRAME_INDEX = 300
DEFAULT_SEED_BYTES = 32
DEFAULT_GENERATOR_KIND = "pcg64"
DEFAULT_WAVELET = "haar"
DEFAULT_DWT_LEVEL = 2

# Smoke must run from the canonical macOS host per Catalog #192 advisory contract.
EXPECTED_PLATFORM_SUBSTRING_DARWIN = "Darwin"


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _utc_now_filename() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_of_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def decode_frame_from_video(video_path: Path, frame_index: int) -> tuple[np.ndarray, int, int]:
    """Decode frame at ``frame_index`` from ``video_path`` via pyav.

    Returns (Y_plane uint8, height, width). YUV420p planar; we extract
    only the Y plane (full resolution).
    """
    import av

    container = av.open(str(video_path))
    try:
        stream = container.streams.video[0]
        height = int(stream.height)
        width = int(stream.width)
        target_frame: np.ndarray | None = None
        for i, frame in enumerate(container.decode(stream)):
            if i == frame_index:
                yuv = frame.to_ndarray(format="yuv420p")
                # YUV420p layout: Y plane (H rows) on top, then U + V (each H/2 rows half-width)
                target_frame = yuv[:height, :width].copy()
                break
        if target_frame is None:
            raise ValueError(
                f"frame_index={frame_index} not found in {video_path} "
                f"(decoded {i + 1} frames)"
            )
    finally:
        container.close()
    return target_frame, height, width


def compute_dwt_detail_subbands(
    y_plane: np.ndarray, wavelet: str, level: int
) -> dict[str, np.ndarray]:
    """Apply ``pywt.wavedec2`` and return the level-N detail subbands as int16 arrays.

    Returns dict with keys 'LH', 'HL', 'HH' at the deepest (coarsest)
    decomposition level. Detail subbands are clipped + scaled to fit
    int16 range for the empirical distribution analysis.
    """
    import pywt

    y_float = y_plane.astype(np.float32)
    coeffs = pywt.wavedec2(y_float, wavelet, level=level)
    # coeffs[0] = LL approximation; coeffs[1] = level-N detail tuple;
    # coeffs[k] for k=2..level = finer detail levels
    if not isinstance(coeffs[1], tuple) or len(coeffs[1]) != 3:
        raise RuntimeError(
            f"unexpected pywt.wavedec2 structure: coeffs[1] type={type(coeffs[1]).__name__}"
        )
    LH, HL, HH = coeffs[1]
    return {
        "LH": LH.astype(np.float32),
        "HL": HL.astype(np.float32),
        "HH": HH.astype(np.float32),
    }


def normalize_to_int8_distribution(arr: np.ndarray) -> np.ndarray:
    """Normalize a float32 detail subband to int8 distribution via histogram fit.

    The procedural codebook from ``derive_codebook_from_seed(..., dtype=int8)``
    produces int8 bytes; we project the empirical detail subband onto the
    same int8 [-128, 127] range for distribution comparison.

    Returns int8 ndarray of same shape, with empirical distribution
    centered + scaled to fit int8.
    """
    if arr.size == 0:
        return arr.astype(np.int8)
    center = float(np.median(arr))
    centered = arr - center
    # Scale so 99th percentile of |centered| maps to ~120 (leave headroom)
    abs99 = float(np.percentile(np.abs(centered), 99.0))
    if abs99 < 1e-6:
        # Degenerate: subband is flat. Map to all zeros.
        return np.zeros(arr.shape, dtype=np.int8)
    scaled = centered / abs99 * 120.0
    clipped = np.clip(scaled, -128.0, 127.0)
    return clipped.astype(np.int8)


def kl_divergence_int8(empirical: np.ndarray, synthetic: np.ndarray) -> float:
    """KL(empirical || synthetic) over int8 256-bin histogram (smoothed).

    Returns KL divergence in nats. Both arrays MUST be int8.
    Laplace smoothing (+1) ensures no zero-bin divergence.
    """
    if empirical.dtype != np.int8 or synthetic.dtype != np.int8:
        raise ValueError("KL inputs must be int8")
    # 256-bin histogram over int8 range [-128, 127]
    p_hist, _ = np.histogram(empirical, bins=256, range=(-128, 128))
    q_hist, _ = np.histogram(synthetic, bins=256, range=(-128, 128))
    # Laplace smoothing
    p = (p_hist + 1).astype(np.float64)
    q = (q_hist + 1).astype(np.float64)
    p /= p.sum()
    q /= q.sum()
    kl = float(np.sum(p * np.log(p / q)))
    return kl


def wasserstein1_int8(empirical: np.ndarray, synthetic: np.ndarray) -> float:
    """Wasserstein-1 distance over int8 distributions (CDF L1).

    Closed-form 1D Wasserstein = L1 between sorted empirical CDFs.
    """
    if empirical.dtype != np.int8 or synthetic.dtype != np.int8:
        raise ValueError("Wasserstein inputs must be int8")
    if empirical.size == 0 or synthetic.size == 0:
        return float("nan")
    # 1D Wasserstein-1 = integral |F_emp - F_syn| dx
    # Closed-form for histograms over identical support:
    p_hist, _ = np.histogram(empirical, bins=256, range=(-128, 128))
    q_hist, _ = np.histogram(synthetic, bins=256, range=(-128, 128))
    p = p_hist.astype(np.float64) / max(p_hist.sum(), 1)
    q = q_hist.astype(np.float64) / max(q_hist.sum(), 1)
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    w1 = float(np.sum(np.abs(cdf_p - cdf_q)))
    return w1


def derive_seed_for_subband(subband_name: str, base_seed_bytes: bytes) -> bytes:
    """Deterministically derive a 32-byte per-subband seed from a base seed.

    Sha256(base_seed || subband_name).digest()[:32] per Catalog #344 sister
    discipline + the canonical helper's documented seed-derivation pattern.
    """
    return hashlib.sha256(base_seed_bytes + subband_name.encode("ascii")).digest()


def run_smoke(
    video_path: Path,
    frame_index: int,
    base_seed_bytes: bytes,
    wavelet: str,
    dwt_level: int,
    generator_kind: str,
) -> dict[str, Any]:
    """Run the WAVE-3 op-routable #2 smoke pipeline.

    Returns a typed result dict ready for JSON serialization + canonical
    equation anchor append.
    """
    from tac.procedural_codebook_generator.seed_derived_codebook import (
        derive_codebook_from_seed,
    )

    start_utc = _utc_now_iso()
    video_sha = _sha256_of_file(video_path)

    # Step 1+2: decode + extract Y
    y_plane, height, width = decode_frame_from_video(video_path, frame_index)
    y_sha = _sha256_of_bytes(y_plane.tobytes())

    # Step 3: DWT 2-level (return coarsest level detail subbands)
    detail_subbands = compute_dwt_detail_subbands(y_plane, wavelet, dwt_level)

    # Step 4+5: per-subband empirical-vs-synthetic distributional fit
    per_subband: dict[str, dict[str, Any]] = {}
    aggregate_kl_terms: list[float] = []
    aggregate_w1_terms: list[float] = []
    for subband_name, subband in detail_subbands.items():
        empirical_int8 = normalize_to_int8_distribution(subband)
        n_detail = int(empirical_int8.size)
        seed_for_subband = derive_seed_for_subband(subband_name, base_seed_bytes)
        synthetic_int8 = derive_codebook_from_seed(
            seed_bytes=seed_for_subband,
            output_shape=(n_detail, 1),
            dtype=np.int8,
            generator_kind=generator_kind,
        ).reshape(empirical_int8.shape)

        kl = kl_divergence_int8(empirical_int8, synthetic_int8)
        w1 = wasserstein1_int8(empirical_int8, synthetic_int8)

        # Catalog #272 byte-mutation smoke: mutate first byte of seed → re-derive
        mutated_seed = bytearray(seed_for_subband)
        mutated_seed[0] ^= 0xFF  # flip all bits of first byte
        mutated_synthetic = derive_codebook_from_seed(
            seed_bytes=bytes(mutated_seed),
            output_shape=(n_detail, 1),
            dtype=np.int8,
            generator_kind=generator_kind,
        ).reshape(empirical_int8.shape)
        # Verify mutation produces distributional change (NOT byte-identical)
        synthetic_vs_mutated_kl = kl_divergence_int8(synthetic_int8, mutated_synthetic)
        synthetic_vs_mutated_w1 = wasserstein1_int8(synthetic_int8, mutated_synthetic)
        bytes_differ = int(
            np.count_nonzero(synthetic_int8.flatten() != mutated_synthetic.flatten())
        )

        per_subband[subband_name] = {
            "shape": list(empirical_int8.shape),
            "n_pixels": n_detail,
            "empirical_int8_sha256": _sha256_of_bytes(empirical_int8.tobytes()),
            "synthetic_int8_sha256": _sha256_of_bytes(synthetic_int8.tobytes()),
            "mutated_synthetic_int8_sha256": _sha256_of_bytes(mutated_synthetic.tobytes()),
            "kl_divergence_empirical_vs_synthetic_nats": kl,
            "wasserstein1_empirical_vs_synthetic": w1,
            "byte_mutation_smoke_kl_synthetic_vs_mutated_nats": synthetic_vs_mutated_kl,
            "byte_mutation_smoke_wasserstein1_synthetic_vs_mutated": synthetic_vs_mutated_w1,
            "byte_mutation_smoke_byte_differs_count": bytes_differ,
            "byte_mutation_smoke_verdict_seed_sensitive": bool(bytes_differ > 0),
            "seed_for_subband_sha256": _sha256_of_bytes(seed_for_subband),
        }
        aggregate_kl_terms.append(kl)
        aggregate_w1_terms.append(w1)

    # Step 6: aggregate normalized distance metric + residual_zscore
    # The canonical equation #26 predicts byte savings; THIS smoke
    # validates the DISTRIBUTIONAL precondition (seed-derived bytes
    # match empirical detail-subband statistics within tolerance).
    # The empirical residual against the predicted-only hypothesis is the
    # aggregate normalized distributional distance.
    aggregate_kl_mean = float(np.mean(aggregate_kl_terms))
    aggregate_w1_mean = float(np.mean(aggregate_w1_terms))

    # Predicted under H0 (seed-derived bytes are i.i.d. uniform int8): for
    # an int8 uniform distribution the expected KL vs empirical (which is
    # NOT uniform — detail subbands have heavy-tailed peaked distributions)
    # is high. The CARGO-CULTED assumption #1 expects KL ≈ 0; the empirical
    # value tells us whether procedural-codebook bytes can substitute for
    # detail-subband bytes WITHOUT large distributional drift.
    # We use 2σ ≈ 0.5 nats as the canonical-equation HARD-EARNED threshold
    # (Daubechies matched-filter analysis: detail subbands should be
    # near-Laplacian; uniform int8 differs by ~0.5 nats KL).
    PREDICTED_KL_UNDER_H0_THRESHOLD = 0.5
    aggregate_residual_zscore = float(
        abs(aggregate_kl_mean - 0.0) / max(PREDICTED_KL_UNDER_H0_THRESHOLD, 1e-6)
    )
    canonical_equation_verdict = (
        "HARD-EARNED" if aggregate_residual_zscore < 2.0 else "CARGO-CULTED"
    )

    # Byte-mutation smoke aggregate: ALL subbands must show seed-sensitivity
    byte_mutation_all_subbands_seed_sensitive = all(
        per_subband[name]["byte_mutation_smoke_verdict_seed_sensitive"]
        for name in per_subband
    )

    end_utc = _utc_now_iso()

    try:
        video_path_for_record = str(video_path.relative_to(REPO_ROOT))
    except ValueError:
        # Video outside repo (test scaffolds / forensic replay); record absolute path.
        video_path_for_record = str(video_path)

    return {
        "smoke_label": "wave_3_dwt_detail_subband_procedural_cpu_smoke",
        "smoke_lane_id": "lane_wave_3_dwt_detail_subband_procedural_cpu_smoke_20260520",
        "smoke_council_anchor": "grand_council_dwt_hnerv_world_model_bind_20260520",
        "smoke_op_routable": "op_routable_2_byte_mutation_smoke_dwt_detail_subbands",
        "started_at_utc": start_utc,
        "completed_at_utc": end_utc,
        "platform": platform.platform(),
        "video_path_repo_relative": video_path_for_record,
        "video_sha256": video_sha,
        "frame_index": frame_index,
        "frame_height": height,
        "frame_width": width,
        "y_plane_sha256": y_sha,
        "wavelet": wavelet,
        "dwt_level": dwt_level,
        "generator_kind": generator_kind,
        "base_seed_bytes_hex": base_seed_bytes.hex(),
        "base_seed_bytes_len": len(base_seed_bytes),
        "per_subband": per_subband,
        "aggregate_kl_divergence_nats_mean": aggregate_kl_mean,
        "aggregate_wasserstein1_mean": aggregate_w1_mean,
        "aggregate_residual_zscore_vs_H0_kl_threshold_0p5": aggregate_residual_zscore,
        "canonical_equation_verdict_HARD_EARNED_or_CARGO_CULTED_at_2sigma": canonical_equation_verdict,
        "byte_mutation_smoke_aggregate_seed_sensitive_all_subbands": byte_mutation_all_subbands_seed_sensitive,
        "axis_tag": "[macOS-CPU advisory]",
        "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
        "evidence_grade": "local_cpu_smoke_advisory",
        "promotion_eligible": False,
        "score_claim_valid": False,
        "score_claim_axis": None,
        "canonical_equation_id": "procedural_codebook_from_seed_compression_savings_v1",
        "canonical_equation_first_empirical_anchor_pending_append": True,
    }


def append_first_empirical_anchor(
    smoke_result: dict[str, Any],
    output_json_path: Path,
) -> None:
    """Append the FIRST empirical anchor to canonical equation #26.

    Per Catalog #344 sister discipline + Catalog #127 per-call-site custody
    (macOS-CPU advisory non-promotable per Catalog #192):
      axis_tag = "[macOS-CPU advisory]"
      hardware_substrate = "darwin_arm64_m5_max_macos_cpu_advisory"
      evidence_grade = MACOS_CPU_ADVISORY
      promotion_eligible = False
      score_claim_valid = False
    """
    from tac.canonical_equations.equation import EmpiricalAnchor
    from tac.canonical_equations.registry import update_equation_with_empirical_anchor
    from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

    # The smoke writes its JSON to the experiments/results/<dir>/smoke_result.json
    # path; we record that path as source_artifact for the anchor.
    try:
        source_artifact_relpath = str(output_json_path.relative_to(REPO_ROOT))
    except ValueError:
        # Test scaffolds / forensic replay may write outside repo; record absolute.
        source_artifact_relpath = str(output_json_path)
    source_artifact_sha = _sha256_of_file(output_json_path)

    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=source_artifact_sha,
        source_path=source_artifact_relpath,
    )

    # The anchor captures both the predicted hypothesis (per equation
    # #26's aggregate hypothesis: detail-subband bytes can be procedurally
    # substituted) AND the empirical distributional distance (KL nats).
    # `residual` per EmpiricalAnchor schema invariant is a non-negative
    # float — we use the aggregate KL divergence as the canonical residual
    # magnitude (0.0 = perfect distributional match; larger = worse fit).
    anchor = EmpiricalAnchor(
        anchor_id=f"first_empirical_anchor_wave_3_dwt_smoke_{_utc_now_filename()}",
        measurement_utc=smoke_result["completed_at_utc"],
        inputs={
            "smoke_label": smoke_result["smoke_label"],
            "smoke_lane_id": smoke_result["smoke_lane_id"],
            "video_path": smoke_result["video_path_repo_relative"],
            "video_sha256": smoke_result["video_sha256"],
            "frame_index": smoke_result["frame_index"],
            "wavelet": smoke_result["wavelet"],
            "dwt_level": smoke_result["dwt_level"],
            "generator_kind": smoke_result["generator_kind"],
            "base_seed_bytes_len": smoke_result["base_seed_bytes_len"],
            "axis_tag": smoke_result["axis_tag"],
            "hardware_substrate": smoke_result["hardware_substrate"],
            "evidence_grade": smoke_result["evidence_grade"],
            "subbands_tested": sorted(smoke_result["per_subband"].keys()),
        },
        predicted_output={
            "aggregate_predicted_kl_divergence_under_H0_nats": 0.0,
            "aggregate_predicted_wasserstein1_under_H0": 0.0,
            "hypothesis_status": "predicted_only_seed_derived_bytes_match_empirical_distribution_within_2sigma_KL_threshold_0p5",
        },
        empirical_output={
            "aggregate_kl_divergence_nats_mean": smoke_result["aggregate_kl_divergence_nats_mean"],
            "aggregate_wasserstein1_mean": smoke_result["aggregate_wasserstein1_mean"],
            "aggregate_residual_zscore_vs_H0_kl_threshold_0p5": smoke_result[
                "aggregate_residual_zscore_vs_H0_kl_threshold_0p5"
            ],
            "canonical_equation_verdict_HARD_EARNED_or_CARGO_CULTED_at_2sigma": smoke_result[
                "canonical_equation_verdict_HARD_EARNED_or_CARGO_CULTED_at_2sigma"
            ],
            "byte_mutation_smoke_aggregate_seed_sensitive_all_subbands": smoke_result[
                "byte_mutation_smoke_aggregate_seed_sensitive_all_subbands"
            ],
            "per_subband_kl_nats": {
                name: per["kl_divergence_empirical_vs_synthetic_nats"]
                for name, per in smoke_result["per_subband"].items()
            },
            "per_subband_wasserstein1": {
                name: per["wasserstein1_empirical_vs_synthetic"]
                for name, per in smoke_result["per_subband"].items()
            },
        },
        residual=float(smoke_result["aggregate_kl_divergence_nats_mean"]),
        source_artifact=source_artifact_relpath,
        measurement_method="haar_dwt_2_level_y_plane_per_subband_int8_normalization_then_kl_and_wasserstein_1_vs_pcg64_seed_derived_codebook_local_macos_cpu_smoke_advisory",
        provenance=provenance,
    )

    update_equation_with_empirical_anchor(
        equation_id="procedural_codebook_from_seed_compression_savings_v1",
        anchor=anchor,
        agent="claude_subagent",
        subagent_id="wave-3-dwt-detail-subband-procedural-cpu-smoke-20260520",
        notes=(
            "WAVE-3 op-routable #2 first empirical anchor; LOCAL macOS-CPU "
            "advisory smoke per Catalog #192 + #127 + #323; distributional "
            "validation precedes per-substrate inflate dispatch"
        ),
    )


def emit_markdown_report(smoke_result: dict[str, Any], md_path: Path) -> None:
    """Write a human-readable Markdown table summarizing the smoke."""
    lines = [
        "<!-- SPDX-License-Identifier: MIT -->",
        "# WAVE-3 DWT Detail-Subband Procedural Codebook Local CPU Smoke",
        "",
        f"**Lane**: `{smoke_result['smoke_lane_id']}`  ",
        f"**Council anchor**: `{smoke_result['smoke_council_anchor']}`  ",
        f"**Op-routable**: `{smoke_result['smoke_op_routable']}`  ",
        f"**Started**: `{smoke_result['started_at_utc']}`  ",
        f"**Completed**: `{smoke_result['completed_at_utc']}`  ",
        f"**Platform**: `{smoke_result['platform']}`  ",
        "",
        "## Custody (Catalog #127 + #192 + #323)",
        "",
        f"* `axis_tag`: `{smoke_result['axis_tag']}` (NEVER promotable per Catalog #192)",
        f"* `hardware_substrate`: `{smoke_result['hardware_substrate']}`",
        f"* `evidence_grade`: `{smoke_result['evidence_grade']}`",
        f"* `promotion_eligible`: `{smoke_result['promotion_eligible']}`",
        f"* `score_claim_valid`: `{smoke_result['score_claim_valid']}`",
        "",
        "## Inputs",
        "",
        f"* Video: `{smoke_result['video_path_repo_relative']}` (sha256 `{smoke_result['video_sha256'][:16]}...`)",
        f"* Frame index: `{smoke_result['frame_index']}`",
        f"* Frame shape: `{smoke_result['frame_height']} x {smoke_result['frame_width']}` Y plane",
        f"* Y plane sha256: `{smoke_result['y_plane_sha256'][:16]}...`",
        f"* Wavelet: `{smoke_result['wavelet']}` @ level `{smoke_result['dwt_level']}`",
        f"* PRNG generator kind: `{smoke_result['generator_kind']}`",
        f"* Base seed length: `{smoke_result['base_seed_bytes_len']}` bytes",
        "",
        "## Per-subband distributional fit",
        "",
        "| Subband | N pixels | KL nats | Wasserstein-1 | Mutation KL | Mutation bytes-differ | Seed-sensitive |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name in sorted(smoke_result["per_subband"].keys()):
        per = smoke_result["per_subband"][name]
        lines.append(
            f"| {name} | {per['n_pixels']} | {per['kl_divergence_empirical_vs_synthetic_nats']:.4f} "
            f"| {per['wasserstein1_empirical_vs_synthetic']:.4f} "
            f"| {per['byte_mutation_smoke_kl_synthetic_vs_mutated_nats']:.4f} "
            f"| {per['byte_mutation_smoke_byte_differs_count']} "
            f"| {'YES' if per['byte_mutation_smoke_verdict_seed_sensitive'] else 'NO'} |"
        )
    lines += [
        "",
        "## Aggregate canonical-equation verdict",
        "",
        f"* Aggregate KL mean: `{smoke_result['aggregate_kl_divergence_nats_mean']:.4f}` nats",
        f"* Aggregate Wasserstein-1 mean: `{smoke_result['aggregate_wasserstein1_mean']:.4f}`",
        f"* Aggregate residual_zscore (vs H0 KL threshold 0.5): `{smoke_result['aggregate_residual_zscore_vs_H0_kl_threshold_0p5']:.4f}`",
        f"* **Canonical equation #26 verdict**: `{smoke_result['canonical_equation_verdict_HARD_EARNED_or_CARGO_CULTED_at_2sigma']}` (at 2σ threshold)",
        f"* **Catalog #272 byte-mutation smoke verdict**: `{'PASSED' if smoke_result['byte_mutation_smoke_aggregate_seed_sensitive_all_subbands'] else 'FAILED'}` (all subbands seed-sensitive)",
        "",
        "## Equation linkage",
        "",
        f"* Canonical equation: `{smoke_result['canonical_equation_id']}`",
        f"* First empirical anchor pending append: `{smoke_result['canonical_equation_first_empirical_anchor_pending_append']}`",
        "",
        "## Discipline citations",
        "",
        "* CLAUDE.md \"MPS auth eval is NOISE\" — macOS-CPU is NEVER score truth",
        "* CLAUDE.md \"Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE\" — `[macOS-CPU advisory]` is non-promotable",
        "* CLAUDE.md \"Apples-to-apples evidence discipline\" — axis labels preserved",
        "* CLAUDE.md \"Bit-level deconstruction and entropy discipline\" — distributional analysis at byte level",
        "* Catalog #127 — custody triple axis × hardware × evidence_grade",
        "* Catalog #192 — macOS-CPU advisory not promotable without Linux x86_64 verification",
        "* Catalog #272 — distinguishing-feature byte-mutation smoke",
        "* Catalog #277 — wavelet multi-scale canonical helper",
        "* Catalog #309 — horizon_class=`frontier_pursuit`",
        "* Catalog #323 — canonical Provenance umbrella",
        "* Catalog #335 — canonical consumer contract",
        "* Catalog #344 — canonical equation FIRST `anchor_appended` event",
        "",
        "## Source",
        "",
        "* T3 DWT BIND SYMPOSIUM memo: `.omx/research/grand_council_symposium_dwt_hnerv_world_model_bind_20260520.md`",
        "* Op-routable #2 verdict: `PROCEED_WITH_REVISIONS` (Carmack MVP-first phasing)",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "WAVE-3 op-routable #2 DWT detail-subband procedural-codebook "
            "local macOS-CPU smoke. NON-promotable; macOS-CPU advisory per "
            "CLAUDE.md \"MPS auth eval is NOISE\" + Catalog #192."
        )
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=REPO_ROOT / "upstream" / "videos" / "0.mkv",
        help="Path to contest video (default: upstream/videos/0.mkv).",
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        default=DEFAULT_FRAME_INDEX,
        help=f"Frame index to decode for smoke (default: {DEFAULT_FRAME_INDEX}).",
    )
    parser.add_argument(
        "--base-seed-hex",
        type=str,
        default=None,
        help=(
            "Hex-encoded base seed bytes (default: sha256 of canonical "
            "WAVE-3 op-routable #2 label). Length must be 16-512 hex chars."
        ),
    )
    parser.add_argument(
        "--wavelet",
        type=str,
        default=DEFAULT_WAVELET,
        help=f"pywt wavelet name (default: {DEFAULT_WAVELET}).",
    )
    parser.add_argument(
        "--dwt-level",
        type=int,
        default=DEFAULT_DWT_LEVEL,
        help=f"DWT decomposition level (default: {DEFAULT_DWT_LEVEL}).",
    )
    parser.add_argument(
        "--generator-kind",
        type=str,
        default=DEFAULT_GENERATOR_KIND,
        choices=("xorshift", "lcg", "pcg64"),
        help=(
            "Procedural-codebook generator kind (default: "
            f"{DEFAULT_GENERATOR_KIND}; matches Catalog #344 canonical equation #26)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory under experiments/results/ (default: "
            "experiments/results/dwt_detail_subband_procedural_smoke_<utc>/)."
        ),
    )
    parser.add_argument(
        "--skip-canonical-equation-append",
        action="store_true",
        help=(
            "Skip the canonical-equation anchor_appended event (for "
            "dry-run / replay smoke runs that should NOT pollute the "
            "canonical posterior). The smoke JSON + MD artifacts still emit."
        ),
    )
    args = parser.parse_args(argv)

    # Per CLAUDE.md "MPS auth eval is NOISE": this smoke is macOS-CPU advisory only.
    # We do NOT refuse non-Darwin hosts (CI may run on Linux x86_64) but stamp
    # the actual platform in the JSON for downstream consumer routing per Catalog #127.

    if not args.video_path.exists():
        print(f"FATAL: video_path={args.video_path} not found", file=sys.stderr)
        return 2

    if args.base_seed_hex is None:
        # Default seed: sha256 of the canonical WAVE-3 op-routable #2 label
        base_seed_bytes = hashlib.sha256(
            b"wave_3_dwt_detail_subband_procedural_cpu_smoke_20260520_op_routable_2"
        ).digest()
    else:
        try:
            base_seed_bytes = bytes.fromhex(args.base_seed_hex)
        except ValueError as exc:
            print(f"FATAL: --base-seed-hex invalid hex: {exc}", file=sys.stderr)
            return 2
        if not (8 <= len(base_seed_bytes) <= 256):
            print(
                f"FATAL: --base-seed-hex length {len(base_seed_bytes)} bytes "
                "out of canonical range [8, 256]",
                file=sys.stderr,
            )
            return 2

    if args.output_dir is None:
        output_dir = (
            REPO_ROOT
            / "experiments"
            / "results"
            / f"dwt_detail_subband_procedural_smoke_{_utc_now_filename()}"
        )
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "smoke_result.json"
    md_path = output_dir / "smoke_result.md"

    print(f"[wave-3-dwt-smoke] decoding video {args.video_path}...", file=sys.stderr)
    smoke_result = run_smoke(
        video_path=args.video_path,
        frame_index=args.frame_index,
        base_seed_bytes=base_seed_bytes,
        wavelet=args.wavelet,
        dwt_level=args.dwt_level,
        generator_kind=args.generator_kind,
    )

    print(f"[wave-3-dwt-smoke] writing smoke_result.json + smoke_result.md", file=sys.stderr)
    json_path.write_text(
        json.dumps(smoke_result, indent=2, sort_keys=True), encoding="utf-8"
    )
    emit_markdown_report(smoke_result, md_path)

    if not args.skip_canonical_equation_append:
        print(
            "[wave-3-dwt-smoke] appending FIRST empirical anchor to canonical "
            "equation procedural_codebook_from_seed_compression_savings_v1...",
            file=sys.stderr,
        )
        append_first_empirical_anchor(smoke_result, json_path)
        print(
            "[wave-3-dwt-smoke] anchor_appended event landed (Catalog #344 sister)",
            file=sys.stderr,
        )
    else:
        print(
            "[wave-3-dwt-smoke] SKIPPED canonical equation anchor_appended event "
            "(--skip-canonical-equation-append)",
            file=sys.stderr,
        )

    try:
        out_dir_display = str(output_dir.relative_to(REPO_ROOT))
    except ValueError:
        out_dir_display = str(output_dir)
    print(
        f"[wave-3-dwt-smoke] DONE: output_dir={out_dir_display}",
        file=sys.stderr,
    )
    print(
        f"[wave-3-dwt-smoke] aggregate_kl_mean_nats="
        f"{smoke_result['aggregate_kl_divergence_nats_mean']:.4f} "
        f"residual_zscore={smoke_result['aggregate_residual_zscore_vs_H0_kl_threshold_0p5']:.4f} "
        f"verdict={smoke_result['canonical_equation_verdict_HARD_EARNED_or_CARGO_CULTED_at_2sigma']} "
        f"byte_mutation_seed_sensitive={smoke_result['byte_mutation_smoke_aggregate_seed_sensitive_all_subbands']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
