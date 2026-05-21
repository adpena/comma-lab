# SPDX-License-Identifier: MIT
"""tools/run_magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke.py — WAVE-3 pair #2.

LOCAL macOS-CPU smoke validating PAIR #2 architecture from the
MAGIC CODEC × TODAY'S CASCADE STACKING ANALYSIS memo
``.omx/research/magic_codec_x_todays_cascade_stacking_analysis_20260520.md``:

    sparse_packet_ir SRL1 × procedural-codebook null-byte residuals
        applied to fec6 frontier 16,292 null-byte positions
        (NOT direct substitution per pair #1 empirical falsification)

Yesterday's MAGIC CODEC × DWT DETAIL-SUBBAND CPU SMOKE (pair #1; landing
commit ``debbc5833``; SECOND empirical anchor for canonical equation #26)
empirically FALSIFIED the (pcg64-uniform predictor, DWT detail, int8) tuple:
residuals between empirical detail-subband bytes and pcg64-seed-derived
uniform bytes are themselves near-uniform after int8 clipping (because
pcg64 is uniform AND empirical is Laplacian-peaked, so empirical −
uniform = empirical shifted-and-mixed → near-uniform). Net effect:
+0.036805 rate-term INCREASE; residual_zscore = 38.8 > 2σ → CARGO-CULTED.

Pair #1 FALSIFIED → CASCADE PIVOTS to pair #2: instead of DWT detail
subbands (continuous Laplacian-peaked distribution that brotli already
compresses optimally), target the **fec6 frontier master-gradient-null
positions** (16,292 / 178,417 = 9.13% of inner-member bytes per
``feedback_null_byte_probe_matrix_landed_20260520.md``). "Null" here
means zero score-gradient leverage across seg/pose/rate axes, NOT
literal byte value zero. The hypothesis under test: a deterministic
procedural predictor may make residuals on this selected leverage-null
index set sparse enough that SRL1 (RLE-of-zeros) beats storing the
empirical bytes in place.

3-way apples-to-apples comparison (the canonical measurement structure):

    Configuration A (direct empirical):
        keep the 16,292 master-gradient-null member bytes in place.
        Primary contest-byte baseline: 16,292 charged bytes.
        Secondary audit baseline: brotli(q=11) over only those selected
        empirical bytes.

    Configuration B (procedural only — yesterday's REFUTED hypothesis at sister surface):
        32 B seed substitutes for the 16,292 gradient-null byte values via
        derive_codebook_from_seed → synthetic-int8 bytes stored INSTEAD
        of the empirical bytes
        rate-term cost: 25 * 32 / 37_545_489 (one seed for all null positions)
        (would corrupt the archive unless synthetic exactly matches the
         original member bytes; pure procedural substitution is lossy at inflate)

    Configuration C (procedural + SRL1 residuals — pair #2 candidate):
        one 32 B seed for all selected null-leverage positions (predictor)
        + SRL1-encoded residual (empirical_uint8 - synthetic_uint8) at the
          16,292 gradient-null positions; residual is exact int16 so
          synthetic + residual reconstructs the original byte value.
        rate-term cost: 25 * (32 + serialize_rle_of_zeros(residual)) / 37_545_489

The pair #2 prediction (from stacking analysis memo §7):
    composition_alpha=0.9 ADDITIVE
    predicted ΔS = -0.00109 ([-0.011, -0.005] cumulative under 4-pair stack)

Verdict at 2σ threshold (per the stacking analysis memo §8 Dykstra-
feasibility): HARD-EARNED if residual_zscore < 2σ (rescue path validated;
the cascade can proceed to pair #3); CARGO-CULTED if residual_zscore > 2σ
(rescue path falsified; cascade further narrows or pivots to pair #4
orthogonality validation OR DP1-only).

Pipeline:

1. Locate fec6 frontier archive via canonical_frontier_pointer.json
   (Catalog #343 pointer-only) → resolve to
   experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip
   (sha 6bae0201fb08...; 178517 bytes; lane pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515)
2. Load gradient-null byte indices from the null_byte_probe_matrix output at
   experiments/results/null_byte_probe_matrix_20260520T223742Z/null_byte_matrix.json
   (per_anchor[anchor_index=5].n_null_bytes=16292; section breakdown
   OUTER_MAGIC[0:4] / selector_len_hdr[178166:178168] / selector_payload[178168:178417] /
   source_len_hdr / many other zero positions)
3. Extract the single stored ZIP member bytes (178,417 bytes) and use the
   master-gradient tensor to identify byte positions whose seg/pose/rate
   gradients are all zero. "Null byte" means scorer-null gradient, not
   literal byte value 0.
4. For Configuration A: primary baseline = 16,292 in-place charged bytes;
   secondary audit = brotli(q=11) over the 16,292 selected empirical bytes.
5. For Configuration B: 32 B seed (lossy substitution placeholder)
6. For Configuration C:
   a. Derive 16,292 synthetic int8 bytes via derive_codebook_from_seed
      (pcg64 default; sister of pair #1 generator_kind)
   b. Compute exact residual = empirical_uint8 - synthetic_uint8 as int16
   c. Encode residual via encode_rle_of_zeros + serialize_rle_of_zeros
   d. total = 32B seed + len(serialize_rle_of_zeros(residual))
7. Catalog #272 byte-mutation smoke: mutate 1 byte of seed → re-derive →
   re-compute residual → re-encode → verify SRL1 encoded bytes change
   (proves residual encoding is seed-sensitive; rules out the empty-byte
   / no-op trap from Catalog #220/#249)
8. Compute aggregate residual_zscore (NEW IN-DOMAIN context per Catalog #344):
   bytes_saved_residual_correction = baseline_bytes - (32 + srl1_residual_bytes)
9. Compare predicted vs empirical ΔS:
   predicted_ΔS = -0.00109 (per pair #2 ADDITIVE α=0.9 prediction)
   empirical_ΔS = -25 * bytes_saved / 37_545_489
10. Append THIRD empirical anchor to canonical equation #26 via
    tac.canonical_equations.update_equation_with_empirical_anchor
    with NEW IN-DOMAIN context
    sparse_packet_ir_srl1_correction_on_fec6_frontier_null_bytes
    (axis=[macOS-CPU advisory]; hardware=darwin_arm64_m5_max_macos_cpu_advisory;
    evidence_grade=MACOS_CPU_ADVISORY; promotion_eligible=False;
    score_claim_valid=False per Catalog #192)
11. Emit JSON + Markdown artifacts under
    experiments/results/magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke_<utc>/

Note: this smoke does NOT run inflate.sh / contest_auth_eval.py — it
measures the byte-budget difference between Configurations A/B/C as the
THIRD empirical anchor for canonical equation #26 with a NEW IN-DOMAIN
context distinct from the FIRST anchor (yesterday's DWT distributional
fit under direct substitution H0) and the SECOND anchor (pair #1
magic_codec_dense_streams residual on DWT detail subbands).

Sister-DISJOINT from:

* DP1 PAIRED-SMOKE DISPATCH PRE-AUTHORIZATION CHECKLIST (`a13d467e`) —
  different substrate; different file path; DP1 designs paired-smoke
  pre-authorization for downstream DP1 dispatch; THIS smoke validates
  pair #2 of the magic_codec cascade-stacking 4-pair matrix
* WAVE-3 END-OF-DAY CASCADE RECONCILIATION (`a89a1cd7`) — different
  file path; end-of-day cascade synthesis vs THIS being one cascade
  rung
* `debbc5833` (pair #1 DWT residual smoke) — sister-COMPLEMENTARY:
  pair #1 SECOND anchor measured DWT-detail-subband byte budget;
  THIS pair #2 THIRD anchor measures fec6 frontier null-byte budget
  (distinct distributions; distinct in-domain context)
* `82c1b3bac` (NULL-BYTE PROBE MATRIX) — sister-COMPLEMENTARY:
  matrix identified WHERE the null bytes are; THIS smoke validates
  whether the SRL1 residual encoding WINS on them

Per CLAUDE.md "MPS auth eval is NOISE" + Catalog #192 (macOS-CPU advisory
not promotable without Linux x86_64 verification) + Catalog #127 (per-
call-site custody triple axis × hardware × evidence_grade): every metric
stamped ``[macOS-CPU advisory]`` + ``hardware_substrate=darwin_arm64_m5_max_macos_cpu_advisory``
+ ``evidence_grade=local_cpu_smoke_advisory``. NOT score truth.

Per Carmack MVP-first phasing + cumulative cascade savings ($3-10 paid
GPU spend saved today via $0 CPU smokes): this smoke validates pair #2
prediction BEFORE any paid GPU investment in magic-codec stacking cascade.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import brotli
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


# Canonical constants per CLAUDE.md "Submission auth eval" + canonical equation #26
CANONICAL_RATE_MULTIPLIER = 25.0
CANONICAL_RATE_DENOM_BYTES = 37_545_489
DEFAULT_SEED_BYTES = 32
DEFAULT_GENERATOR_KIND = "pcg64"

# Pair #2 prediction from MAGIC CODEC × TODAY'S CASCADE STACKING ANALYSIS memo §7
# Composition alpha = 0.9 ADDITIVE; predicted ΔS = -0.00109
PAIR_2_PREDICTED_DELTA_S = -0.00109
PAIR_2_COMPOSITION_ALPHA_ESTIMATE = 0.9

# 2σ threshold for residual_zscore HARD-EARNED vs CARGO-CULTED verdict
# (sister of pair #1; for pair #2 the H0 is "rescue path produces no net
#  byte savings on the master-gradient-null index set")
# 2σ threshold = 0.5 * |predicted ΔS| (empirical within ±0.5x of predicted = HARD-EARNED)
PAIR_2_ZSCORE_HARD_EARNED_THRESHOLD = 2.0

# Brotli baseline canonical parameters (must match
# magic_codec_dense_streams._BROTLI_QUALITY / _BROTLI_LGWIN)
BROTLI_QUALITY = 11
BROTLI_LGWIN = 22

# fec6 frontier archive canonical sha (canonical_frontier_pointer.json)
FEC6_FRONTIER_SHA256_PREFIX = "6bae0201fb08"
FEC6_FRONTIER_LANE_ID = (
    "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515"
)
DEFAULT_FEC6_ARCHIVE_PATH = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex"
    / "archive.zip"
)
DEFAULT_NULL_BYTE_MATRIX_PATH = (
    REPO_ROOT
    / "experiments"
    / "results"
    / "null_byte_probe_matrix_20260520T223742Z"
    / "null_byte_matrix.json"
)


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


def load_fec6_frontier_archive_bytes(archive_path: Path) -> tuple[bytes, str]:
    """Load fec6 frontier archive bytes + verify sha256 prefix.

    Per Catalog #343 canonical pointer discipline: the fec6 frontier
    archive is identified by its sha256 prefix 6bae0201fb08 from
    .omx/state/canonical_frontier_pointer.json. We verify the prefix here
    rather than hard-codding the full sha so a future frontier pointer
    update can flow through without breaking the smoke.

    Returns (archive_bytes, archive_sha256).
    """
    if not archive_path.exists():
        raise FileNotFoundError(
            f"fec6 frontier archive not found at {archive_path}"
        )
    archive_bytes = archive_path.read_bytes()
    archive_sha = hashlib.sha256(archive_bytes).hexdigest()
    if not archive_sha.startswith(FEC6_FRONTIER_SHA256_PREFIX):
        raise RuntimeError(
            f"archive sha {archive_sha[:16]}... does not match expected "
            f"fec6 frontier prefix {FEC6_FRONTIER_SHA256_PREFIX}..."
        )
    return archive_bytes, archive_sha


def load_fec6_null_byte_anchor_metadata(
    matrix_path: Path,
) -> dict[str, Any]:
    """Load the fec6 frontier per_anchor row metadata from the null_byte_matrix.

    Per ``feedback_null_byte_probe_matrix_landed_20260520.md`` the matrix
    at ``experiments/results/null_byte_probe_matrix_*/null_byte_matrix.json``
    contains per_anchor rows with master_gradient .npy paths. The
    canonical "null byte" surface is the master_gradient tensor: a byte
    index is "null" iff all 3 axes (seg, pose, rate) have zero gradient
    at that position. The fec6 frontier anchor has 16,292 such null byte
    positions out of 178,417 inner-member bytes (9.13%).
    """
    if not matrix_path.exists():
        raise FileNotFoundError(
            f"null_byte_matrix not found at {matrix_path}"
        )
    payload = json.loads(matrix_path.read_text())
    matching_anchors = [
        a
        for a in payload.get("per_anchor", [])
        if a.get("archive_sha256", "").startswith(FEC6_FRONTIER_SHA256_PREFIX)
    ]
    if not matching_anchors:
        raise RuntimeError(
            f"no per_anchor row in {matrix_path} matches fec6 frontier prefix "
            f"{FEC6_FRONTIER_SHA256_PREFIX}; available shas: "
            f"{[a.get('archive_sha256', '')[:12] for a in payload.get('per_anchor', [])]}"
        )
    # Prefer the [macOS-CPU advisory] axis anchor for apples-to-apples with
    # the smoke's hardware substrate (cross-hardware drift is 0.0000pp per
    # the matrix landing memo so any axis is interchangeable in practice).
    macos_anchors = [
        a for a in matching_anchors if a.get("axis") == "[macOS-CPU advisory]"
    ]
    return macos_anchors[0] if macos_anchors else matching_anchors[0]


def find_master_gradient_null_byte_indices(
    master_gradient_npy_path: Path,
) -> np.ndarray:
    """Find byte indices where ALL 3 master-gradient axes are zero.

    The canonical "null byte" surface per Catalog #344 canonical equation
    `master_gradient_null_space_byte_fraction_v1`: a byte index is "null"
    iff ALL 3 axes (seg, pose, rate) of the master gradient have zero
    gradient at that position (i.e., the byte has zero leverage on score
    along EVERY axis). For the fec6 frontier this yields 16,292 indices
    (9.13% null fraction) per the null_byte_probe_matrix.

    Returns sorted uint64 array of null byte indices.
    """
    if not master_gradient_npy_path.exists():
        raise FileNotFoundError(
            f"master_gradient npy not found at {master_gradient_npy_path}"
        )
    grad = np.load(master_gradient_npy_path)
    if grad.ndim != 2 or grad.shape[1] != 3:
        raise RuntimeError(
            f"master_gradient shape {grad.shape} != expected (N, 3) "
            f"per canonical (seg, pose, rate) layout"
        )
    null_mask = np.all(grad == 0, axis=1)
    indices = np.flatnonzero(null_mask).astype(np.uint64, copy=False)
    return indices


def extract_inner_member_bytes_from_archive(archive_path: Path) -> bytes:
    """Extract the (single) inner member bytes from the fec6 frontier archive.

    The fec6 frontier archive has exactly one member ``x`` with the brotli
    + huffman + selector-encoded payload bytes (178,417 bytes). The master-
    gradient indices reference positions in THIS inner-member byte array
    (NOT the outer archive.zip bytes).
    """
    import zipfile

    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        if len(names) != 1:
            raise RuntimeError(
                f"fec6 frontier archive expected single member; got {names}"
            )
        with zf.open(names[0]) as f:
            return f.read()


def encode_config_a_baseline_brotli(empirical_bytes: np.ndarray) -> int:
    """Configuration A: direct empirical brotli baseline byte count.

    Encodes the selected empirical bytes via brotli(q=11) — the SAME
    parameters used by encode_magic_codec_dense_streams._try_brotli for
    apples-to-apples comparison with pair #1. This is a secondary audit
    baseline; the primary contest-byte baseline is the in-place count of
    gradient-null member bytes.
    """
    raw_bytes = empirical_bytes.tobytes()
    encoded = brotli.compress(raw_bytes, quality=BROTLI_QUALITY, lgwin=BROTLI_LGWIN)
    return len(encoded)


def compute_residual_int16(
    empirical_uint8: np.ndarray, synthetic_uint8: np.ndarray
) -> np.ndarray:
    """Compute exact ``empirical - synthetic`` byte residual as int16.

    The pair #2 smoke must be byte-exact: the decoder-side reconstruction
    equation is ``synthetic_uint8.astype(int16) + residual_int16 ==
    empirical_uint8.astype(int16)``. Clipping to int8 would silently lose
    information for byte values whose residual is outside [-128, 127].
    """
    if empirical_uint8.dtype != np.uint8 or synthetic_uint8.dtype != np.uint8:
        raise ValueError("residual inputs must be uint8")
    if empirical_uint8.shape != synthetic_uint8.shape:
        raise ValueError(
            f"shape mismatch: empirical {empirical_uint8.shape} vs "
            f"synthetic {synthetic_uint8.shape}"
        )
    return empirical_uint8.astype(np.int16) - synthetic_uint8.astype(np.int16)


def encode_config_c_procedural_plus_srl1(
    residual: np.ndarray,
    seed_bytes: bytes,
) -> tuple[int, dict[str, Any]]:
    """Configuration C: 32B seed + sparse_packet_ir SRL1 residual byte count.

    Encodes the residual via ``tac.packet_compiler.sparse_packet_ir.encode_rle_of_zeros``
    (SRL1 = RLE-of-zeros = the canonical sparse residual primitive #1
    per the module docstring; "Best for: dense-grid residuals that are
    uniformly sparse"). Returns (total_bytes, selection_log) where:

      total_bytes = len(seed_bytes) + len(serialize_rle_of_zeros(residual))

    The selection_log captures per-codec encoded byte counts so the
    landing memo can document the SRL1 wire-format efficiency and the
    sparsity_ratio.
    """
    from tac.packet_compiler.sparse_packet_ir import (
        encode_rle_of_zeros,
        serialize_rle_of_zeros,
    )

    # Encode the residual via SRL1
    stream = encode_rle_of_zeros(residual)
    serialized = serialize_rle_of_zeros(stream)

    total_bytes = len(seed_bytes) + len(serialized)
    value_itemsize = int(stream.nonzero_values.dtype.itemsize)
    selection_log = {
        "seed_bytes_len": len(seed_bytes),
        "srl1_serialized_payload_len": len(serialized),
        "srl1_serialized_payload_sha256": _sha256_of_bytes(serialized),
        "total_config_c_bytes": total_bytes,
        "n_nonzero_residual_entries": int(stream.nonzero_indices.size),
        "n_zero_residual_entries": int(
            stream.total_length - stream.nonzero_indices.size
        ),
        "residual_sparsity_ratio": float(stream.sparsity_ratio),
        "srl1_envelope_overhead_bytes": int(13),  # magic(4) + total_length(4) + n_nonzero(4) + dtype_code(1)
        "srl1_indices_bytes": int(stream.nonzero_indices.size * 4),
        "srl1_value_dtype": str(stream.nonzero_values.dtype),
        "srl1_value_itemsize": value_itemsize,
        "srl1_values_bytes": int(stream.nonzero_values.size * value_itemsize),
        "codec_name": "sparse_packet_ir_srl1",
        "codec_id": "SRL1",
    }
    return total_bytes, selection_log


def encode_config_c_alt_brotli_residual(residual: np.ndarray) -> int:
    """Sister comparison: brotli(q=11) on the residual for orthogonality check.

    The pair #2 hypothesis is that a procedural predictor will make this
    selected master-gradient-null residual sparse enough for SRL1 to win.
    This alternative shows what brotli would have charged on the same
    residual so the operator can audit whether SRL1's overhead is justified.
    """
    encoded = brotli.compress(residual.tobytes(), quality=BROTLI_QUALITY, lgwin=BROTLI_LGWIN)
    return len(encoded)


def run_smoke(
    archive_path: Path,
    matrix_path: Path,
    base_seed_bytes: bytes,
    generator_kind: str,
    max_null_indices: int | None = None,
) -> dict[str, Any]:
    """Run the WAVE-3 pair #2 smoke pipeline.

    Returns a typed result dict ready for JSON serialization + canonical
    equation THIRD anchor append.
    """
    from tac.procedural_codebook_generator.seed_derived_codebook import (
        derive_codebook_from_seed,
    )

    start_utc = _utc_now_iso()

    # Step 1: load fec6 frontier archive + verify outer ZIP sha.
    archive_bytes, archive_sha = load_fec6_frontier_archive_bytes(archive_path)
    archive_size = len(archive_bytes)

    # Step 2: extract inner-member bytes (178,417 bytes for fec6 frontier)
    # The master-gradient indices reference positions in this inner-member
    # byte array, NOT the outer archive.zip raw bytes.
    inner_bytes = extract_inner_member_bytes_from_archive(archive_path)
    inner_size = len(inner_bytes)

    # Step 3: load null-byte matrix metadata for the fec6 frontier anchor
    matrix_anchor = load_fec6_null_byte_anchor_metadata(matrix_path)

    # Step 4: locate the master_gradient npy and find ALL-3-axes-zero indices
    # (the canonical "null byte" surface per Catalog #344 canonical equation
    # master_gradient_null_space_byte_fraction_v1)
    master_gradient_npy_path = Path(matrix_anchor["npy_path"])
    if not master_gradient_npy_path.is_absolute():
        master_gradient_npy_path = REPO_ROOT / master_gradient_npy_path
    null_indices = find_master_gradient_null_byte_indices(master_gradient_npy_path)
    n_null_observed = int(null_indices.size)
    n_null_expected = int(matrix_anchor.get("n_null_bytes", 0))
    if n_null_observed != n_null_expected:
        raise RuntimeError(
            f"master-gradient null count mismatch: observed={n_null_observed} "
            f"expected={n_null_expected} from {matrix_path}"
        )
    n_total_expected = int(matrix_anchor.get("n_total_bytes", 0))
    if n_total_expected and n_total_expected != inner_size:
        raise RuntimeError(
            f"inner member size {inner_size} != matrix n_total_bytes {n_total_expected}"
        )

    # Defensive: scope to the canonical null-byte set; cap to first
    # max_null_indices if specified for fast-smoke mode.
    if max_null_indices is not None:
        null_indices = null_indices[:max_null_indices]
    n_null_used = int(null_indices.size)

    # Sanity: master_gradient null index range must fit within inner_bytes
    if n_null_used > 0 and int(null_indices.max()) >= inner_size:
        raise RuntimeError(
            f"master_gradient null index {int(null_indices.max())} >= "
            f"inner_bytes size {inner_size}"
        )

    # Extract empirical bytes at the master-gradient null positions
    # (these are the actual archive inner-member bytes at zero-leverage
    # positions; their VALUES are arbitrary — what's "null" about them is
    # their gradient leverage, not their byte value)
    arr_u8 = np.frombuffer(inner_bytes, dtype=np.uint8)
    empirical_u8 = arr_u8[null_indices]
    empirical_null_value_zero_fraction = float(np.mean(empirical_u8 == 0))

    # Configuration A: primary contest-byte baseline is the in-place byte
    # count. Secondary audit baseline is brotli over only the selected
    # empirical bytes.
    config_a_in_place_bytes = n_null_used
    config_a_brotli_bytes = encode_config_a_baseline_brotli(empirical_u8)

    # Configuration B: procedural only (32B seed; would corrupt the archive
    # unless the generated bytes exactly matched the selected empirical bytes).
    config_b_bytes = len(base_seed_bytes)

    # Configuration C: procedural + SRL1 residuals (pair #2)
    synthetic_u8 = derive_codebook_from_seed(
        seed_bytes=base_seed_bytes,
        output_shape=(n_null_used,),
        dtype=np.uint8,
        generator_kind=generator_kind,
    )
    residual_int16 = compute_residual_int16(empirical_u8, synthetic_u8)
    reconstructed = synthetic_u8.astype(np.int16) + residual_int16
    if not np.array_equal(reconstructed, empirical_u8.astype(np.int16)):
        raise RuntimeError("procedural + residual reconstruction is not byte-exact")
    config_c_bytes, config_c_selection_log = encode_config_c_procedural_plus_srl1(
        residual=residual_int16,
        seed_bytes=base_seed_bytes,
    )

    # Sister comparison: brotli on the residual (NOT pair #2 candidate;
    # just for orthogonality audit — shows whether SRL1's overhead is
    # justified by sparsity exploitation)
    config_c_alt_brotli_bytes = encode_config_c_alt_brotli_residual(residual_int16)

    # Catalog #272 byte-mutation smoke: mutate first byte of seed →
    # re-derive synthetic → re-compute residual → re-encode SRL1 → verify
    # encoded bytes change (proves the residual encoding is seed-sensitive)
    mutated_seed = bytearray(base_seed_bytes)
    mutated_seed[0] ^= 0xFF
    mutated_synthetic = derive_codebook_from_seed(
        seed_bytes=bytes(mutated_seed),
        output_shape=(n_null_used,),
        dtype=np.uint8,
        generator_kind=generator_kind,
    )
    mutated_residual = compute_residual_int16(empirical_u8, mutated_synthetic)
    mutated_config_c_bytes, mutated_log = encode_config_c_procedural_plus_srl1(
        residual=mutated_residual,
        seed_bytes=bytes(mutated_seed),
    )
    byte_diff_count = int(
        np.count_nonzero(residual_int16 != mutated_residual)
    )
    byte_mutation_seed_sensitive = bool(
        byte_diff_count > 0
        and config_c_selection_log["srl1_serialized_payload_sha256"]
        != mutated_log["srl1_serialized_payload_sha256"]
    )

    # Step 6: aggregate bytes saved + empirical ΔS
    bytes_saved_c_vs_a = config_a_in_place_bytes - config_c_bytes
    bytes_saved_c_vs_a_brotli_audit = config_a_brotli_bytes - config_c_bytes
    empirical_delta_s = (
        -CANONICAL_RATE_MULTIPLIER * bytes_saved_c_vs_a / CANONICAL_RATE_DENOM_BYTES
    )

    # Step 7: residual_zscore vs predicted ΔS (HARD-EARNED if empirical
    # is within 2σ of predicted; CARGO-CULTED if outside 2σ)
    # The 2σ threshold is 0.5 * |predicted ΔS| (so empirical within
    # ±0.5x of predicted = HARD-EARNED; empirical outside = CARGO-CULTED)
    sigma_predicted = 0.5 * abs(PAIR_2_PREDICTED_DELTA_S)
    if sigma_predicted < 1e-12:
        residual_zscore = float("inf")
    else:
        residual_zscore = abs(empirical_delta_s - PAIR_2_PREDICTED_DELTA_S) / sigma_predicted
    canonical_equation_verdict = (
        "HARD-EARNED"
        if residual_zscore < PAIR_2_ZSCORE_HARD_EARNED_THRESHOLD
        else "CARGO-CULTED"
    )

    # Sub-verdict: rescue path validated if bytes_saved > 0 (net savings)
    rescue_path_net_savings_validated = bytes_saved_c_vs_a > 0

    # Magic-codec cascade verdict (operator-facing summary)
    if canonical_equation_verdict == "HARD-EARNED" and rescue_path_net_savings_validated:
        cascade_verdict = (
            "PAIR_2_VALIDATED_PROCEED_TO_PAIR_3_OR_PAIR_4_ORTHOGONALITY"
        )
    elif rescue_path_net_savings_validated:
        cascade_verdict = (
            "PARTIAL_RESCUE_NET_SAVINGS_BUT_OUTSIDE_PREDICTED_BAND"
        )
    elif canonical_equation_verdict == "CARGO-CULTED":
        cascade_verdict = (
            "PAIR_2_FALSIFIED_CASCADE_FURTHER_NARROWS_PIVOT_TO_PAIR_4_OR_DP1_ONLY"
        )
    else:
        cascade_verdict = "INDETERMINATE_REQUIRES_PAIRED_LINUX_X86_64_VERIFICATION"

    end_utc = _utc_now_iso()

    try:
        archive_path_for_record = str(archive_path.relative_to(REPO_ROOT))
    except ValueError:
        archive_path_for_record = str(archive_path)
    try:
        matrix_path_for_record = str(matrix_path.relative_to(REPO_ROOT))
    except ValueError:
        matrix_path_for_record = str(matrix_path)

    return {
        "smoke_label": "wave_3_magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_cpu_smoke",
        "smoke_lane_id": "lane_wave_3_magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_cpu_smoke_20260520",
        "smoke_cascade_stacking_analysis_memo": (
            ".omx/research/magic_codec_x_todays_cascade_stacking_analysis_20260520.md"
        ),
        "smoke_pair_id": "pair_2_sparse_packet_ir_srl1_x_procedural_codebook_null_byte_residuals",
        "smoke_baseline_anchors": [
            "feedback_null_byte_probe_matrix_landed_20260520",
            "feedback_magic_codec_dense_streams_dwt_residual_smoke_landed_20260520",
        ],
        "started_at_utc": start_utc,
        "completed_at_utc": end_utc,
        "platform": platform.platform(),
        # fec6 frontier custody (Catalog #343 + #245 canonical pointers)
        "fec6_archive_path_repo_relative": archive_path_for_record,
        "fec6_archive_sha256": archive_sha,
        "fec6_archive_size_bytes": archive_size,
        "fec6_inner_member_sha256": _sha256_of_bytes(inner_bytes),
        "fec6_inner_member_size_bytes": inner_size,
        "fec6_frontier_lane_id": FEC6_FRONTIER_LANE_ID,
        "master_gradient_npy_path": str(master_gradient_npy_path),
        "empirical_null_value_zero_fraction": empirical_null_value_zero_fraction,
        "null_byte_matrix_path_repo_relative": matrix_path_for_record,
        "null_byte_matrix_anchor_n_null_expected": n_null_expected,
        "null_byte_matrix_anchor_n_null_observed": n_null_observed,
        "null_byte_matrix_anchor_match_observed_vs_expected": (
            n_null_observed == n_null_expected
        ),
        "null_byte_matrix_anchor_label": matrix_anchor.get(
            "grammar_detected_label", "FEC6"
        ),
        "null_byte_matrix_anchor_codec_family": matrix_anchor.get(
            "codec_family", "hnerv_family"
        ),
        "null_byte_matrix_anchor_axis": matrix_anchor.get(
            "axis", "[macOS-CPU advisory]"
        ),
        "null_byte_matrix_anchor_null_fraction": matrix_anchor.get(
            "null_fraction"
        ),
        "n_null_used_in_smoke": n_null_used,
        "generator_kind": generator_kind,
        "base_seed_bytes_hex": base_seed_bytes.hex(),
        "base_seed_bytes_len": len(base_seed_bytes),
        "empirical_uint8_sha256": _sha256_of_bytes(empirical_u8.tobytes()),
        "synthetic_uint8_sha256": _sha256_of_bytes(synthetic_u8.tobytes()),
        "residual_int16_sha256": _sha256_of_bytes(residual_int16.tobytes()),
        "mutated_residual_int16_sha256": _sha256_of_bytes(mutated_residual.tobytes()),
        "config_a_in_place_charged_bytes": config_a_in_place_bytes,
        "config_a_baseline_brotli_bytes": config_a_brotli_bytes,
        "config_b_procedural_only_bytes": config_b_bytes,
        "config_c_procedural_plus_srl1_bytes": config_c_bytes,
        "config_c_selection_log": config_c_selection_log,
        "config_c_alt_brotli_on_residual_bytes": config_c_alt_brotli_bytes,
        "config_c_srl1_vs_brotli_residual_delta_bytes": (
            config_c_bytes - (len(base_seed_bytes) + config_c_alt_brotli_bytes)
        ),
        "bytes_saved_c_vs_a": bytes_saved_c_vs_a,
        "bytes_saved_c_vs_a_brotli_audit": bytes_saved_c_vs_a_brotli_audit,
        "empirical_delta_s": empirical_delta_s,
        "predicted_delta_s_pair_2": PAIR_2_PREDICTED_DELTA_S,
        "composition_alpha_estimate_pair_2": PAIR_2_COMPOSITION_ALPHA_ESTIMATE,
        "sigma_predicted_for_zscore": sigma_predicted,
        "residual_zscore_empirical_vs_predicted": residual_zscore,
        "canonical_equation_verdict_HARD_EARNED_or_CARGO_CULTED_at_2sigma": canonical_equation_verdict,
        "rescue_path_net_savings_validated": rescue_path_net_savings_validated,
        "cascade_verdict": cascade_verdict,
        "residual_distribution": {
            "dtype": str(residual_int16.dtype),
            "mean": float(residual_int16.mean()),
            "std": float(residual_int16.std()),
            "abs_max": int(np.abs(residual_int16).max()),
            "n_zero_entries": int(np.count_nonzero(residual_int16 == 0)),
            "n_nonzero_entries": int(np.count_nonzero(residual_int16)),
            "zero_fraction": float(np.mean(residual_int16 == 0)),
        },
        "byte_mutation_smoke_mutated_config_c_bytes": mutated_config_c_bytes,
        "byte_mutation_smoke_byte_diff_count": byte_diff_count,
        "byte_mutation_smoke_encoded_bytes_differ": (
            config_c_selection_log["srl1_serialized_payload_sha256"]
            != mutated_log["srl1_serialized_payload_sha256"]
        ),
        "byte_mutation_smoke_verdict_seed_sensitive": byte_mutation_seed_sensitive,
        "axis_tag": "[macOS-CPU advisory]",
        "hardware_substrate": "darwin_arm64_m5_max_macos_cpu_advisory",
        "evidence_grade": "local_cpu_smoke_advisory",
        "promotion_eligible": False,
        "score_claim_valid": False,
        "score_claim_axis": None,
        "canonical_equation_id": "procedural_codebook_from_seed_compression_savings_v1",
        "canonical_equation_in_domain_context": (
            "sparse_packet_ir_srl1_correction_on_fec6_frontier_null_bytes"
        ),
        "canonical_equation_third_empirical_anchor_pending_append": True,
    }


def append_third_empirical_anchor(
    smoke_result: dict[str, Any],
    output_json_path: Path,
) -> None:
    """Append the THIRD empirical anchor to canonical equation #26.

    Per Catalog #344 sister discipline + Catalog #127 per-call-site custody
    (macOS-CPU advisory non-promotable per Catalog #192). The NEW IN-DOMAIN
    context is ``sparse_packet_ir_srl1_correction_on_fec6_frontier_null_bytes``,
    distinct from:
    * FIRST anchor (yesterday's DWT distributional fit under direct
      substitution H0; CARGO-CULTED at KL=1.638 nats)
    * SECOND anchor (pair #1 magic_codec_dense_streams residual correction
      on DWT detail subbands; CARGO-CULTED at residual_zscore=38.8 due to
      pcg64-uniform predictor / Laplacian empirical mismatch)

    Per CLAUDE.md "Forbidden premature KILL": a CARGO-CULTED verdict on
    THIS smoke does NOT mean the cascade is dead — it means pair #2
    falsified at byte-budget surface for the (pcg64, fec6
    master-gradient-null positions, exact int16 residual) tuple and the
    cascade either further narrows (pair #4 orthogonality validation) or
    pivots to DP1-only.
    """
    from tac.canonical_equations.equation import EmpiricalAnchor
    from tac.canonical_equations.registry import update_equation_with_empirical_anchor
    from tac.provenance.builders import build_provenance_for_macos_cpu_advisory

    try:
        source_artifact_relpath = str(output_json_path.relative_to(REPO_ROOT))
    except ValueError:
        source_artifact_relpath = str(output_json_path)
    source_artifact_sha = _sha256_of_file(output_json_path)

    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=source_artifact_sha,
        source_path=source_artifact_relpath,
    )

    anchor = EmpiricalAnchor(
        anchor_id=f"third_empirical_anchor_wave_3_magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke_{_utc_now_filename()}",
        measurement_utc=smoke_result["completed_at_utc"],
        inputs={
            "smoke_label": smoke_result["smoke_label"],
            "smoke_lane_id": smoke_result["smoke_lane_id"],
            "smoke_pair_id": smoke_result["smoke_pair_id"],
            "fec6_archive_path": smoke_result["fec6_archive_path_repo_relative"],
            "fec6_archive_sha256": smoke_result["fec6_archive_sha256"],
            "fec6_archive_size_bytes": smoke_result["fec6_archive_size_bytes"],
            "fec6_frontier_lane_id": smoke_result["fec6_frontier_lane_id"],
            "null_byte_matrix_path": smoke_result["null_byte_matrix_path_repo_relative"],
            "n_null_used_in_smoke": smoke_result["n_null_used_in_smoke"],
            "null_fraction": smoke_result["null_byte_matrix_anchor_null_fraction"],
            "generator_kind": smoke_result["generator_kind"],
            "base_seed_bytes_len": smoke_result["base_seed_bytes_len"],
            "axis_tag": smoke_result["axis_tag"],
            "hardware_substrate": smoke_result["hardware_substrate"],
            "evidence_grade": smoke_result["evidence_grade"],
            "in_domain_context": smoke_result["canonical_equation_in_domain_context"],
            "composition_alpha_estimate_pair_2": smoke_result[
                "composition_alpha_estimate_pair_2"
            ],
        },
        predicted_output={
            "predicted_delta_s": smoke_result["predicted_delta_s_pair_2"],
            "predicted_delta_s_source": (
                "magic_codec_x_todays_cascade_stacking_analysis_20260520_memo_section_7_"
                "pair_2_additive_alpha_0p9_prediction"
            ),
            "sigma_predicted_for_zscore": smoke_result["sigma_predicted_for_zscore"],
            "hypothesis_status": (
                "predicted_pair_2_sparse_packet_ir_srl1_byte_savings_on_fec6_frontier_"
                "null_bytes_within_2sigma_of_additive_alpha_0p9_composition"
            ),
        },
        empirical_output={
            "config_a_in_place_charged_bytes": smoke_result[
                "config_a_in_place_charged_bytes"
            ],
            "config_a_baseline_brotli_bytes": smoke_result[
                "config_a_baseline_brotli_bytes"
            ],
            "config_b_procedural_only_bytes": smoke_result[
                "config_b_procedural_only_bytes"
            ],
            "config_c_procedural_plus_srl1_bytes": smoke_result[
                "config_c_procedural_plus_srl1_bytes"
            ],
            "config_c_alt_brotli_on_residual_bytes": smoke_result[
                "config_c_alt_brotli_on_residual_bytes"
            ],
            "config_c_srl1_vs_brotli_residual_delta_bytes": smoke_result[
                "config_c_srl1_vs_brotli_residual_delta_bytes"
            ],
            "bytes_saved_c_vs_a": smoke_result["bytes_saved_c_vs_a"],
            "bytes_saved_c_vs_a_brotli_audit": smoke_result[
                "bytes_saved_c_vs_a_brotli_audit"
            ],
            "empirical_delta_s": smoke_result["empirical_delta_s"],
            "residual_zscore_empirical_vs_predicted": smoke_result[
                "residual_zscore_empirical_vs_predicted"
            ],
            "canonical_equation_verdict_HARD_EARNED_or_CARGO_CULTED_at_2sigma": smoke_result[
                "canonical_equation_verdict_HARD_EARNED_or_CARGO_CULTED_at_2sigma"
            ],
            "rescue_path_net_savings_validated": smoke_result[
                "rescue_path_net_savings_validated"
            ],
            "cascade_verdict": smoke_result["cascade_verdict"],
            "byte_mutation_smoke_verdict_seed_sensitive": smoke_result[
                "byte_mutation_smoke_verdict_seed_sensitive"
            ],
            "residual_zero_fraction": smoke_result["residual_distribution"][
                "zero_fraction"
            ],
            "residual_n_nonzero_entries": smoke_result["residual_distribution"][
                "n_nonzero_entries"
            ],
            "srl1_codec_name": smoke_result["config_c_selection_log"][
                "codec_name"
            ],
        },
        residual=float(
            abs(smoke_result["empirical_delta_s"] - smoke_result["predicted_delta_s_pair_2"])
        ),
        source_artifact=source_artifact_relpath,
        measurement_method=(
            "fec6_frontier_inner_member_gradient_null_indices_from_master_gradient_npy_"
            "paired_with_null_byte_matrix_anchor_then_three_way_comparison_config_a_"
            "in_place_charged_bytes_and_secondary_brotli_baseline_vs_config_b_"
            "procedural_only_32B_seed_vs_config_c_procedural_"
            "plus_sparse_packet_ir_srl1_encode_rle_of_zeros_serialize_rle_of_zeros_"
            "residual_correction_local_macos_cpu_smoke_advisory"
        ),
        provenance=provenance,
    )

    update_equation_with_empirical_anchor(
        equation_id="procedural_codebook_from_seed_compression_savings_v1",
        anchor=anchor,
        agent="codex",
        subagent_id="codex-main-wave-3-magic-codec-pair-2-sparse-packet-ir-fec6-null-byte-cpu-smoke-20260521",
        notes=(
            "WAVE-3 pair #2 THIRD empirical anchor; LOCAL macOS-CPU advisory smoke "
            "per Catalog #192 + #127 + #323; NEW IN-DOMAIN context "
            "sparse_packet_ir_srl1_correction_on_fec6_frontier_null_bytes; "
            "third $0 CPU smoke today validating magic-codec cascade per Carmack "
            "MVP-first phasing BEFORE any paid GPU investment in the cascade"
        ),
    )


def emit_markdown_report(smoke_result: dict[str, Any], md_path: Path) -> None:
    """Write a human-readable Markdown table summarizing the smoke."""
    lines = [
        "<!-- SPDX-License-Identifier: MIT -->",
        "# WAVE-3 Pair #2: sparse_packet_ir SRL1 × procedural-codebook null-byte residual smoke",
        "",
        f"**Lane**: `{smoke_result['smoke_lane_id']}`  ",
        f"**Stacking analysis memo**: `{smoke_result['smoke_cascade_stacking_analysis_memo']}`  ",
        f"**Pair ID**: `{smoke_result['smoke_pair_id']}`  ",
        f"**Baseline anchors**: `{smoke_result['smoke_baseline_anchors']}`  ",
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
        f"* fec6 frontier archive: `{smoke_result['fec6_archive_path_repo_relative']}`",
        f"* Archive sha256: `{smoke_result['fec6_archive_sha256'][:16]}...`",
        f"* Archive size (zip wrapper): `{smoke_result['fec6_archive_size_bytes']}` bytes",
        f"* Inner-member size: `{smoke_result['fec6_inner_member_size_bytes']}` bytes (canonical for master-gradient indices)",
        f"* Frontier lane id: `{smoke_result['fec6_frontier_lane_id']}`",
        f"* Master gradient npy: `{smoke_result['master_gradient_npy_path']}`",
        f"* Empirical bytes at null-leverage positions: `{smoke_result['empirical_null_value_zero_fraction']:.4%}` are byte-value zero (canonical \"null leverage\" != byte-value zero)",
        f"* Null byte matrix: `{smoke_result['null_byte_matrix_path_repo_relative']}`",
        f"* Expected null bytes (matrix anchor): `{smoke_result['null_byte_matrix_anchor_n_null_expected']}`",
        f"* Observed null bytes (master-gradient tensor scan): `{smoke_result['null_byte_matrix_anchor_n_null_observed']}`",
        f"* Match observed vs expected: `{smoke_result['null_byte_matrix_anchor_match_observed_vs_expected']}`",
        f"* Null fraction: `{smoke_result['null_byte_matrix_anchor_null_fraction']:.4%}`",
        f"* N null bytes used in smoke: `{smoke_result['n_null_used_in_smoke']}`",
        f"* PRNG generator kind: `{smoke_result['generator_kind']}`",
        f"* Base seed length: `{smoke_result['base_seed_bytes_len']}` bytes",
        "",
        "## 3-way apples-to-apples byte budget",
        "",
        "| Configuration | Bytes | Notes |",
        "|---|---:|---|",
        f"| A-primary: empirical bytes kept in place | {smoke_result['config_a_in_place_charged_bytes']} | contest-byte baseline for replacing these member-byte positions |",
        f"| A-secondary: empirical selected bytes via brotli | {smoke_result['config_a_baseline_brotli_bytes']} | audit only; brotli(q=11) over selected gradient-null byte values |",
        f"| B: procedural only (32B seed substituting selected bytes) | {smoke_result['config_b_procedural_only_bytes']} | lossy at inflate unless generated bytes exactly match empirical bytes |",
        f"| C: procedural + SRL1 residual (pair #2 candidate) | {smoke_result['config_c_procedural_plus_srl1_bytes']} | 32B seed + serialize_rle_of_zeros(exact int16 residual) |",
        f"| Sister: procedural + brotli on residual (orthogonality audit) | {smoke_result['config_c_alt_brotli_on_residual_bytes'] + smoke_result['base_seed_bytes_len']} | shows brotli alternative cost for the same residual |",
        f"| SRL1 vs brotli-on-residual delta | {smoke_result['config_c_srl1_vs_brotli_residual_delta_bytes']:+d} | positive = SRL1 LARGER than brotli (SRL1 lost orthogonality audit) |",
        f"| **Bytes saved (C vs A-primary)** | **{smoke_result['bytes_saved_c_vs_a']:+d}** | pair #2 contest-byte net savings |",
        f"| Bytes saved (C vs A-secondary brotli audit) | {smoke_result['bytes_saved_c_vs_a_brotli_audit']:+d} | secondary audit only |",
        "",
        "## SRL1 selection log",
        "",
        f"* Codec: `{smoke_result['config_c_selection_log']['codec_name']}` (id `{smoke_result['config_c_selection_log']['codec_id']}`)",
        f"* SRL1 serialized payload: `{smoke_result['config_c_selection_log']['srl1_serialized_payload_len']}` bytes",
        f"* SRL1 envelope overhead: `{smoke_result['config_c_selection_log']['srl1_envelope_overhead_bytes']}` bytes",
        f"* SRL1 indices bytes: `{smoke_result['config_c_selection_log']['srl1_indices_bytes']}`",
        f"* SRL1 values bytes: `{smoke_result['config_c_selection_log']['srl1_values_bytes']}`",
        f"* Residual dtype: `{smoke_result['residual_distribution']['dtype']}`",
        f"* Residual sparsity ratio: `{smoke_result['config_c_selection_log']['residual_sparsity_ratio']:.6f}` (fraction of zero entries in residual)",
        f"* Residual n_nonzero entries: `{smoke_result['config_c_selection_log']['n_nonzero_residual_entries']}`",
        f"* Residual n_zero entries: `{smoke_result['config_c_selection_log']['n_zero_residual_entries']}`",
        "",
        "## Residual distribution",
        "",
        f"* Mean: `{smoke_result['residual_distribution']['mean']:.4f}`",
        f"* Std: `{smoke_result['residual_distribution']['std']:.4f}`",
        f"* Abs max: `{smoke_result['residual_distribution']['abs_max']}`",
        f"* Zero entries: `{smoke_result['residual_distribution']['n_zero_entries']}`",
        f"* Nonzero entries: `{smoke_result['residual_distribution']['n_nonzero_entries']}`",
        f"* Zero fraction: `{smoke_result['residual_distribution']['zero_fraction']:.6f}`",
        "",
        "## Aggregate canonical-equation verdict",
        "",
        f"* Empirical ΔS: `{smoke_result['empirical_delta_s']:+.6f}`",
        f"* Predicted ΔS (pair #2 per stacking analysis memo §7): `{smoke_result['predicted_delta_s_pair_2']:+.6f}`",
        f"* Composition_alpha estimate: `{smoke_result['composition_alpha_estimate_pair_2']}` (ADDITIVE)",
        f"* Residual zscore (empirical vs predicted): `{smoke_result['residual_zscore_empirical_vs_predicted']:.4f}`",
        f"* **Canonical equation #26 verdict**: `{smoke_result['canonical_equation_verdict_HARD_EARNED_or_CARGO_CULTED_at_2sigma']}` (at 2σ threshold)",
        f"* **Rescue path net savings validated**: `{'YES' if smoke_result['rescue_path_net_savings_validated'] else 'NO'}`",
        f"* **Cascade verdict**: `{smoke_result['cascade_verdict']}`",
        f"* **Catalog #272 byte-mutation smoke verdict**: `{'PASSED' if smoke_result['byte_mutation_smoke_verdict_seed_sensitive'] else 'FAILED'}` (seed-sensitive)",
        "",
        "## Equation linkage",
        "",
        f"* Canonical equation: `{smoke_result['canonical_equation_id']}`",
        f"* NEW IN-DOMAIN context: `{smoke_result['canonical_equation_in_domain_context']}`",
        f"* THIRD empirical anchor pending append: `{smoke_result['canonical_equation_third_empirical_anchor_pending_append']}`",
        "",
        "## Discipline citations",
        "",
        "* CLAUDE.md \"MPS auth eval is NOISE\" — macOS-CPU is NEVER score truth",
        "* CLAUDE.md \"Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE\" — `[macOS-CPU advisory]` non-promotable",
        "* CLAUDE.md \"Apples-to-apples evidence discipline\" — axis labels preserved",
        "* CLAUDE.md \"Bit-level deconstruction and entropy discipline\" — 3-way comparison at byte level on master-gradient-null fec6 member positions",
        "* CLAUDE.md \"Forbidden premature KILL\" — CARGO-CULTED pair #2 verdict pivots cascade further (pair #4 OR DP1-only); does NOT kill the magic-codec stacking paradigm",
        "* Catalog #127 — custody triple axis × hardware × evidence_grade",
        "* Catalog #192 — macOS-CPU advisory not promotable without Linux x86_64 verification",
        "* Catalog #272 — distinguishing-feature byte-mutation smoke (seed-sensitivity verified)",
        "* Catalog #287 — placeholder-rationale rejection",
        "* Catalog #309 — horizon_class=`frontier_breaking_enabler`",
        "* Catalog #318 — master-gradient null-space surface (sister)",
        "* Catalog #323 — canonical Provenance umbrella",
        "* Catalog #324 — predicted_band validation (predicted_band_validation_status pending_post_training)",
        "* Catalog #335 — canonical consumer contract (sister auto-discoverable)",
        "* Catalog #343 — frontier pointer (fec6 archive identified via canonical sha prefix not hardcoded)",
        "* Catalog #344 — canonical equation THIRD `anchor_appended` event with NEW IN-DOMAIN context",
        "",
        "## Source",
        "",
        "* MAGIC CODEC × CASCADE STACKING ANALYSIS memo: `.omx/research/magic_codec_x_todays_cascade_stacking_analysis_20260520.md`",
        "* Pair #1 baseline (sister-COMPLEMENTARY): `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_magic_codec_dense_streams_dwt_residual_smoke_landed_20260520.md`",
        "* Null-byte matrix (sister-COMPLEMENTARY): `~/.claude/projects/-Users-adpena-Projects-pact/memory/feedback_null_byte_probe_matrix_landed_20260520.md`",
        "* Op-routable: stacking analysis memo §14 Top-3 #2 (FREE local probe; pair #2)",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "WAVE-3 pair #2 sparse_packet_ir SRL1 × procedural-codebook "
            "null-byte residual local macOS-CPU smoke. NON-promotable; "
            "macOS-CPU advisory per CLAUDE.md \"MPS auth eval is NOISE\" + "
            "Catalog #192."
        )
    )
    parser.add_argument(
        "--archive-path",
        type=Path,
        default=DEFAULT_FEC6_ARCHIVE_PATH,
        help="Path to fec6 frontier archive.zip (default: canonical fec6 frontier path).",
    )
    parser.add_argument(
        "--null-byte-matrix-path",
        type=Path,
        default=DEFAULT_NULL_BYTE_MATRIX_PATH,
        help="Path to null_byte_matrix.json (default: canonical null_byte_probe_matrix output).",
    )
    parser.add_argument(
        "--base-seed-hex",
        type=str,
        default=None,
        help=(
            "Hex-encoded base seed bytes (default: sha256 of canonical "
            "WAVE-3 pair #2 label). Length must be 16-512 hex chars."
        ),
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
        "--max-null-indices",
        type=int,
        default=None,
        help=(
            "Optional cap on the number of null-byte indices used in the "
            "smoke (default: ALL 16,292 fec6 frontier null bytes). For "
            "fast iteration / regression tests."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory under experiments/results/ (default: "
            "experiments/results/magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke_<utc>/)."
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

    if not args.archive_path.exists():
        print(f"FATAL: archive_path={args.archive_path} not found", file=sys.stderr)
        return 2
    if not args.null_byte_matrix_path.exists():
        print(
            f"FATAL: null_byte_matrix_path={args.null_byte_matrix_path} not found",
            file=sys.stderr,
        )
        return 2

    if args.base_seed_hex is None:
        base_seed_bytes = hashlib.sha256(
            b"wave_3_magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_cpu_smoke_20260520"
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
            / f"magic_codec_pair_2_sparse_packet_ir_fec6_null_byte_smoke_{_utc_now_filename()}"
        )
    else:
        output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "smoke_result.json"
    md_path = output_dir / "smoke_result.md"

    print(
        f"[wave-3-magic-codec-pair-2] loading fec6 frontier {args.archive_path}...",
        file=sys.stderr,
    )
    smoke_result = run_smoke(
        archive_path=args.archive_path,
        matrix_path=args.null_byte_matrix_path,
        base_seed_bytes=base_seed_bytes,
        generator_kind=args.generator_kind,
        max_null_indices=args.max_null_indices,
    )

    print(
        "[wave-3-magic-codec-pair-2] writing smoke_result.json + smoke_result.md",
        file=sys.stderr,
    )
    json_path.write_text(
        json.dumps(smoke_result, indent=2, sort_keys=True), encoding="utf-8"
    )
    emit_markdown_report(smoke_result, md_path)

    if not args.skip_canonical_equation_append:
        print(
            "[wave-3-magic-codec-pair-2] appending THIRD empirical anchor to canonical "
            "equation procedural_codebook_from_seed_compression_savings_v1 "
            "(NEW IN-DOMAIN context sparse_packet_ir_srl1_correction_on_"
            "fec6_frontier_null_bytes)...",
            file=sys.stderr,
        )
        append_third_empirical_anchor(smoke_result, json_path)
        print(
            "[wave-3-magic-codec-pair-2] anchor_appended event landed (Catalog #344 sister)",
            file=sys.stderr,
        )
    else:
        print(
            "[wave-3-magic-codec-pair-2] SKIPPED canonical equation anchor_appended event "
            "(--skip-canonical-equation-append)",
            file=sys.stderr,
        )

    try:
        out_dir_display = str(output_dir.relative_to(REPO_ROOT))
    except ValueError:
        out_dir_display = str(output_dir)
    print(
        f"[wave-3-magic-codec-pair-2] DONE: output_dir={out_dir_display}",
        file=sys.stderr,
    )
    print(
        f"[wave-3-magic-codec-pair-2] "
        f"config_a_in_place={smoke_result['config_a_in_place_charged_bytes']}B "
        f"config_a_brotli_audit={smoke_result['config_a_baseline_brotli_bytes']}B "
        f"config_b_procedural_only={smoke_result['config_b_procedural_only_bytes']}B "
        f"config_c_procedural_plus_srl1={smoke_result['config_c_procedural_plus_srl1_bytes']}B "
        f"bytes_saved_c_vs_a={smoke_result['bytes_saved_c_vs_a']:+d} "
        f"empirical_delta_s={smoke_result['empirical_delta_s']:+.6f} "
        f"predicted_delta_s={smoke_result['predicted_delta_s_pair_2']:+.6f} "
        f"residual_zscore={smoke_result['residual_zscore_empirical_vs_predicted']:.4f} "
        f"verdict={smoke_result['canonical_equation_verdict_HARD_EARNED_or_CARGO_CULTED_at_2sigma']} "
        f"cascade={smoke_result['cascade_verdict']} "
        f"byte_mutation_seed_sensitive={smoke_result['byte_mutation_smoke_verdict_seed_sensitive']}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
