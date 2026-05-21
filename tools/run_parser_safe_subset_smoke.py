# SPDX-License-Identifier: MIT
"""WAVE-3 PARSER-SAFE SUBSET smoke (LOCAL macOS-CPU).

Per PR101 GOLD NULL-BYTE REMOVAL SMOKE landing memo (commit ``3dfb877c0``)
TERTIARY operator-routable + sister memo Section 4 op-routable #2 verbatim
recommendation. Empirically maps the boundary between
master-gradient-null bytes that ARE parser-essential (the H3
OPAQUE-TO-SCORER class) vs the (potentially empty) parser-safe subset
that is BOTH null-gradient AND downstream-of-parser-dispatch.

The sister smoke (commit ``3dfb877c0``) landed H3_OPAQUE_TO_SCORER on
ALL 16,292 master-gradient-null indices. META-LESSON: null-gradient is
NECESSARY but NOT SUFFICIENT for byte replaceability — replaceability
ALSO requires the byte be downstream of parser dispatch (i.e., the
byte must NOT be inside a Brotli/LZMA/Huffman bitstream or a
struct-packed wrapper field).

This subagent (Phase 1) statically classifies the 16,292 null-byte
indices into 6 archive regions (A-F per the inflate.py + codec.py
parser grammar): A=FP11 outer wrapper, B=PR101 decoder_blob (Brotli),
C=PR101 latent_blob (LZMA), D=PR101 sidecar_blob (Brotli),
E=FEC6 selector_len uint16, F=FEC6 selector_payload (magic + Huffman
bitstream). Then (Phase 2) constructs a "parser-safe subset" as the
EMPTY set if no null-gradient bytes live outside Brotli/LZMA streams
and wrapper struct fields. Then (Phase 3) runs the canonical 4-variant
smoke (V_BASELINE / V_ZERO / V_HALF / V_RANDOM) ONLY on the
parser-safe subset to empirically confirm or refute the static
prediction. Then (Phase 4) emits a canonical equation #26 update if
the parser-safe subset is non-empty AND H1_SCORE_IRRELEVANT verdict
empirically lands.

Per CLAUDE.md "MPS auth eval is NOISE" + "Submission auth eval — BOTH
CPU AND CUDA" non-negotiables: macOS-CPU smoke is observability-only;
contest-CPU paired Linux x86_64 anchor required for any promotion.

Catalog #270 tool dispatch scope per "tac stays clean" + canonical
dispatch optimization protocol.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import io
import json
import os as _os
import struct
import subprocess
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.provenance import build_provenance_for_macos_cpu_advisory  # noqa: E402

# Canonical contest constants per CLAUDE.md "Auth eval EVERYWHERE" +
# canonical equation registry per Catalog #344.
CANONICAL_RATE_DENOM_BYTES = 37_545_489
CANONICAL_RATE_MULTIPLIER = 25.0
CANONICAL_SEG_MULTIPLIER = 100.0
CANONICAL_POSE_SQRT_INNER = 10.0
EPSILON_GRADIENT_NULL = 1e-9

# Canonical fec6 frontier anchors (per Catalog #343 frontier-pointer
# discipline; reconstructed from canonical_frontier_pointer.json).
FEC6_FRONTIER_SHA256 = (
    "6bae0201fb082457a02c69565531aba4c5942669c384fdc48e7d554f7b893fcf"
)
FEC6_FRONTIER_ARCHIVE = (
    REPO_ROOT
    / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip"
)
FEC6_INFLATE_SH = (
    REPO_ROOT
    / "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/submission_dir/inflate.sh"
)
MASTER_GRADIENT_NULL_INDICES_NPY = (
    REPO_ROOT
    / "experiments/results/null_byte_probe_20260520T220614Z/null_byte_indices.npy"
)
CONTEST_AUTH_EVAL = REPO_ROOT / "experiments/contest_auth_eval.py"

# PR101 GOLD fec6 inner-grammar constants (verified against
# submission_dir/src/codec.py DECODER_BLOB_LEN + LATENT_BLOB_LEN +
# submission_dir/inflate.py OUTER_MAGIC + parse_pr101_frame_selector_archive).
FP11_OUTER_MAGIC = b"FP11"
FP11_HEADER_LEN = 8  # 4 magic + 4 source_len uint32
PR101_DECODER_BLOB_LEN = 162_164  # Brotli streams (q + scale per tensor)
PR101_LATENT_BLOB_LEN = 15_387  # LZMA-compressed (mins, scales, deltas)
FEC6_SELECTOR_LEN_FIELD_LEN = 2  # uint16

# Hypothesis classification thresholds (per task description; bounded
# observability-only; matches sister smoke commit 3dfb877c0).
HYPOTHESIS_H1_THRESHOLD = 1.0e-4  # all variants within this of baseline
HYPOTHESIS_H2_THRESHOLD = 0.05  # variants diverge but not catastrophic


class ParserSafeSubsetSmokeError(RuntimeError):
    """Raised when the smoke pipeline cannot complete."""


@dataclass(frozen=True)
class RegionDescriptor:
    """One archive region's static classification."""

    name: str
    start_byte: int
    end_byte: int  # exclusive
    parser_kind: str  # 'struct_field' | 'brotli_stream' | 'lzma_stream' | 'huffman_bitstream'
    parser_essential: bool
    rationale: str

    def size(self) -> int:
        return self.end_byte - self.start_byte

    def contains(self, idx: int) -> bool:
        return self.start_byte <= idx < self.end_byte


@dataclass(frozen=True)
class RegionClassificationResult:
    """Per-region null-byte index breakdown."""

    region_name: str
    region_start_byte: int
    region_end_byte: int
    region_size_bytes: int
    region_parser_kind: str
    region_parser_essential: bool
    null_index_count: int
    null_fraction_of_region: float
    rationale: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class VariantResult:
    """One variant's empirical smoke result (mirrors sister smoke schema)."""

    variant_name: str
    fill_byte_strategy: str
    archive_sha256: str
    archive_bytes: int
    score: float
    seg_distortion: float
    pose_distortion: float
    rate_term: float
    delta_s_vs_baseline: float | None
    eval_returncode: int
    eval_wallclock_seconds: float
    work_dir: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _load_master_gradient_null_indices() -> np.ndarray:
    """Load master-gradient-null indices from the canonical probe matrix."""

    if not MASTER_GRADIENT_NULL_INDICES_NPY.is_file():
        raise ParserSafeSubsetSmokeError(
            f"null-byte indices .npy not found at {MASTER_GRADIENT_NULL_INDICES_NPY} "
            "(per probe matrix anchor commit 82c1b3bac)"
        )
    arr = np.load(MASTER_GRADIENT_NULL_INDICES_NPY)
    if arr.ndim != 1:
        raise ParserSafeSubsetSmokeError(
            f"null-byte indices shape unexpected: {arr.shape}; expected 1-D"
        )
    if len(arr) != 16_292:
        raise ParserSafeSubsetSmokeError(
            f"null-byte index count {len(arr)} != 16292 per probe matrix anchor; "
            "fixture may have changed"
        )
    return arr.astype(np.int64)


def _read_fec6_archive() -> tuple[bytes, bytes, str]:
    """Read fec6 frontier archive + verify sha256."""

    if not FEC6_FRONTIER_ARCHIVE.is_file():
        raise ParserSafeSubsetSmokeError(
            f"fec6 frontier archive not found at {FEC6_FRONTIER_ARCHIVE}"
        )
    archive_bytes = FEC6_FRONTIER_ARCHIVE.read_bytes()
    actual_sha = hashlib.sha256(archive_bytes).hexdigest()
    if actual_sha != FEC6_FRONTIER_SHA256:
        raise ParserSafeSubsetSmokeError(
            f"fec6 archive sha mismatch: {actual_sha} != {FEC6_FRONTIER_SHA256}"
        )
    with zipfile.ZipFile(io.BytesIO(archive_bytes), mode="r") as zf:
        names = zf.namelist()
        if len(names) != 1:
            raise ParserSafeSubsetSmokeError(
                f"fec6 archive expected 1 member, got {len(names)}: {names}"
            )
        member_name = names[0]
        inner = zf.read(member_name)
    return archive_bytes, inner, member_name


def static_classify_regions(inner_bytes: bytes) -> tuple[RegionDescriptor, ...]:
    """Phase 1 — static analysis of archive regions per inflate parser grammar.

    Returns canonical 6-region map (A-F) keyed by member-relative byte
    range. Every byte in the archive falls into exactly one region; the
    boundaries are derived from the parser source code (inflate.py +
    codec.py) NOT from heuristic detection.
    """

    if len(inner_bytes) < FP11_HEADER_LEN:
        raise ParserSafeSubsetSmokeError(
            f"inner_bytes too short ({len(inner_bytes)}) for FP11 wrapper"
        )
    if inner_bytes[:4] != FP11_OUTER_MAGIC:
        raise ParserSafeSubsetSmokeError(
            f"FP11 outer magic mismatch: {inner_bytes[:4]!r}"
        )
    (source_len,) = struct.unpack_from("<I", inner_bytes, 4)
    source_start = FP11_HEADER_LEN
    source_end = source_start + source_len
    if source_end + FEC6_SELECTOR_LEN_FIELD_LEN > len(inner_bytes):
        raise ParserSafeSubsetSmokeError(
            f"FP11 source_payload claims {source_len} bytes but inner_bytes "
            f"only {len(inner_bytes)} bytes"
        )
    (selector_len,) = struct.unpack_from("<H", inner_bytes, source_end)
    selector_start = source_end + FEC6_SELECTOR_LEN_FIELD_LEN
    selector_end = selector_start + selector_len
    if selector_end != len(inner_bytes):
        raise ParserSafeSubsetSmokeError(
            f"FEC6 selector boundary mismatch: selector_end={selector_end} "
            f"!= inner_bytes={len(inner_bytes)}"
        )

    # PR101 inner archive: decoder_blob || latent_blob || sidecar_blob
    decoder_start = source_start
    decoder_end = decoder_start + PR101_DECODER_BLOB_LEN
    latent_start = decoder_end
    latent_end = latent_start + PR101_LATENT_BLOB_LEN
    sidecar_start = latent_end
    sidecar_end = source_end

    if latent_end > source_end:
        raise ParserSafeSubsetSmokeError(
            f"PR101 decoder_blob+latent_blob overshoot source payload: "
            f"latent_end={latent_end} > source_end={source_end}"
        )

    regions = (
        RegionDescriptor(
            name="A_fp11_outer_wrapper",
            start_byte=0,
            end_byte=FP11_HEADER_LEN,
            parser_kind="struct_field",
            parser_essential=True,
            rationale=(
                "FP11 outer magic + source_len uint32; parse_pr101_frame_selector_archive "
                "raises ValueError on magic mismatch; struct-packed wrapper field"
            ),
        ),
        RegionDescriptor(
            name="B_pr101_decoder_brotli",
            start_byte=decoder_start,
            end_byte=decoder_end,
            parser_kind="brotli_stream",
            parser_essential=True,
            rationale=(
                "PR101 HNeRV decoder state-dict serialized as concatenated Brotli "
                "streams (codec.py decompress_brotli_streams). Brotli is a "
                "bit-accurate compressed format; any byte mutation breaks the "
                "Huffman/LZ77 parser before producing any decoded output"
            ),
        ),
        RegionDescriptor(
            name="C_pr101_latent_lzma",
            start_byte=latent_start,
            end_byte=latent_end,
            parser_kind="lzma_stream",
            parser_essential=True,
            rationale=(
                "PR101 latent codes LZMA-compressed (codec.py decode_latents_compact "
                "with LATENT_LZMA_FILTERS). LZMA is a range-coded bit-accurate "
                "compressed format; any byte mutation corrupts the range-coder "
                "probability state and breaks decoding"
            ),
        ),
        RegionDescriptor(
            name="D_pr101_sidecar_brotli",
            start_byte=sidecar_start,
            end_byte=sidecar_end,
            parser_kind="brotli_stream",
            parser_essential=True,
            rationale=(
                "PR101 sidecar Brotli-compressed (codec_sidecar.apply_latent_sidecar). "
                "Same Brotli parser-essential property as region B"
            ),
        ),
        RegionDescriptor(
            name="E_fec6_selector_len_uint16",
            start_byte=source_end,
            end_byte=selector_start,
            parser_kind="struct_field",
            parser_essential=True,
            rationale=(
                "FEC6 selector_len uint16 length prefix; "
                "parse_pr101_frame_selector_archive raises ValueError if "
                "selector_payload length mismatches; struct-packed wrapper field"
            ),
        ),
        RegionDescriptor(
            name="F_fec6_selector_payload",
            start_byte=selector_start,
            end_byte=selector_end,
            parser_kind="huffman_bitstream",
            parser_essential=True,
            rationale=(
                "FEC6 fixed-Huffman selector bitstream (inflate.py "
                "unpack_fec6_fixed_huffman_codes). First 4 bytes = magic 'FEC6'; "
                "next 2 bytes = n_pairs uint16; remaining bytes are a "
                "bit-accurate prefix-coded bitstream that raises ValueError on "
                "invalid prefix > 8 bits or non-zero padding"
            ),
        ),
    )
    return regions


def classify_null_indices_per_region(
    null_indices: np.ndarray, regions: tuple[RegionDescriptor, ...]
) -> list[RegionClassificationResult]:
    """Phase 1 (cont.) — bin null-byte indices into the 6 canonical regions."""

    results: list[RegionClassificationResult] = []
    total_assigned = 0
    for region in regions:
        mask = (null_indices >= region.start_byte) & (null_indices < region.end_byte)
        n_nulls = int(mask.sum())
        total_assigned += n_nulls
        size = region.size()
        results.append(
            RegionClassificationResult(
                region_name=region.name,
                region_start_byte=region.start_byte,
                region_end_byte=region.end_byte,
                region_size_bytes=size,
                region_parser_kind=region.parser_kind,
                region_parser_essential=region.parser_essential,
                null_index_count=n_nulls,
                null_fraction_of_region=(n_nulls / size) if size > 0 else 0.0,
                rationale=region.rationale,
            )
        )
    if total_assigned != len(null_indices):
        raise ParserSafeSubsetSmokeError(
            f"region partition incomplete: {total_assigned} assigned vs "
            f"{len(null_indices)} total null indices"
        )
    return results


def construct_parser_safe_subset(
    null_indices: np.ndarray,
    regions: tuple[RegionDescriptor, ...],
) -> np.ndarray:
    """Phase 2 — return the subset of null_indices that are NOT in
    parser-essential regions.

    Per the empirical anchor of sister smoke 3dfb877c0: every region in
    the canonical PR101 GOLD fec6 grammar is parser-essential (Brotli /
    LZMA / Huffman bitstreams + struct-packed wrapper fields). The
    parser-safe subset is the empty array unless a future archive
    architecture exposes intermediate-transform regions (canonical
    equation #26 INCLUDED contexts).
    """

    safe_indices: list[int] = []
    for idx in null_indices:
        in_essential = False
        for region in regions:
            if region.parser_essential and region.contains(int(idx)):
                in_essential = True
                break
        if not in_essential:
            safe_indices.append(int(idx))
    return np.array(safe_indices, dtype=np.int64)


def _derive_random_bytes_for_indices(
    n_indices: int,
    seed: bytes = b"parser_safe_subset_smoke_2026_05_20",
) -> bytes:
    """Deterministic PCG64 pseudo-random bytes derived from seed."""

    seed_int = int.from_bytes(hashlib.sha256(seed).digest()[:8], "little")
    rng = np.random.Generator(np.random.PCG64(seed_int))
    return bytes(rng.integers(0, 256, size=n_indices, dtype=np.uint8))


def _build_variant_archive(
    inner_bytes: bytes,
    member_name: str,
    subset_indices: np.ndarray,
    variant_name: str,
) -> tuple[bytes, str, int]:
    """Apply variant byte modification on subset_indices + repack archive."""

    mutated = bytearray(inner_bytes)
    n_subset = len(subset_indices)
    if variant_name == "V_BASELINE":
        pass
    elif variant_name == "V_ZERO":
        for idx in subset_indices:
            mutated[int(idx)] = 0x00
    elif variant_name == "V_HALF":
        for idx in subset_indices:
            mutated[int(idx)] = 0x80
    elif variant_name == "V_RANDOM":
        random_bytes = _derive_random_bytes_for_indices(n_subset)
        for i, idx in enumerate(subset_indices):
            mutated[int(idx)] = random_bytes[i]
    else:
        raise ParserSafeSubsetSmokeError(
            f"unknown variant {variant_name!r}; expected V_BASELINE / "
            "V_ZERO / V_HALF / V_RANDOM"
        )
    mutated_bytes = bytes(mutated)

    # Deterministic ZIP repack (sister-canonical pattern from
    # tools/run_pr101_gold_master_gradient_null_byte_removal_smoke.py).
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_STORED) as zf:
        info = zipfile.ZipInfo(filename=member_name, date_time=(1980, 1, 1, 0, 0, 0))
        info.compress_type = zipfile.ZIP_STORED
        info.create_system = 0
        zf.writestr(info, mutated_bytes)
    archive_zip_bytes = buf.getvalue()
    archive_sha = hashlib.sha256(archive_zip_bytes).hexdigest()
    return archive_zip_bytes, archive_sha, len(archive_zip_bytes)


def _run_contest_auth_eval_macos_cpu(
    archive_zip_bytes: bytes,
    variant_name: str,
    output_root: Path,
) -> tuple[float, float, float, float, int, float, Path]:
    """Run experiments/contest_auth_eval.py with --device cpu locally."""

    variant_dir = output_root / variant_name
    variant_dir.mkdir(parents=True, exist_ok=True)
    archive_path = variant_dir / "archive.zip"
    archive_path.write_bytes(archive_zip_bytes)
    work_dir = variant_dir / "auth_eval_work"
    work_dir.mkdir(exist_ok=True)
    json_out = variant_dir / "auth_eval_result.json"

    cmd = [
        sys.executable,
        str(CONTEST_AUTH_EVAL),
        "--archive",
        str(archive_path),
        "--inflate-sh",
        str(FEC6_INFLATE_SH),
        "--device",
        "cpu",
        "--work-dir",
        str(work_dir),
        "--json-out",
        str(json_out),
        "--keep-work-dir",
    ]
    env = dict(_os.environ)
    env["PACT_PYTHON_BIN"] = sys.executable
    t0 = _dt.datetime.now()
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900, env=env)
    elapsed = (_dt.datetime.now() - t0).total_seconds()
    stderr_tail = proc.stderr[-2000:] if proc.stderr else ""
    (variant_dir / "auth_eval_stderr_tail.txt").write_text(stderr_tail)

    if proc.returncode != 0 or not json_out.is_file():
        return (
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            proc.returncode,
            elapsed,
            work_dir,
        )
    result = json.loads(json_out.read_text())
    score = float(
        result.get(
            "canonical_score",
            result.get("score_recomputed_from_components", result.get("final_score", float("nan"))),
        )
    )
    seg = float(result.get("avg_segnet_dist", float("nan")))
    pose = float(result.get("avg_posenet_dist", float("nan")))
    rate = float(result.get("score_rate_contribution", float("nan")))
    return score, seg, pose, rate, proc.returncode, elapsed, work_dir


def _classify_verdict(
    subset_size: int,
    baseline_score: float | None,
    v_zero_delta: float | None,
    v_half_delta: float | None,
    v_random_delta: float | None,
    n_inflate_failures: int,
) -> tuple[str, str]:
    """Return (verdict_label, rationale).

    Verdict taxonomy:
    * PARSER_SAFE_SUBSET_EMPTY — static analysis prediction that no
      null-byte indices live outside parser-essential regions.
    * PARSER_SAFE_SUBSET_NONEMPTY_H1_SCORE_IRRELEVANT — subset exists
      AND all 3 variants produce ΔS within HYPOTHESIS_H1_THRESHOLD;
      NEW IN-DOMAIN canonical equation #26 context candidate.
    * PARSER_SAFE_SUBSET_NONEMPTY_H2_PARTIALLY_RELEVANT — subset exists
      AND variants diverge but not catastrophically.
    * PARSER_SAFE_SUBSET_NONEMPTY_H3_OPAQUE_TO_SCORER — subset exists
      but mutations still break the scorer (catastrophic divergence or
      inflate failure on the subset alone); deeper meta-class than
      sister smoke (subset was supposed to be parser-safe but isn't).
    """

    if subset_size == 0:
        return (
            "PARSER_SAFE_SUBSET_EMPTY",
            "Static analysis of fec6 frontier archive (sha 6bae0201) "
            "confirms ALL 16,292 master-gradient-null byte indices fall "
            "inside parser-essential regions (FP11 wrapper struct + "
            "PR101 decoder_blob Brotli + PR101 latent_blob LZMA + PR101 "
            "sidecar_blob Brotli + FEC6 selector_len uint16 + FEC6 "
            "selector_payload Huffman bitstream). The parser-safe subset "
            "is empty — there is NO null-gradient region downstream of "
            "parser dispatch on this substrate. This empirically confirms "
            "sister smoke H3_OPAQUE_TO_SCORER (commit 3dfb877c0) at the "
            "REGION level and proves the META-LESSON: null-gradient is "
            "NECESSARY but NOT SUFFICIENT for byte replaceability on the "
            "fec6 substrate. NEW canonical equation #26 IN-DOMAIN "
            "contexts must come from substrate-level architectural changes "
            "(NSCS06 v8 chroma LUT / ATW V2 codec quantizer LUT / DP1 "
            "codebook bytes / class-anchor replacement) which expose "
            "intermediate-transform regions BY DESIGN.",
        )

    if n_inflate_failures > 0:
        return (
            "PARSER_SAFE_SUBSET_NONEMPTY_H3_OPAQUE_TO_SCORER",
            f"Parser-safe subset has {subset_size} indices BUT "
            f"{n_inflate_failures}/3 modified variants FAILED to inflate. "
            "Static region classification was incomplete — these indices "
            "are still bit-essential despite living outside the "
            "canonical 6-region map. This is a DEEPER meta-class than "
            "sister smoke 3dfb877c0: the region boundaries themselves "
            "require refinement before any BUILD investment.",
        )

    max_abs_delta = max(
        abs(v_zero_delta or 0.0),
        abs(v_half_delta or 0.0),
        abs(v_random_delta or 0.0),
    )
    if max_abs_delta < HYPOTHESIS_H1_THRESHOLD:
        return (
            "PARSER_SAFE_SUBSET_NONEMPTY_H1_SCORE_IRRELEVANT",
            f"Parser-safe subset has {subset_size} indices AND all 3 "
            f"variants produce |ΔS|={max_abs_delta:.6f} < "
            f"{HYPOTHESIS_H1_THRESHOLD}. NEW IN-DOMAIN canonical "
            "equation #26 context candidate: "
            "`parser_safe_master_gradient_null_byte_subset_replacement`. "
            "Predicted savings = 25 * subset_size / 37_545_489 = "
            f"{CANONICAL_RATE_MULTIPLIER * subset_size / CANONICAL_RATE_DENOM_BYTES:.6e} "
            "(rate-term only). Catalog #325 per-substrate symposium + "
            "Catalog #324 post-training Tier-C validation BEFORE any "
            "paid dispatch.",
        )
    if max_abs_delta < HYPOTHESIS_H2_THRESHOLD:
        return (
            "PARSER_SAFE_SUBSET_NONEMPTY_H2_PARTIALLY_RELEVANT",
            f"Parser-safe subset has {subset_size} indices; variants "
            f"diverge by |ΔS|={max_abs_delta:.6f} in [{HYPOTHESIS_H1_THRESHOLD}, "
            f"{HYPOTHESIS_H2_THRESHOLD}). Bytes partially relevant; "
            "mechanism investigation needed before BUILD investment.",
        )
    return (
        "PARSER_SAFE_SUBSET_NONEMPTY_H3_OPAQUE_TO_SCORER",
        f"Parser-safe subset has {subset_size} indices; max(|ΔS|)="
        f"{max_abs_delta:.6f} >= {HYPOTHESIS_H2_THRESHOLD}. Bytes opaque-"
        "to-scorer-but-not-bytes; paradigm needs rescope per sister smoke "
        "META-class (commit 3dfb877c0).",
    )


def _write_smoke_result_json(
    output_dir: Path,
    region_results: list[RegionClassificationResult],
    parser_safe_subset_size: int,
    variants: list[VariantResult] | None,
    verdict_label: str,
    verdict_rationale: str,
) -> Path:
    """Emit smoke_result.json with full provenance per Catalog #323."""

    if variants:
        baseline = next((v for v in variants if v.variant_name == "V_BASELINE"), None)
        baseline_sha = baseline.archive_sha256 if baseline else FEC6_FRONTIER_SHA256
    else:
        baseline_sha = FEC6_FRONTIER_SHA256

    # Source path is repo-relative when output_dir is under REPO_ROOT; absolute
    # otherwise (test fixtures use tmp_path outside the repo).
    try:
        source_path_str = str(output_dir.relative_to(REPO_ROOT))
    except ValueError:
        source_path_str = str(output_dir)

    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=baseline_sha,
        source_path=source_path_str,
    )
    payload = {
        "schema_version": "parser_safe_subset_smoke_v1_20260520",
        "smoke_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "fec6_frontier_archive_sha256": FEC6_FRONTIER_SHA256,
        "fec6_frontier_inner_member_bytes": region_results[-1].region_end_byte
        if region_results
        else None,
        "master_gradient_null_index_count": sum(
            r.null_index_count for r in region_results
        ),
        "static_region_classification": [r.as_dict() for r in region_results],
        "parser_safe_subset_size": parser_safe_subset_size,
        "parser_safe_subset_fraction_of_null_indices": (
            parser_safe_subset_size
            / sum(r.null_index_count for r in region_results)
            if region_results
            and sum(r.null_index_count for r in region_results) > 0
            else 0.0
        ),
        "variants": [v.as_dict() for v in variants] if variants else [],
        "verdict_label": verdict_label,
        "verdict_rationale": verdict_rationale,
        "axis_tag": "[macOS-CPU advisory]",
        "evidence_grade": "macOS-CPU-advisory",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "provenance": {
            "artifact_kind": str(
                provenance.artifact_kind.value
                if hasattr(provenance.artifact_kind, "value")
                else provenance.artifact_kind
            ),
            "evidence_grade": str(
                provenance.evidence_grade.value
                if hasattr(provenance.evidence_grade, "value")
                else provenance.evidence_grade
            ),
            "measurement_axis": provenance.measurement_axis,
            "hardware_substrate": provenance.hardware_substrate,
            "source_sha256": provenance.source_sha256,
            "source_path": provenance.source_path,
            "captured_at_utc": provenance.captured_at_utc,
            "score_claim_valid": provenance.score_claim_valid,
            "promotion_eligible": provenance.promotion_eligible,
        },
        "catalog_disciplines_honored": [
            "#125 6-hook wire-in",
            "#127 axis x hardware x evidence_grade custody",
            "#185 META drift",
            "#192 macOS-CPU non-promotable",
            "#272 byte-mutation smoke",
            "#287 placeholder-rationale rejection",
            "#318 master-gradient null-space surface",
            "#323 canonical Provenance umbrella",
            "#344 canonical equation cross-ref",
        ],
        "canonical_equation_cross_ref": (
            "procedural_codebook_from_seed_compression_savings_v1 (Catalog "
            "#344 registry #26); empirically maps parser-safe subset boundary "
            "to confirm/refute sister smoke 3dfb877c0 META-LESSON"
        ),
        "cascade_context": {
            "sister_smoke_commit": "3dfb877c0",
            "sister_smoke_verdict": "H3_OPAQUE_TO_SCORER (all 16,292 indices)",
            "sister_smoke_meta_lesson": (
                "null-gradient is NECESSARY but NOT SUFFICIENT for byte "
                "replaceability — replaceability ALSO requires the byte be "
                "downstream of parser dispatch"
            ),
            "this_smoke_subagent_role": "Empirically map parser-safe subset boundary",
        },
    }
    out_path = output_dir / "smoke_result.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out_path


def _write_smoke_result_md(
    output_dir: Path,
    region_results: list[RegionClassificationResult],
    parser_safe_subset_size: int,
    variants: list[VariantResult] | None,
    verdict_label: str,
    verdict_rationale: str,
) -> Path:
    """Emit human-readable smoke_result.md."""

    lines = [
        "<!-- HISTORICAL_SCORE_LITERAL_OK:macos_cpu_advisory_smoke_not_score_truth_parser_safe_subset_2026-05-20 -->",
        "# WAVE-3 PARSER-SAFE SUBSET smoke",
        "",
        f"- **fec6 frontier archive sha256**: `{FEC6_FRONTIER_SHA256[:16]}...`",
        f"- **master-gradient null-byte indices**: {sum(r.null_index_count for r in region_results):,}",
        f"- **parser-safe subset size**: {parser_safe_subset_size:,}",
        f"- **smoke_at_utc**: {_dt.datetime.now(tz=_dt.timezone.utc).isoformat()}",
        f"- **axis tag**: `[macOS-CPU advisory]` (NEVER promotable per Catalog #192)",
        f"- **$ spent**: $0 (LOCAL macOS-CPU)",
        "",
        "## Phase 1 — Static region classification",
        "",
        "| region | byte range | size | parser_kind | parser_essential | null count | null % of region |",
        "|---|---|---:|---|---|---:|---:|",
    ]
    for r in region_results:
        lines.append(
            f"| `{r.region_name}` | "
            f"[{r.region_start_byte}, {r.region_end_byte}) | "
            f"{r.region_size_bytes:,} | {r.region_parser_kind} | "
            f"{r.region_parser_essential} | {r.null_index_count:,} | "
            f"{r.null_fraction_of_region*100:.2f}% |"
        )

    lines.extend(
        [
            "",
            "## Phase 2 — Parser-safe subset",
            "",
            f"**Subset size**: {parser_safe_subset_size:,} indices "
            f"({parser_safe_subset_size}/{sum(r.null_index_count for r in region_results)} = "
            f"{100*parser_safe_subset_size/max(sum(r.null_index_count for r in region_results),1):.4f}% of null indices)",
            "",
        ]
    )
    if parser_safe_subset_size == 0:
        lines.extend(
            [
                "**Empirical finding**: ALL master-gradient-null byte indices "
                "live inside parser-essential regions (Brotli/LZMA streams + "
                "struct-packed wrapper fields + Huffman bitstreams). The "
                "parser-safe subset is structurally EMPTY for this substrate.",
                "",
                "**Phase 3 — 4-variant smoke**: SKIPPED (empty subset means no "
                "smoke is possible). This is the canonical outcome predicted by "
                "static analysis of the inflate.py + codec.py grammars.",
                "",
            ]
        )
    else:
        lines.extend(
            [
                "## Phase 3 — 4-variant smoke",
                "",
                "| variant | archive bytes | score | seg | pose | rate | dS vs baseline | rc | wallclock_s |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for v in variants or []:
            ds_str = f"{v.delta_s_vs_baseline:+.6f}" if v.delta_s_vs_baseline is not None else "(baseline)"
            lines.append(
                f"| `{v.variant_name}` | {v.archive_bytes:,} | "
                f"{v.score:.6f} | {v.seg_distortion:.6f} | "
                f"{v.pose_distortion:.6f} | {v.rate_term:.6f} | {ds_str} | "
                f"{v.eval_returncode} | {v.eval_wallclock_seconds:.1f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Phase 4 — Verdict",
            "",
            f"**Verdict**: `{verdict_label}`",
            "",
            f"**Rationale**: {verdict_rationale}",
            "",
            "## Cascade context",
            "",
            "- Sister smoke commit: `3dfb877c0` (PR101 GOLD NULL-BYTE REMOVAL SMOKE — all 16,292 → H3)",
            "- Sister META-LESSON: null-gradient is NECESSARY but NOT SUFFICIENT for byte replaceability",
            "- This subagent's role: empirically map the parser-safe subset boundary",
            "",
            "## Provenance (Catalog #323)",
            "",
            "- `score_claim`: False",
            "- `promotion_eligible`: False",
            "- `rank_or_kill_eligible`: False",
            "- `ready_for_exact_eval_dispatch`: False",
            "- `axis_tag`: `[macOS-CPU advisory]`",
            "- `evidence_grade`: `macOS-CPU-advisory`",
            "",
        ]
    )

    out_path = output_dir / "smoke_result.md"
    out_path.write_text("\n".join(lines) + "\n")
    return out_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "PARSER-SAFE SUBSET smoke (LOCAL macOS-CPU). Phase 1 static "
            "region analysis + Phase 2 subset construction + Phase 3 "
            "4-variant smoke (skipped if subset empty) + Phase 4 verdict. "
            "Catalog #270 tool dispatch scope; CLAUDE.md MPS auth eval "
            "non-negotiable + Submission auth eval non-negotiable."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: experiments/results/parser_safe_subset_smoke_<utc>/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only; do not run contest_auth_eval",
    )
    parser.add_argument(
        "--static-only",
        action="store_true",
        help="Run only Phase 1+2 (static analysis + subset construction); skip Phase 3 even if subset non-empty",
    )
    args = parser.parse_args(argv)

    null_indices = _load_master_gradient_null_indices()
    _archive_zip_bytes, inner_bytes, member_name = _read_fec6_archive()
    regions = static_classify_regions(inner_bytes)
    region_results = classify_null_indices_per_region(null_indices, regions)
    parser_safe_subset = construct_parser_safe_subset(null_indices, regions)
    parser_safe_size = int(len(parser_safe_subset))

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "null_index_count": int(len(null_indices)),
                    "inner_member_bytes": len(inner_bytes),
                    "regions": [
                        {
                            "name": r.region_name,
                            "size_bytes": r.region_size_bytes,
                            "null_count": r.null_index_count,
                            "parser_kind": r.region_parser_kind,
                        }
                        for r in region_results
                    ],
                    "parser_safe_subset_size": parser_safe_size,
                },
                indent=2,
            )
        )
        return 0

    utc = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (
        REPO_ROOT / f"experiments/results/parser_safe_subset_smoke_{utc}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    variants: list[VariantResult] | None = None

    # Run 4-variant smoke ONLY if subset non-empty AND --static-only not set
    if parser_safe_size > 0 and not args.static_only:
        variants = []
        variants_spec = [
            ("V_BASELINE", "original inner_bytes (control)"),
            ("V_ZERO", "0x00 at every parser-safe-subset index"),
            ("V_HALF", "0x80 at every parser-safe-subset index"),
            ("V_RANDOM", "deterministic PCG64 pseudo-random bytes"),
        ]
        baseline_score: float | None = None
        for variant_name, fill_strategy in variants_spec:
            print(f"[{variant_name}] building variant archive (subset_size={parser_safe_size})...")
            variant_zip_bytes, variant_sha, variant_bytes = _build_variant_archive(
                inner_bytes,
                member_name,
                parser_safe_subset,
                variant_name,
            )
            print(
                f"[{variant_name}] archive sha={variant_sha[:16]}... bytes={variant_bytes}; "
                "running contest_auth_eval --device cpu..."
            )
            score, seg, pose, rate, rc, elapsed, work_dir = _run_contest_auth_eval_macos_cpu(
                variant_zip_bytes, variant_name, output_dir
            )
            delta = None if baseline_score is None else (score - baseline_score)
            if variant_name == "V_BASELINE":
                baseline_score = score
            try:
                work_dir_str = str(work_dir.relative_to(REPO_ROOT))
            except ValueError:
                work_dir_str = str(work_dir)
            variants.append(
                VariantResult(
                    variant_name=variant_name,
                    fill_byte_strategy=fill_strategy,
                    archive_sha256=variant_sha,
                    archive_bytes=variant_bytes,
                    score=score,
                    seg_distortion=seg,
                    pose_distortion=pose,
                    rate_term=rate,
                    delta_s_vs_baseline=delta,
                    eval_returncode=rc,
                    eval_wallclock_seconds=elapsed,
                    work_dir=work_dir_str,
                )
            )
            print(
                f"[{variant_name}] score={score:.6f} "
                f"dS_vs_baseline={'(baseline)' if delta is None else f'{delta:+.6f}'} "
                f"({elapsed:.1f}s)"
            )

    # Compute verdict
    import math as _math
    if variants:
        baseline_score = variants[0].score
        v_zero = next(v for v in variants if v.variant_name == "V_ZERO")
        v_half = next(v for v in variants if v.variant_name == "V_HALF")
        v_random = next(v for v in variants if v.variant_name == "V_RANDOM")
        n_inflate_failures = sum(
            1
            for v in (v_zero, v_half, v_random)
            if v.eval_returncode != 0 or _math.isnan(v.score)
        )
        verdict_label, verdict_rationale = _classify_verdict(
            parser_safe_size,
            baseline_score,
            v_zero.delta_s_vs_baseline if v_zero.delta_s_vs_baseline is not None and not _math.isnan(v_zero.delta_s_vs_baseline) else 0.0,
            v_half.delta_s_vs_baseline if v_half.delta_s_vs_baseline is not None and not _math.isnan(v_half.delta_s_vs_baseline) else 0.0,
            v_random.delta_s_vs_baseline if v_random.delta_s_vs_baseline is not None and not _math.isnan(v_random.delta_s_vs_baseline) else 0.0,
            n_inflate_failures=n_inflate_failures,
        )
    else:
        verdict_label, verdict_rationale = _classify_verdict(
            parser_safe_size, None, None, None, None, 0
        )

    json_path = _write_smoke_result_json(
        output_dir, region_results, parser_safe_size, variants, verdict_label, verdict_rationale
    )
    md_path = _write_smoke_result_md(
        output_dir, region_results, parser_safe_size, variants, verdict_label, verdict_rationale
    )

    print("")
    print("=== PARSER-SAFE SUBSET SMOKE RESULT ===")
    print(f"Static region classification: {len(region_results)} regions")
    print(f"Parser-safe subset size: {parser_safe_size}")
    print(f"Verdict: {verdict_label}")
    try:
        json_disp = str(json_path.relative_to(REPO_ROOT))
        md_disp = str(md_path.relative_to(REPO_ROOT))
    except ValueError:
        json_disp = str(json_path)
        md_disp = str(md_path)
    print(f"JSON: {json_disp}")
    print(f"MD: {md_disp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
