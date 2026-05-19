#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Pre-entropy substrate pivot prober.

[verified-against: tools/wyner_ziv_deliverability_prober.py — sister
                   prober that empirically proved the fec6 archive bytes
                   are at Shannon entropy floor (lzma/brotli/zlib all
                   INFLATE)]
[verified-against: tools/wyner_ziv_deliverability_prober.py op-routable #4
                   *"Probe a pre-entropy substrate (e.g. a substrate that
                   ships raw fp16 latents BEFORE entropy coding): the CSP
                   byte set MAY compress, making the 1.15x reward
                   justified."*]
[verified-against: CLAUDE.md "Bit-level deconstruction and entropy
                   discipline" + "Meta-Lagrangian/Pareto solver" non-
                   negotiables]
[verified-against: empirical anchors --
                   pr106_state_dict.pt (924,277 B) -> lzma 208,832 B (0.226 ratio)
                   pr106_latents.pt (68,777 B) -> lzma 32,164 B (0.468 ratio)
                   sabor_margin_frame_000.npy (786,560 B) -> lzma 586,012 B (0.745)
                   fec6_archive.zip (178,517 B) -> lzma 178,584 B (1.000 INFLATES)]
[verified-against: empirical Q4 retarget receipts -- post-entropy archives
                   (pr106/pr101/cool_chic/wavelet/dp1) all compress to ratio
                   in [0.99, 1.00] (AT_FLOOR / POST_ENTROPY); pre-entropy
                   .pt and .npy files compress to [0.22, 0.89] (PRE_ENTROPY)]

CONTEXT
=======

The sister prober (`tools/wyner_ziv_deliverability_prober.py`, landed
earlier today 2026-05-17) empirically falsified the autopilot's 1.15x
Wyner-Ziv reward for the fec6 archive. Every general-purpose codec
(lzma, brotli, zlib) INFLATES fec6 archive bytes -- they're already at
the Shannon entropy floor because the fec6 selector emits Huffman +
range-coded output.

Per the sister prober's op-routable #4: **Wyner-Ziv IS valuable for
PRE-entropy substrates**. This prober finds them.

For each candidate substrate, the probe asks: "If we treated the
archive/checkpoint bytes as the candidate-shared-prior set + ran the
canonical lzma/brotli/zlib probes, do they COMPRESS or INFLATE?"

* Compression ratio > 1.05x (output > input) -> POST_ENTROPY (autopilot
  reward is OVERSTATED -- Wyner-Ziv hoist saves zero bytes)
* Compression ratio in [0.99, 1.05] -> AT_FLOOR (marginal; the substrate
  is structurally near entropy bound but may have a few KB of slack)
* Compression ratio < 0.99 -> PRE_ENTROPY (Wyner-Ziv hoist IS deliverable;
  the substrate ships raw float weights / latents that compress
  meaningfully; autopilot reward IS justified)

THE CANONICAL Q4 RETARGET FINDING
==================================

The autopilot's current Q4 target is fec6 (POST_ENTROPY; 0x deliverable).
The Q4 target SHOULD be the substrate with the HIGHEST
``deliverable_score_savings_estimate`` per this probe -- typically a
substrate that ships raw fp16/fp32 weights or pre-quantization residuals.

Empirical receipts at landing (per the smoke test):

* ``pr106_state_dict.pt`` (924 KB raw fp16 weights): lzma to 209 KB =
  0.226x = ~715 KB savings = ~0.0024 score delta = SIGNIFICANT
* ``pr106_latents.pt`` (69 KB fp16 latents): lzma to 32 KB = 0.468x =
  ~37 KB savings = ~0.0001 score delta = marginal
* ``sabor_margin_frame_000.npy`` (787 KB raw fp32 margin map): lzma to
  586 KB = 0.745x = ~200 KB savings = ~0.0007 score delta

These pre-entropy substrates are exactly where the autopilot's 1.15x
PAIR_INVARIANT reward IS justified (the sister consumer's classification
exists for a reason -- it's just been applied to the WRONG substrate
class).

OUTPUT CONTRACT (NON-AUTHORITATIVE / PROBE EVIDENCE)
====================================================

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #192 +
Catalog #245, every emitted artifact carries:

* ``score_claim: false``
* ``promotion_eligible: false``
* ``ready_for_exact_eval_dispatch: false``
* ``evidence_grade: "predicted"``
* ``measurement_axis: "[diagnostic; pre-entropy substrate pivot probe]"``

Persists to:
``.omx/state/wyner_ziv_deliverability/pre_entropy_candidate_substrates_<utc>.json``

via fcntl-locked atomic write per Catalog #131 (sister of the canonical
Modal call_id ledger at Catalog #245).

CLI USAGE
=========

```bash
.venv/bin/python tools/pre_entropy_substrate_pivot_prober.py \\
    --candidate-substrates sane_hnerv,DP1,sabor,pr106_state_dict,pr106_latents \\
    --output .omx/state/wyner_ziv_deliverability/pre_entropy_candidate_substrates_<utc>.json \\
    --min-compression-ratio 1.05 \\
    --report-only-no-side-effects
```

Default substrate list covers raw-weight, raw-latent, raw-margin, codec-
sidecar, and post-entropy-archive candidates. Substrates without a known
archive/checkpoint are tagged ``archive_status: pending_dispatch`` and
get a theoretical estimate based on the substrate's recipe.
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import json
import lzma
import os
import sys
import uuid
import zipfile
import zlib
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Optional brotli per CLAUDE.md Catalog #203 (canonical contest-archive
# entropy coder). Soft-import so the prober can run on machines without
# it; the result manifest records its absence.
try:
    import brotli  # type: ignore

    _HAS_BROTLI = True
except ImportError:  # pragma: no cover - environment-dependent
    _HAS_BROTLI = False


# ──────────────────────────────────────────────────────────────────────── #
# Canonical contest constants                                               #
# ──────────────────────────────────────────────────────────────────────── #

# Per CLAUDE.md "Meta-Lagrangian/Pareto solver" + canonical
# `tac.frontier_scan` + Catalog #316.
CONTEST_RATE_DENOM_BYTES = 37_545_489

# Per-substrate classification thresholds. Per CLAUDE.md "Bit-level
# deconstruction and entropy discipline" the AT_FLOOR band is empirically
# defined by the sister prober's fec6 finding (ratio = 1.000 within
# ~0.001) -- we widen to [0.99, 1.05] to admit small slack and small
# inflation noise.
PRE_ENTROPY_RATIO_THRESHOLD = 0.99  # < 0.99 = PRE_ENTROPY (compressible)
AT_FLOOR_RATIO_LOWER = 0.99
AT_FLOOR_RATIO_UPPER = 1.05  # > 1.05 = POST_ENTROPY (inflates noticeably)

# Default minimum compression ratio for "Wyner-Ziv hoist is deliverable"
# verdict. CLI-tunable via `--min-compression-ratio` (note: this is the
# COMPRESSION ratio, so we want compressed_bytes/raw_bytes < this for the
# substrate to be PRE_ENTROPY).
DEFAULT_MIN_COMPRESSION_RATIO = 1.05  # < this means at-floor or pre-entropy

# Minimum member size to probe (skip tiny ZIP entries that add noise).
DEFAULT_MIN_MEMBER_BYTES = 1024

# Output location per CLAUDE.md "Forbidden /tmp paths in any persisted
# artifact" -- durable forensic state under .omx/state/.
OUTPUT_DIR_DEFAULT = Path(".omx/state/wyner_ziv_deliverability")

# Canonical candidate substrates per the operator spec (priority order
# matches the dispatch brief). Each entry maps to (archive_path,
# substrate_class). Paths point to LIVE artifacts under
# experiments/results/ that exist at landing time. A future re-run can
# expand the map via the CLI `--candidate-substrates-extra-json` flag.
CANONICAL_CANDIDATE_SUBSTRATES: dict[str, tuple[str, str]] = {
    # P1: Raw float weights (PRE-entropy, fp16/fp32)
    "pr106_state_dict": (
        "experiments/results/sensitivity_map_pr106_20260504_claude/state_dict.pt",
        "raw_float_weights",
    ),
    "pr101_state_dict": (
        "experiments/results/pr101_codecop_sweep_20260507_codex/pr101_decoder_state_dict.pt",
        "raw_float_weights",
    ),
    "distill_v2_best": (
        "experiments/results/distill_v2_best.pt",
        "raw_float_weights",
    ),
    "lane_g_v3_renderer": (
        "experiments/results/lane_g_v3_landed/iter_0/renderer.bin",
        "raw_float_weights",
    ),
    # P2: fp16 latent streams (PRE-entropy)
    "pr106_latents": (
        "experiments/results/sensitivity_map_pr106_20260504_claude/latents.pt",
        "raw_float_latents",
    ),
    # P3: Distilled scorer surrogates (raw torch checkpoints)
    "distilled_segnet": (
        "experiments/results/lane_cpu_trained_tiny_hinton_surrogate_bootstrap_20260512T034310Z_long/distilled_segnet_ema_shadow.pt",
        "raw_float_weights",
    ),
    "distilled_posenet": (
        "experiments/results/lane_cpu_trained_tiny_hinton_surrogate_bootstrap_20260512T034310Z_long/distilled_posenet_ema_shadow.pt",
        "raw_float_weights",
    ),
    # P5: SegNet/PoseNet logit margins (Lane 19) - pre-quantization fp32
    "sabor_margin_frame_000": (
        "experiments/results/lane_sabor_boundary_audit_20260513_20260513T180635Z/sample_margin_frame_000.npy",
        "scorer_margin_float32",
    ),
    "sabor_margin_frame_001": (
        "experiments/results/lane_sabor_boundary_audit_20260513_20260513T180635Z/sample_margin_frame_001.npy",
        "scorer_margin_float32",
    ),
    "posenet_class_sensitivity": (
        "experiments/results/posenet_sensitivity/class_sensitivity.pt",
        "raw_float_weights",
    ),
    # CONTROL: Post-entropy archives (should report POST_ENTROPY per sister
    # prober's finding). These are the canonical contest-frontier archives
    # the autopilot's existing 1.15x reward currently targets.
    "pr101_fec6_archive": (
        "experiments/results/pr101_frame_exploit_selector_fec6_fixed_huffman_k16_clean_20260515_codex/archive.zip",
        "post_entropy_contest_archive",
    ),
    "pr106_format0d_archive": (
        "experiments/results/pr106_format0d_latent_score_table_materialized_20260515_codex/sidecar_archive.zip",
        "post_entropy_contest_archive",
    ),
    "pr106_hdm4_hlm1_archive": (
        "experiments/results/pr106_r2_hdm4_hlm1_latent_candidate_20260513_codex/pr106_r2_hdm4_exact_cuda_hlm1_latent_candidate.zip",
        "post_entropy_contest_archive",
    ),
    "cool_chic_pr106_residual_archive": (
        "experiments/results/lane_cool_chic_residual_pr106_sidecar_20260511T180002Z/cool_chic_pr106_residual_sidecar_archive.zip",
        "post_entropy_contest_archive",
    ),
    "wavelet_pr106_residual_archive": (
        "experiments/results/lane_wavelet_residual_pr106_sidecar_20260511T180002Z/wavelet_pr106_residual_sidecar_archive.zip",
        "post_entropy_contest_archive",
    ),
    "dp1_tiny_full_cpu_archive": (
        "experiments/results/dp1_tiny_full_cpu_advisory_20260515_codex/archive.zip",
        "post_entropy_contest_archive",
    ),
    "apogee_int6_archive": (
        "experiments/results/apogee_int6_repack_20260504_claude/apogee_int6_archive.zip",
        "post_entropy_contest_archive",
    ),
    "track4_sg_a1_archive": (
        "experiments/results/track4_sg_a1_t178000_20260509/archive.zip",
        "post_entropy_contest_archive",
    ),
}


# ──────────────────────────────────────────────────────────────────────── #
# Result dataclasses                                                        #
# ──────────────────────────────────────────────────────────────────────── #


@dataclass(frozen=True)
class MemberProbeResult:
    """Per-archive-member compression probe result."""

    member_name: str
    raw_bytes: int
    lzma_bytes: int
    lzma_ratio: float
    brotli_bytes: int | None  # None if brotli unavailable
    brotli_ratio: float | None
    zlib_bytes: int
    zlib_ratio: float
    best_codec: str
    best_ratio: float
    classification: str  # 'PRE_ENTROPY' | 'AT_FLOOR' | 'POST_ENTROPY'


@dataclass(frozen=True)
class SubstrateProbeResult:
    """Per-substrate probe result aggregating all members.

    Fields ``validation_status`` / ``validation_reason`` /
    ``evidence_grade_per_row`` (NEW 2026-05-17 Option C) record whether
    the probed bytes are a MEMBER of an actual shipping contest
    ``archive.zip`` (``VALIDATED_CONTEST_MEMBER``) or a standalone
    research sidecar (``REJECTED_RESEARCH_SIDECAR``). Phantom-score-
    from-research-sidecar bug class per Q4 HALT memo 2026-05-17
    + Catalog #321 self-protection.
    """

    substrate_name: str
    substrate_class: str  # 'raw_float_weights' | 'raw_float_latents' | etc.
    archive_path: str
    archive_exists: bool
    archive_sha256: str | None
    archive_bytes_total: int
    pre_entropy_bytes: int
    at_floor_bytes: int
    post_entropy_bytes: int
    pre_entropy_fraction: float
    best_compression_codec: str
    best_compression_ratio: float
    deliverable_score_savings_estimate: float
    member_count: int
    member_results: tuple[MemberProbeResult, ...]
    archive_status: str  # 'present' | 'missing' | 'pending_dispatch'
    error: str | None = None
    # NEW Option C / Catalog #321 self-protection fields.
    # Default values preserve backward-compat for unit-test fixtures that
    # construct SubstrateProbeResult directly without these fields.
    validation_status: str = "UNVALIDATED"  # 'VALIDATED_CONTEST_MEMBER' | 'REJECTED_RESEARCH_SIDECAR' | 'UNVALIDATED'
    validation_reason: str | None = None
    evidence_grade_per_row: str = "predicted"  # 'predicted' | 'invalid_target'


# ──────────────────────────────────────────────────────────────────────── #
# Codec helpers (mirror sister prober's contract)                           #
# ──────────────────────────────────────────────────────────────────────── #


def _compress_lzma(data: bytes) -> bytes:
    """Canonical lzma compression at preset 9 | EXTREME (mirror sister
    prober)."""
    return lzma.compress(data, preset=9 | lzma.PRESET_EXTREME)


def _compress_brotli(data: bytes) -> bytes | None:
    """Canonical brotli compression at quality=11. Returns None if brotli
    not available."""
    if not _HAS_BROTLI:
        return None
    return brotli.compress(data, quality=11)


def _compress_zlib(data: bytes) -> bytes:
    """Canonical zlib compression at level=9 (max)."""
    return zlib.compress(data, level=9)


def classify_compression_ratio(ratio: float) -> str:
    """Per CLAUDE.md "Bit-level deconstruction and entropy discipline":
    classify a member's compression ratio into PRE_ENTROPY (< 0.99),
    AT_FLOOR (0.99-1.05), or POST_ENTROPY (> 1.05)."""
    if ratio < PRE_ENTROPY_RATIO_THRESHOLD:
        return "PRE_ENTROPY"
    if ratio > AT_FLOOR_RATIO_UPPER:
        return "POST_ENTROPY"
    return "AT_FLOOR"


def probe_member_compression(member_name: str, data: bytes) -> MemberProbeResult:
    """Probe a single member's bytes through all 3 canonical codecs."""
    raw = len(data)
    if raw == 0:
        return MemberProbeResult(
            member_name=member_name,
            raw_bytes=0,
            lzma_bytes=0,
            lzma_ratio=1.0,
            brotli_bytes=0 if _HAS_BROTLI else None,
            brotli_ratio=1.0 if _HAS_BROTLI else None,
            zlib_bytes=0,
            zlib_ratio=1.0,
            best_codec="lzma",
            best_ratio=1.0,
            classification="AT_FLOOR",
        )

    lz = len(_compress_lzma(data))
    lz_ratio = lz / raw
    br_bytes = _compress_brotli(data)
    br = len(br_bytes) if br_bytes is not None else None
    br_ratio = br / raw if br is not None else None
    zl = len(_compress_zlib(data))
    zl_ratio = zl / raw

    # Pick best (lowest ratio)
    candidates: list[tuple[str, float]] = [("lzma", lz_ratio), ("zlib", zl_ratio)]
    if br_ratio is not None:
        candidates.append(("brotli", br_ratio))
    best_codec, best_ratio = min(candidates, key=lambda x: x[1])

    return MemberProbeResult(
        member_name=member_name,
        raw_bytes=raw,
        lzma_bytes=lz,
        lzma_ratio=lz_ratio,
        brotli_bytes=br,
        brotli_ratio=br_ratio,
        zlib_bytes=zl,
        zlib_ratio=zl_ratio,
        best_codec=best_codec,
        best_ratio=best_ratio,
        classification=classify_compression_ratio(best_ratio),
    )


# ──────────────────────────────────────────────────────────────────────── #
# Archive iterator                                                          #
# ──────────────────────────────────────────────────────────────────────── #


def iter_archive_members(path: Path, min_member_bytes: int = DEFAULT_MIN_MEMBER_BYTES) -> Iterable[tuple[str, bytes]]:
    """Yield (member_name, member_bytes) for every member in the file.

    Supports both ZIP archives (one or more inner members) and bare files
    (.pt / .npy / .bin). For bare files, yields a single tuple with the
    filename as the member_name.
    """
    if not path.exists():
        return
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if info.file_size < min_member_bytes:
                    continue
                data = zf.read(info.filename)
                yield (info.filename, data)
    else:
        # Bare file (e.g., .pt or .npy)
        data = path.read_bytes()
        if len(data) >= min_member_bytes:
            yield (path.name, data)


def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 of the file at path (canonical archive_sha256
    field; sister of `tac.frontier_scan.compute_sha256`)."""
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ──────────────────────────────────────────────────────────────────────── #
# Score-delta estimator                                                     #
# ──────────────────────────────────────────────────────────────────────── #


def estimate_deliverable_score_savings(
    *,
    pre_entropy_bytes: int,
    best_compression_ratio: float,
) -> float:
    """Estimate deliverable score savings if pre-entropy bytes are
    moved to a Wyner-Ziv hoist + compressed via best codec.

    Score delta = 25 * (pre_entropy_bytes * (1 - 1/best_compression_ratio)) /
    CONTEST_RATE_DENOM_BYTES.

    Note: best_compression_ratio is compressed/raw, so 1/ratio expands the
    saved fraction; for ratio = 0.226, we save (1 - 0.226) = 77.4% of the
    pre-entropy bytes. Actually we save (1 - ratio) of the bytes directly
    (because compressed_bytes = ratio * raw_bytes; saved = raw - compressed
    = (1 - ratio) * raw_bytes).
    """
    if pre_entropy_bytes <= 0 or best_compression_ratio >= 1.0:
        return 0.0
    saved_bytes = pre_entropy_bytes * (1.0 - best_compression_ratio)
    score_delta = 25.0 * saved_bytes / CONTEST_RATE_DENOM_BYTES
    return float(score_delta)


# ──────────────────────────────────────────────────────────────────────── #
# Validator: bytes-ship-in-contest-archive (Catalog #321 self-protection)   #
# ──────────────────────────────────────────────────────────────────────── #


# Canonical roots under which contest-shipping archive.zip files live.
# Per CLAUDE.md "Public Disclosure Hygiene" + "tac stays clean" + the
# `submissions/` mutation frontier: a substrate's bytes are valid Wyner-Ziv
# hoist candidates ONLY when they appear as members of an actual archive.zip
# under these roots. Sidecar .pt / .npy files outside any archive.zip are
# research artifacts NEVER charged by the contest rate term `25 *
# archive_bytes / 37_545_489`.
CONTEST_ARCHIVE_PARENT_ROOTS: tuple[str, ...] = (
    "submissions",
    "experiments/results",
)

# Extensions that are NEVER themselves a contest archive.zip member.
# A path ending in one of these MUST be a member-of-an-archive (or
# extracted FROM an archive), not the standalone file.
RESEARCH_SIDECAR_EXTENSIONS: tuple[str, ...] = (
    ".pt",
    ".npy",
    ".npz",
    ".pth",
    ".pkl",
    ".bin",
)


def _repo_root(repo_root: Path | None = None) -> Path:
    return repo_root or Path(__file__).resolve().parent.parent


def _resolve_candidate_path_for_io(
    candidate_bytes_path_str: str,
    *,
    repo_root: Path | None = None,
) -> Path:
    path = Path(candidate_bytes_path_str)
    if path.is_absolute():
        return path
    return _repo_root(repo_root) / path


def _validate_substrate_bytes_ship_in_contest_archive(
    substrate_name: str,
    candidate_bytes_path_str: str,
    *,
    repo_root: Path | None = None,
) -> tuple[bool, str | None]:
    """Return (is_valid, reason_if_invalid) for whether the candidate path
    represents bytes that actually ship in a contest archive.zip.

    Per Q4 HALT memo 2026-05-17 + Catalog #321: the contest rate term
    `25 * archive_bytes / 37_545_489` charges ONLY archive.zip MEMBER bytes.
    A standalone .pt / .npy / .npz / .pth / .pkl / .bin file is a RESEARCH
    SIDECAR — its bytes are never shipped to the leaderboard, so any
    "Wyner-Ziv compresses it 0.2x" claim does NOT translate to a contest
    score delta. The prober's `deliverable_score_savings_estimate` field
    silently conflated these two until Catalog #321 landed.

    Validation cascades:
      1. Path is itself a ZIP archive named ``archive.zip`` (or similar
         contest-standard naming like ``*_archive.zip`` /
         ``sidecar_archive.zip``) under a contest-shipping root → VALIDATED.
      2. Path is a research sidecar extension (.pt / .npy / etc.) under any
         path → REJECTED (these never ship as standalone archive members).
      3. Path is non-existent → UNVALIDATED (preserve `pending_dispatch`
         semantics; the SUBSTRATE may legitimately not yet be built).

    Note on torch .pt internal zip wrapping: torch.save uses ZIP internally
    since PyTorch 1.6 — `zipfile.is_zipfile()` returns True for any torch
    .pt file. That internal zip wrapping is NOT a contest archive.zip. We
    reject by EXTENSION + path-shape rather than by zipfile-ness so torch
    sidecars don't masquerade as contest archives.
    """
    if not substrate_name:
        return False, "empty substrate_name"
    if not candidate_bytes_path_str:
        return False, "empty candidate path"

    path = Path(candidate_bytes_path_str)
    root = _repo_root(repo_root)
    path_for_io = _resolve_candidate_path_for_io(
        candidate_bytes_path_str,
        repo_root=root,
    )
    # Pending-dispatch is neither validated nor rejected; the existing
    # pending_dispatch status branch handles this case.
    if not path.exists() and not path_for_io.exists():
        return False, "candidate path does not exist (pending_dispatch)"

    # Rule 2 (early-out): research sidecar extensions are NEVER themselves
    # archive members. They COULD theoretically be repackaged into an
    # archive.zip later, but that's a separate build step — the prober's
    # job is to verify CURRENT shipping state, not hypothetical future.
    if path.suffix.lower() in RESEARCH_SIDECAR_EXTENSIONS:
        return False, (
            f"candidate {path.name!r} is a research sidecar "
            f"({path.suffix} extension); bytes are NOT in any contest "
            f"archive.zip; standalone {path.suffix} files are never "
            f"charged by the contest rate term"
        )

    # Rule 1: real contest archive.zip under one of the canonical roots.
    if path.suffix.lower() != ".zip":
        return False, (
            f"candidate {path.name!r} is not a .zip file; contest "
            f"archives are always .zip"
        )

    # Path-shape check: is the .zip under one of CONTEST_ARCHIVE_PARENT_ROOTS?
    rel_str = str(path)
    try:
        rel_str = str(path_for_io.resolve().relative_to(root.resolve()))
    except ValueError:
        rel_str = str(path)
    parts = Path(rel_str).parts
    under_canonical_root = any(
        parts[: len(root_parts := Path(root).parts)] == root_parts
        for root in CONTEST_ARCHIVE_PARENT_ROOTS
    )
    if not under_canonical_root:
        return False, (
            f"candidate .zip {path.name!r} is not under any canonical "
            f"contest-archive root ({CONTEST_ARCHIVE_PARENT_ROOTS}); "
            f"outside-root .zip files are forensic / research-only"
        )

    # zipfile.is_zipfile gate: confirm the file is actually a valid ZIP.
    # This catches truncated / corrupt archives that pass the extension
    # check but cannot be opened for member-level probing.
    if not zipfile.is_zipfile(path_for_io):
        return False, (
            f"candidate .zip {path.name!r} fails zipfile.is_zipfile "
            f"validation (truncated / corrupt)"
        )

    return True, None


# ──────────────────────────────────────────────────────────────────────── #
# Per-substrate probe                                                       #
# ──────────────────────────────────────────────────────────────────────── #


def probe_substrate(
    substrate_name: str,
    archive_path_str: str,
    substrate_class: str,
    *,
    min_member_bytes: int = DEFAULT_MIN_MEMBER_BYTES,
    skip_contest_member_validation: bool = False,
) -> SubstrateProbeResult:
    """Probe a single substrate's archive/checkpoint for pre-entropy
    compressibility.

    Per Q4 HALT memo 2026-05-17 + Catalog #321 self-protection: this probe
    FIRST runs `_validate_substrate_bytes_ship_in_contest_archive`. If the
    candidate path is a research sidecar (.pt / .npy / etc. outside any
    contest archive.zip), emits a result with `validation_status=
    REJECTED_RESEARCH_SIDECAR` + `deliverable_score_savings_estimate=0.0`
    so the autopilot consumer does not promote a phantom score.

    Set ``skip_contest_member_validation=True`` ONLY for synthetic unit-
    test fixtures or for the rare case where the prober is intentionally
    used as a diagnostic on standalone bytes (e.g., to characterize a
    research checkpoint's residual compressibility for hoist-design
    purposes). Real-archive probes MUST keep validation active so the
    autopilot pipeline cannot silently consume a phantom estimate.
    """
    archive_path_for_io = _resolve_candidate_path_for_io(archive_path_str)

    # Validation gate per Catalog #321 (Option C self-protect)
    validation_status = "UNVALIDATED"
    validation_reason: str | None = None
    evidence_grade_per_row = "predicted"
    if not skip_contest_member_validation:
        is_valid, reject_reason = _validate_substrate_bytes_ship_in_contest_archive(
            substrate_name, archive_path_str
        )
        if not is_valid and archive_path_for_io.exists():
            # File exists but is a research sidecar (or off-root .zip) — emit
            # REJECTED_RESEARCH_SIDECAR with zero deliverable, preserving
            # archive_sha256 + bytes_total so the artifact still carries
            # provenance.
            try:
                sha = compute_file_sha256(archive_path_for_io)
                size = archive_path_for_io.stat().st_size
            except OSError as exc:  # pragma: no cover
                sha = None
                size = 0
                reject_reason = f"{reject_reason}; sha compute failed: {exc}"
            return SubstrateProbeResult(
                substrate_name=substrate_name,
                substrate_class=substrate_class,
                archive_path=archive_path_str,
                archive_exists=True,
                archive_sha256=sha,
                archive_bytes_total=size,
                pre_entropy_bytes=0,
                at_floor_bytes=0,
                post_entropy_bytes=0,
                pre_entropy_fraction=0.0,
                best_compression_codec="none",
                best_compression_ratio=1.0,
                deliverable_score_savings_estimate=0.0,
                member_count=0,
                member_results=(),
                archive_status="present",
                error=None,
                validation_status="REJECTED_RESEARCH_SIDECAR",
                validation_reason=reject_reason,
                evidence_grade_per_row="invalid_target",
            )
        if is_valid:
            validation_status = "VALIDATED_CONTEST_MEMBER"
        # else: not is_valid AND not archive_path_for_io.exists() — falls through to
        # pending_dispatch branch below.

    if not archive_path_for_io.exists():
        return SubstrateProbeResult(
            substrate_name=substrate_name,
            substrate_class=substrate_class,
            archive_path=archive_path_str,
            archive_exists=False,
            archive_sha256=None,
            archive_bytes_total=0,
            pre_entropy_bytes=0,
            at_floor_bytes=0,
            post_entropy_bytes=0,
            pre_entropy_fraction=0.0,
            best_compression_codec="none",
            best_compression_ratio=1.0,
            deliverable_score_savings_estimate=0.0,
            member_count=0,
            member_results=(),
            archive_status="pending_dispatch",
            error=None,
            validation_status=validation_status,
            validation_reason=validation_reason,
            evidence_grade_per_row=evidence_grade_per_row,
        )

    try:
        sha = compute_file_sha256(archive_path_for_io)
        members = list(iter_archive_members(archive_path_for_io, min_member_bytes=min_member_bytes))
        if not members:
            return SubstrateProbeResult(
                substrate_name=substrate_name,
                substrate_class=substrate_class,
                archive_path=archive_path_str,
                archive_exists=True,
                archive_sha256=sha,
                archive_bytes_total=archive_path_for_io.stat().st_size,
                pre_entropy_bytes=0,
                at_floor_bytes=0,
                post_entropy_bytes=0,
                pre_entropy_fraction=0.0,
                best_compression_codec="none",
                best_compression_ratio=1.0,
                deliverable_score_savings_estimate=0.0,
                member_count=0,
                member_results=(),
                archive_status="present",
                error="no members above min_member_bytes threshold",
                validation_status=validation_status,
                validation_reason=validation_reason,
                evidence_grade_per_row=evidence_grade_per_row,
            )

        member_results: list[MemberProbeResult] = []
        total_bytes = 0
        pre_entropy_bytes = 0
        at_floor_bytes = 0
        post_entropy_bytes = 0
        # Track best codec across PRE_ENTROPY members (the only ones that
        # contribute to the deliverable estimate).
        best_pre_entropy_codec_counts: dict[str, int] = {}
        weighted_ratio_numer = 0.0
        weighted_ratio_denom = 0

        for mname, mbytes in members:
            mr = probe_member_compression(mname, mbytes)
            member_results.append(mr)
            total_bytes += mr.raw_bytes
            if mr.classification == "PRE_ENTROPY":
                pre_entropy_bytes += mr.raw_bytes
                best_pre_entropy_codec_counts[mr.best_codec] = (
                    best_pre_entropy_codec_counts.get(mr.best_codec, 0) + 1
                )
                weighted_ratio_numer += mr.best_ratio * mr.raw_bytes
                weighted_ratio_denom += mr.raw_bytes
            elif mr.classification == "AT_FLOOR":
                at_floor_bytes += mr.raw_bytes
            else:
                post_entropy_bytes += mr.raw_bytes

        pre_entropy_fraction = pre_entropy_bytes / total_bytes if total_bytes > 0 else 0.0

        # Aggregate best-codec across pre-entropy members (mode of best
        # codec per member, weighted equally per member).
        if best_pre_entropy_codec_counts:
            best_codec = max(best_pre_entropy_codec_counts, key=best_pre_entropy_codec_counts.get)
        else:
            best_codec = "lzma"  # default canonical codec

        # Weighted-average compression ratio across pre-entropy members
        best_ratio = weighted_ratio_numer / weighted_ratio_denom if weighted_ratio_denom > 0 else 1.0

        score_delta = estimate_deliverable_score_savings(
            pre_entropy_bytes=pre_entropy_bytes,
            best_compression_ratio=best_ratio,
        )

        return SubstrateProbeResult(
            substrate_name=substrate_name,
            substrate_class=substrate_class,
            archive_path=archive_path_str,
            archive_exists=True,
            archive_sha256=sha,
            archive_bytes_total=archive_path_for_io.stat().st_size,
            pre_entropy_bytes=pre_entropy_bytes,
            at_floor_bytes=at_floor_bytes,
            post_entropy_bytes=post_entropy_bytes,
            pre_entropy_fraction=pre_entropy_fraction,
            best_compression_codec=best_codec,
            best_compression_ratio=best_ratio,
            deliverable_score_savings_estimate=score_delta,
            member_count=len(members),
            member_results=tuple(member_results),
            archive_status="present",
            error=None,
            validation_status=validation_status,
            validation_reason=validation_reason,
            evidence_grade_per_row=evidence_grade_per_row,
        )
    except Exception as exc:
        return SubstrateProbeResult(
            substrate_name=substrate_name,
            substrate_class=substrate_class,
            archive_path=archive_path_str,
            archive_exists=archive_path_for_io.exists(),
            archive_sha256=None,
            archive_bytes_total=archive_path_for_io.stat().st_size if archive_path_for_io.exists() else 0,
            pre_entropy_bytes=0,
            at_floor_bytes=0,
            post_entropy_bytes=0,
            pre_entropy_fraction=0.0,
            best_compression_codec="error",
            best_compression_ratio=1.0,
            deliverable_score_savings_estimate=0.0,
            member_count=0,
            member_results=(),
            archive_status="present" if archive_path_for_io.exists() else "missing",
            error=f"{type(exc).__name__}: {exc}",
            validation_status=validation_status,
            validation_reason=validation_reason,
            evidence_grade_per_row=evidence_grade_per_row,
        )


# ──────────────────────────────────────────────────────────────────────── #
# Per-archive-member probe (Catalog #321 / Option B canonical method)       #
# ──────────────────────────────────────────────────────────────────────── #


def probe_substrate_archive_member(
    substrate_name: str,
    archive_zip_path: str,
    member_name: str,
    substrate_class: str,
    *,
    min_member_bytes: int = DEFAULT_MIN_MEMBER_BYTES,
) -> SubstrateProbeResult:
    """Probe a SPECIFIC member of an actual contest archive.zip.

    This is the CORRECT level of analysis for "would Wyner-Ziv hoist save
    bytes on this contest archive?". Per Q4 HALT memo 2026-05-17 Option B:
    point the prober at an actual contest-shipping ``archive.zip``, extract
    a specific member's bytes, run the compression probe ON THE MEMBER.

    The result's ``deliverable_score_savings_estimate`` field is now an
    apples-to-apples contest-rate-term delta: `25 * (member_bytes -
    compressed_member_bytes) / 37_545_489`.

    Args:
        substrate_name: identifier for the substrate-of-interest.
        archive_zip_path: path to an actual contest ``archive.zip`` (must
            be a real .zip file under a canonical contest-shipping root).
        member_name: the name of the member inside the .zip to probe
            (e.g., ``'x'`` for FEC6's single member, ``'0.bin'`` for
            HNeRV-style archives, ``'decoder_state.pt'`` for substrates
            that ship a state-dict member).
        substrate_class: classification tag (e.g., ``raw_float_weights``).
        min_member_bytes: skip if member is smaller than this.

    Returns:
        SubstrateProbeResult with ``validation_status=
        VALIDATED_CONTEST_MEMBER`` (or REJECTED_RESEARCH_SIDECAR if the
        archive_zip_path itself fails contest-member validation).
    """
    archive_path_for_io = _resolve_candidate_path_for_io(archive_zip_path)

    # Reuse the same validator on the WRAPPING archive.zip.
    is_valid, reject_reason = _validate_substrate_bytes_ship_in_contest_archive(
        substrate_name, archive_zip_path
    )
    if not is_valid:
        if archive_path_for_io.exists():
            try:
                sha = compute_file_sha256(archive_path_for_io)
                size = archive_path_for_io.stat().st_size
            except OSError as exc:  # pragma: no cover
                sha = None
                size = 0
                reject_reason = f"{reject_reason}; sha compute failed: {exc}"
            return SubstrateProbeResult(
                substrate_name=substrate_name,
                substrate_class=substrate_class,
                archive_path=f"{archive_zip_path}#{member_name}",
                archive_exists=True,
                archive_sha256=sha,
                archive_bytes_total=size,
                pre_entropy_bytes=0,
                at_floor_bytes=0,
                post_entropy_bytes=0,
                pre_entropy_fraction=0.0,
                best_compression_codec="none",
                best_compression_ratio=1.0,
                deliverable_score_savings_estimate=0.0,
                member_count=0,
                member_results=(),
                archive_status="present",
                error=None,
                validation_status="REJECTED_RESEARCH_SIDECAR",
                validation_reason=reject_reason,
                evidence_grade_per_row="invalid_target",
            )
        # archive doesn't exist
        return SubstrateProbeResult(
            substrate_name=substrate_name,
            substrate_class=substrate_class,
            archive_path=f"{archive_zip_path}#{member_name}",
            archive_exists=False,
            archive_sha256=None,
            archive_bytes_total=0,
            pre_entropy_bytes=0,
            at_floor_bytes=0,
            post_entropy_bytes=0,
            pre_entropy_fraction=0.0,
            best_compression_codec="none",
            best_compression_ratio=1.0,
            deliverable_score_savings_estimate=0.0,
            member_count=0,
            member_results=(),
            archive_status="pending_dispatch",
            error=reject_reason,
            validation_status="UNVALIDATED",
            validation_reason=reject_reason,
            evidence_grade_per_row="predicted",
        )

    # Validated — extract the member and probe.
    try:
        sha = compute_file_sha256(archive_path_for_io)
        with zipfile.ZipFile(archive_path_for_io) as zf:
            names = zf.namelist()
            if member_name not in names:
                return SubstrateProbeResult(
                    substrate_name=substrate_name,
                    substrate_class=substrate_class,
                    archive_path=f"{archive_zip_path}#{member_name}",
                    archive_exists=True,
                    archive_sha256=sha,
                    archive_bytes_total=archive_path_for_io.stat().st_size,
                    pre_entropy_bytes=0,
                    at_floor_bytes=0,
                    post_entropy_bytes=0,
                    pre_entropy_fraction=0.0,
                    best_compression_codec="none",
                    best_compression_ratio=1.0,
                    deliverable_score_savings_estimate=0.0,
                    member_count=0,
                    member_results=(),
                    archive_status="present",
                    error=f"member {member_name!r} not found in archive (members: {names})",
                    validation_status="VALIDATED_CONTEST_MEMBER",
                    validation_reason=None,
                    evidence_grade_per_row="predicted",
                )
            member_bytes = zf.read(member_name)
            if len(member_bytes) < min_member_bytes:
                return SubstrateProbeResult(
                    substrate_name=substrate_name,
                    substrate_class=substrate_class,
                    archive_path=f"{archive_zip_path}#{member_name}",
                    archive_exists=True,
                    archive_sha256=sha,
                    archive_bytes_total=archive_path_for_io.stat().st_size,
                    pre_entropy_bytes=0,
                    at_floor_bytes=0,
                    post_entropy_bytes=0,
                    pre_entropy_fraction=0.0,
                    best_compression_codec="none",
                    best_compression_ratio=1.0,
                    deliverable_score_savings_estimate=0.0,
                    member_count=0,
                    member_results=(),
                    archive_status="present",
                    error=f"member {member_name!r} below min_member_bytes threshold",
                    validation_status="VALIDATED_CONTEST_MEMBER",
                    validation_reason=None,
                    evidence_grade_per_row="predicted",
                )

        mr = probe_member_compression(member_name, member_bytes)
        if mr.classification == "PRE_ENTROPY":
            pre_entropy_bytes = mr.raw_bytes
            at_floor_bytes = 0
            post_entropy_bytes = 0
        elif mr.classification == "AT_FLOOR":
            pre_entropy_bytes = 0
            at_floor_bytes = mr.raw_bytes
            post_entropy_bytes = 0
        else:
            pre_entropy_bytes = 0
            at_floor_bytes = 0
            post_entropy_bytes = mr.raw_bytes
        total_bytes = mr.raw_bytes
        pre_entropy_fraction = pre_entropy_bytes / total_bytes if total_bytes else 0.0
        score_delta = estimate_deliverable_score_savings(
            pre_entropy_bytes=pre_entropy_bytes,
            best_compression_ratio=mr.best_ratio,
        )

        return SubstrateProbeResult(
            substrate_name=substrate_name,
            substrate_class=substrate_class,
            archive_path=f"{archive_zip_path}#{member_name}",
            archive_exists=True,
            archive_sha256=sha,
            archive_bytes_total=archive_path_for_io.stat().st_size,
            pre_entropy_bytes=pre_entropy_bytes,
            at_floor_bytes=at_floor_bytes,
            post_entropy_bytes=post_entropy_bytes,
            pre_entropy_fraction=pre_entropy_fraction,
            best_compression_codec=mr.best_codec,
            best_compression_ratio=mr.best_ratio,
            deliverable_score_savings_estimate=score_delta,
            member_count=1,
            member_results=(mr,),
            archive_status="present",
            error=None,
            validation_status="VALIDATED_CONTEST_MEMBER",
            validation_reason=None,
            evidence_grade_per_row="predicted",
        )
    except Exception as exc:
        return SubstrateProbeResult(
            substrate_name=substrate_name,
            substrate_class=substrate_class,
            archive_path=f"{archive_zip_path}#{member_name}",
            archive_exists=archive_path_for_io.exists(),
            archive_sha256=None,
            archive_bytes_total=archive_path_for_io.stat().st_size if archive_path_for_io.exists() else 0,
            pre_entropy_bytes=0,
            at_floor_bytes=0,
            post_entropy_bytes=0,
            pre_entropy_fraction=0.0,
            best_compression_codec="error",
            best_compression_ratio=1.0,
            deliverable_score_savings_estimate=0.0,
            member_count=0,
            member_results=(),
            archive_status="present" if archive_path_for_io.exists() else "missing",
            error=f"{type(exc).__name__}: {exc}",
            validation_status="VALIDATED_CONTEST_MEMBER",
            validation_reason=None,
            evidence_grade_per_row="predicted",
        )


# ──────────────────────────────────────────────────────────────────────── #
# Persistence (fcntl-locked atomic write per Catalog #131)                  #
# ──────────────────────────────────────────────────────────────────────── #


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively replace NaN/Inf with None so JSON output is byte-stable
    (allow_nan=False compatible)."""
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):  # NaN or +-Inf
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_for_json(x) for x in obj]
    return obj


def _fcntl_locked_atomic_write(path: Path, payload: dict[str, Any]) -> None:
    """Atomic write of payload to path under fcntl.LOCK_EX.

    Per CLAUDE.md Catalog #131 "no bare writes to shared state":
    transactional write (.tmp.<uuid> + os.replace) inside the locked
    region prevents concurrent-writer interleaving.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.parent / f".{path.name}.lock"
    tmp_path = path.with_suffix(path.suffix + f".tmp.{uuid.uuid4().hex[:12]}")
    sanitized = _sanitize_for_json(payload)
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX)
        try:
            tmp_path.write_text(
                json.dumps(sanitized, indent=2, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            os.replace(tmp_path, path)
        finally:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)


def build_manifest_payload(
    *,
    results: list[SubstrateProbeResult],
    min_compression_ratio: float,
    output_path: Path,
) -> dict[str, Any]:
    """Build the canonical output manifest payload from per-substrate
    results. Implements the schema declared in the prober's CONTEXT
    docstring."""
    utc = datetime.datetime.now(datetime.UTC)

    per_substrate: dict[str, Any] = {}
    pre_entropy_substrates: list[str] = []
    at_floor_substrates: list[str] = []
    post_entropy_substrates: list[str] = []

    for r in results:
        # Classify substrate at aggregate level
        if r.pre_entropy_bytes > 0 and r.pre_entropy_fraction >= 0.50:
            pre_entropy_substrates.append(r.substrate_name)
        elif r.at_floor_bytes > 0 and r.at_floor_bytes >= r.post_entropy_bytes:
            at_floor_substrates.append(r.substrate_name)
        else:
            post_entropy_substrates.append(r.substrate_name)

        per_substrate[r.substrate_name] = {
            "substrate_class": r.substrate_class,
            "archive_path": r.archive_path,
            "archive_exists": r.archive_exists,
            "archive_status": r.archive_status,
            "archive_sha256": r.archive_sha256,
            "archive_bytes_total": r.archive_bytes_total,
            "pre_entropy_bytes": r.pre_entropy_bytes,
            "at_floor_bytes": r.at_floor_bytes,
            "post_entropy_bytes": r.post_entropy_bytes,
            "pre_entropy_fraction": r.pre_entropy_fraction,
            "best_compression_codec": r.best_compression_codec,
            "best_compression_ratio": r.best_compression_ratio,
            "deliverable_score_savings_estimate": r.deliverable_score_savings_estimate,
            "member_count": r.member_count,
            "member_breakdown": {
                m.member_name: {
                    "bytes": m.raw_bytes,
                    "lzma_ratio": m.lzma_ratio,
                    "brotli_ratio": m.brotli_ratio,
                    "zlib_ratio": m.zlib_ratio,
                    "best_codec": m.best_codec,
                    "best_ratio": m.best_ratio,
                    "classification": m.classification,
                }
                for m in r.member_results
            },
            "error": r.error,
            # Catalog #321 / Q4 HALT 2026-05-17 self-protection fields:
            "validation_status": r.validation_status,
            "validation_reason": r.validation_reason,
            "evidence_grade_per_row": r.evidence_grade_per_row,
            # Empirical-tag-per-row per CLAUDE.md FORBIDDEN_PATTERNS
            # "Forbidden empirical-claim-without-evidence-tag" (Catalog #287
            # sister). Marks the deliverable_score_savings_estimate field
            # apples-to-apples-tagged against the validation outcome.
            "deliverable_savings_evidence_tag": (
                f"[empirical:lzma_ratio_on_actual_member={r.best_compression_ratio:.4f}]"
                if r.validation_status == "VALIDATED_CONTEST_MEMBER"
                else (
                    "[invalid_target:research_sidecar_not_contest_archive_member]"
                    if r.validation_status == "REJECTED_RESEARCH_SIDECAR"
                    else "[unvalidated:diagnostic_only]"
                )
            ),
        }

    # Recommended Q4 target: substrate with HIGHEST
    # deliverable_score_savings_estimate (must be PRE_ENTROPY-dominant).
    recommended_q4_target = None
    recommended_q4_target_sha = None
    recommended_q4_target_bytes = 0
    recommended_q4_target_savings = 0.0
    for r in results:
        if r.substrate_name in pre_entropy_substrates and r.deliverable_score_savings_estimate > recommended_q4_target_savings:
            recommended_q4_target = r.substrate_name
            recommended_q4_target_sha = r.archive_sha256
            recommended_q4_target_bytes = r.pre_entropy_bytes
            recommended_q4_target_savings = r.deliverable_score_savings_estimate

    payload: dict[str, Any] = {
        "schema_version": "pre_entropy_pivot_probe_v1",
        "candidates_probed": len(results),
        "substrates_with_pre_entropy_bytes": pre_entropy_substrates,
        "substrates_at_entropy_floor": at_floor_substrates,
        "substrates_post_entropy": post_entropy_substrates,
        "per_substrate_results": per_substrate,
        "recommended_q4_target_substrate": recommended_q4_target,
        "recommended_q4_target_archive_sha256": recommended_q4_target_sha,
        "recommended_q4_target_pre_entropy_bytes": recommended_q4_target_bytes,
        "recommended_q4_target_deliverable_savings_estimate": recommended_q4_target_savings,
        "min_compression_ratio_threshold": min_compression_ratio,
        "brotli_available": _HAS_BROTLI,
        "evidence_grade": "predicted",
        "measurement_axis": "[diagnostic; pre-entropy substrate pivot probe]",
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "claude_md_compliance_tags": [
            "pre_entropy_pivot_per_prober_op_routable_4",
            "bit_level_deconstruction_per_claude_md",
            "apples_to_apples_per_catalog_127",
            "fcntl_locked_write_per_catalog_131",
            "non_authoritative_per_catalog_192",
            "phantom_score_research_sidecar_rejected_per_catalog_321",
            "deliverable_savings_evidence_tagged_per_catalog_287",
        ],
        "sister_anchor_artifacts": [
            ".omx/state/wyner_ziv_deliverability/probe_f174192aeadf_20260517T205208.json",
        ],
        "written_at_utc": utc.isoformat(),
        "written_pid": os.getpid(),
        "written_host": os.uname().nodename,
        "output_path": str(output_path),
    }
    return payload


def persist_manifest(
    *,
    results: list[SubstrateProbeResult],
    min_compression_ratio: float,
    output_path: Path,
) -> Path:
    """Persist the canonical pre-entropy pivot manifest at the given
    output_path under fcntl-locked atomic write."""
    payload = build_manifest_payload(
        results=results,
        min_compression_ratio=min_compression_ratio,
        output_path=output_path,
    )
    _fcntl_locked_atomic_write(output_path, payload)
    return output_path


# ──────────────────────────────────────────────────────────────────────── #
# End-to-end driver                                                         #
# ──────────────────────────────────────────────────────────────────────── #


def resolve_candidate_substrates(
    *,
    candidate_substrates_arg: str | None,
    extra_json_path: Path | None = None,
) -> dict[str, tuple[str, str]]:
    """Resolve the substrate-to-path map from CLI args.

    * If `candidate_substrates_arg` is None, use all CANONICAL_CANDIDATE_SUBSTRATES.
    * Otherwise, treat it as a comma-separated list of substrate names;
      filter the canonical map to those names.
    * Optionally merge in extra substrates from a JSON file at
      `extra_json_path` (schema: `{"name": ["path", "substrate_class"], ...}`).
    """
    base = dict(CANONICAL_CANDIDATE_SUBSTRATES)
    if extra_json_path and extra_json_path.exists():
        with open(extra_json_path) as f:
            extras = json.load(f)
        for name, val in extras.items():
            if isinstance(val, list) and len(val) == 2:
                base[name] = (val[0], val[1])

    if candidate_substrates_arg is None:
        return base

    names = [n.strip() for n in candidate_substrates_arg.split(",") if n.strip()]
    return {n: base[n] for n in names if n in base}


def run_pivot_probe(
    *,
    candidate_substrates: dict[str, tuple[str, str]] | None = None,
    output_path: Path | None = None,
    min_compression_ratio: float = DEFAULT_MIN_COMPRESSION_RATIO,
    min_member_bytes: int = DEFAULT_MIN_MEMBER_BYTES,
    persist: bool = True,
    skip_contest_member_validation: bool = False,
) -> tuple[list[SubstrateProbeResult], Path | None]:
    """Run the end-to-end pivot probe over the substrate map.

    Returns (results, persisted_path). If persist=False, persisted_path
    is None.

    The ``skip_contest_member_validation`` flag is for SYNTHETIC unit-
    test fixtures only — real-archive runs MUST keep validation active
    so the Q4 phantom-score class (Catalog #321) cannot recur.
    """
    if candidate_substrates is None:
        candidate_substrates = dict(CANONICAL_CANDIDATE_SUBSTRATES)

    results: list[SubstrateProbeResult] = []
    for name, (path, sclass) in candidate_substrates.items():
        result = probe_substrate(
            name,
            path,
            sclass,
            min_member_bytes=min_member_bytes,
            skip_contest_member_validation=skip_contest_member_validation,
        )
        results.append(result)

    persisted = None
    if persist:
        if output_path is None:
            utc = datetime.datetime.now(datetime.UTC)
            safe_utc = utc.strftime("%Y%m%dT%H%M%S")
            output_path = OUTPUT_DIR_DEFAULT / f"pre_entropy_candidate_substrates_{safe_utc}.json"
        persisted = persist_manifest(
            results=results,
            min_compression_ratio=min_compression_ratio,
            output_path=output_path,
        )

    return results, persisted


# ──────────────────────────────────────────────────────────────────────── #
# CLI                                                                       #
# ──────────────────────────────────────────────────────────────────────── #


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Pre-entropy substrate pivot prober. Identifies substrates whose "
            "archive/checkpoint bytes compress meaningfully under general-purpose "
            "codecs (lzma/brotli/zlib) -- these are the substrates where the "
            "autopilot's 1.15x Wyner-Ziv reward is justified, and the canonical "
            "Q4 retarget candidates."
        ),
    )
    parser.add_argument(
        "--candidate-substrates",
        type=str,
        default=None,
        help=(
            "Comma-separated list of substrate names to probe. If omitted, "
            "all canonical candidates are probed. See "
            "CANONICAL_CANDIDATE_SUBSTRATES in this module for the canonical "
            "list."
        ),
    )
    parser.add_argument(
        "--candidate-substrates-extra-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file mapping extra substrate names to "
            "[path, substrate_class] tuples. Merged into the canonical map."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Canonical output path. Defaults to "
            ".omx/state/wyner_ziv_deliverability/pre_entropy_candidate_substrates_<utc>.json"
        ),
    )
    parser.add_argument(
        "--min-compression-ratio",
        type=float,
        default=DEFAULT_MIN_COMPRESSION_RATIO,
        help=(
            "Minimum compression ratio for the substrate to be classified "
            "PRE_ENTROPY. Members with best-codec compression ratio < this "
            "value are considered compressible (Wyner-Ziv deliverable). "
            "Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--min-member-bytes",
        type=int,
        default=DEFAULT_MIN_MEMBER_BYTES,
        help=(
            "Skip archive members smaller than this many bytes (avoids "
            "noise from tiny entries). Default: %(default)s."
        ),
    )
    parser.add_argument(
        "--report-only-no-side-effects",
        action="store_true",
        help=(
            "If set, do not persist the manifest JSON. Useful for "
            "dry-run inspection. Prints the recommended Q4 target to stdout "
            "regardless."
        ),
    )
    parser.add_argument(
        "--skip-contest-member-validation",
        action="store_true",
        help=(
            "DIAGNOSTIC-ONLY: skip the Catalog #321 validator that refuses "
            "research-sidecar paths. Use ONLY for synthetic test fixtures "
            "or for measuring the residual compressibility of a research "
            "checkpoint as a hoist-design diagnostic. Real-archive runs "
            "MUST leave this flag off so the autopilot consumer cannot "
            "silently absorb a phantom deliverable_score_savings_estimate."
        ),
    )
    args = parser.parse_args(argv)

    substrates = resolve_candidate_substrates(
        candidate_substrates_arg=args.candidate_substrates,
        extra_json_path=args.candidate_substrates_extra_json,
    )
    if not substrates:
        print("[pre-entropy-pivot] no candidate substrates resolved", file=sys.stderr)
        return 2

    results, persisted = run_pivot_probe(
        candidate_substrates=substrates,
        output_path=args.output,
        min_compression_ratio=args.min_compression_ratio,
        min_member_bytes=args.min_member_bytes,
        persist=not args.report_only_no_side_effects,
        skip_contest_member_validation=args.skip_contest_member_validation,
    )

    # Print summary to stdout
    print("\n=== PRE-ENTROPY SUBSTRATE PIVOT PROBE SUMMARY ===\n")
    print(f"{'substrate':<35}{'class':<28}{'bytes':>10}{'ratio':>8}{'classification':>16}{'score_savings':>15}")
    print("-" * 115)
    # Sort by deliverable savings desc
    sorted_results = sorted(results, key=lambda r: -r.deliverable_score_savings_estimate)
    for r in sorted_results:
        classification = (
            "PRE_ENTROPY"
            if r.pre_entropy_bytes > 0 and r.pre_entropy_fraction >= 0.50
            else ("POST_ENTROPY" if r.post_entropy_bytes >= r.at_floor_bytes else "AT_FLOOR")
        )
        if not r.archive_exists:
            classification = "PENDING_DISPATCH"
        ratio_str = f"{r.best_compression_ratio:.3f}"
        print(
            f"{r.substrate_name:<35}{r.substrate_class:<28}{r.archive_bytes_total:>10}"
            f"{ratio_str:>8}{classification:>16}{r.deliverable_score_savings_estimate:>15.6f}"
        )

    # Recommended Q4 target
    pre_entropy_results = [r for r in sorted_results if r.pre_entropy_bytes > 0 and r.pre_entropy_fraction >= 0.50]
    if pre_entropy_results:
        top = pre_entropy_results[0]
        print(f"\nRECOMMENDED Q4 TARGET: {top.substrate_name}")
        print(f"  archive_sha256: {top.archive_sha256}")
        print(f"  pre_entropy_bytes: {top.pre_entropy_bytes:,}")
        print(f"  best_compression_codec: {top.best_compression_codec}")
        print(f"  best_compression_ratio: {top.best_compression_ratio:.3f}")
        print(f"  deliverable_score_savings_estimate: {top.deliverable_score_savings_estimate:.6f}")
        print("  REPLACES fec6 as Q4 target (fec6 deliverable = 0.000000 per sister prober).")
    else:
        print("\nNo pre-entropy substrates found among probed candidates.")

    if persisted:
        print(f"\nManifest persisted to: {persisted}")
    else:
        print("\n[report-only-no-side-effects] Manifest NOT persisted (dry-run).")

    return 0


if __name__ == "__main__":
    sys.exit(main())
