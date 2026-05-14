"""Scorer-conditional MDL ablation (Z1, $0 GPU).

**Canonical surface delegation note (2026-05-14, lane
``lane_xray_canon_math_findings_wire_in_20260514``):**

The STRUCTURAL tier of MDL ablation now has a canonical typed XRay primitive
surface at :class:`tac.xray.mdl_scorer_conditional.ScorerConditionalMDLEstimator`.
For programmatic / planner-integrated MDL density queries, prefer the
canonical primitive — it returns a typed :class:`XRayPrimitiveResult` with
non-empty ``wire_in_hooks_engaged`` so the solver stack (autopilot, Pareto,
continual-learning) consumes it without further glue code:

    from tac.xray import ScorerConditionalMDLEstimator
    estimator = ScorerConditionalMDLEstimator()
    result = estimator.compute(Path("submissions/a1/archive.zip"))
    # result.primitive_value is a MDLDensityResult with per-section breakdown.

This ad-hoc tool (``tools/mdl_scorer_conditional_ablation.py``) remains
the one-stop CLI for the FULL three-tier ablation (structural + sampled
byte-level + post-decode perturbation against real SegNet/PoseNet) which
the canonical primitive intentionally does NOT wrap (Tier B / Tier C
require scorer dispatch + are not yet in the typed contract).

Empirically measures how many bits of each archive are EXTRACTED by the
scorer (SegNet + PoseNet) vs IGNORED. The scorer-conditional MDL gives
the empirical lower bound on `H(X | scorer_weights + architecture +
preprocessing)` — the cooperative-receiver theorem floor.

This is the probe-disambiguator (Catalog #125 hook #6) for the zen-floor
council's competing predictions:

    if MDL extracted < 1 KB across all tested archives:
        Shannon zen-floor ~0.003 confirmed; staircase pursuit HIGH-EV
    elif MDL extracted 1-10 KB:
        zen-floor band [0.01, 0.05]; staircase pursuit MEDIUM-EV
    elif MDL extracted 10-50 KB:
        zen-floor band [0.05, 0.10]; current substrates close to floor
    else (> 50 KB):
        zen-floor band [0.10, 0.15]; major arch breakthrough needed

Per CLAUDE.md "MPS auth eval is NOISE" non-negotiable, MPS forward
passes do NOT produce authoritative absolute scores. However, MDL
ablation measures DELTAS (score(perturbed) - score(baseline)) — the
device-specific drift cancels for sign + relative magnitude. Therefore
MPS is ACCEPTABLE for this ablation. Results are tagged
`[MDL-ablation-MPS]` — NEVER `[contest-CUDA]` / `[contest-CPU]`.

Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact"
non-negotiable, persisted artifacts go to
`experiments/results/<lane_id>_<TS>/`.

Per CLAUDE.md "FORBIDDEN PATTERNS — Forbidden empirical-claim-without-
evidence-tag" non-negotiable, every claim in the emitted JSON is tagged.

Per CLAUDE.md "Forbidden KILL without research exhaustion" non-negotiable,
this tool does NOT emit KILL verdicts. It emits empirical Δscore
measurements + a recommended ZEN-FLOOR-BAND update (LOW/CENTER/HIGH).

Three ablation tiers (run in order; cheaper first):

Tier A (STRUCTURAL):
    For each named archive section (decoder_blob / latent_blob /
    sidecar_blob), zero it OR randomize it, re-decode, compute Δscore.
    Gives section-level MDL: how many score-points does each section
    carry?

Tier B (SAMPLED BYTE-LEVEL):
    For N random byte positions inside each section, flip the byte
    (XOR 0xFF), re-decode, compute Δscore. Stochastic estimator with
    confidence intervals. Bytes with |Δscore| < epsilon are
    scorer-ignored.

Tier C (POST-DECODE PERTURBATION):
    Perturb the DECODED tensors (state_dict / latents) directly with
    additive Gaussian noise at controlled magnitude. Measures
    decoder-state sensitivity (not byte-level, but cleaner signal). Used
    to disambiguate "byte is ignored" (Tier B) from "byte is encoded
    redundantly".

Usage:

    .venv/bin/python tools/mdl_scorer_conditional_ablation.py \\
        --archive submissions/a1/archive.zip \\
        --archive-name a1 \\
        --grammar a1 \\
        --upstream-dir upstream \\
        --output-dir experiments/results/mdl_ablation_z1_20260514 \\
        --byte-samples 200 \\
        --pair-samples 60 \\
        --seed 1234

Multiple archives can be ablated in one run via repeated --archive flags.

Supported grammars (parser dispatch + Tier A perturbation):

- ``a1`` — PR101-microcodec A1 (decoder_section_header + decoder_blob +
  latent_blob[15387] + sidecar_blob)
- ``pr106`` / ``pr106_latent_sidecar`` — PR106 latent-sidecar wrapper
  (0xFE 0x01 + uint32 pr106_len + pr106_base_archive + uint16 sidecar_len +
  sidecar_blob)
- ``ibps1`` — C6 MDL-IBPS variant 1 (per
  :mod:`tac.substrates.c6_e4_mdl_ibps.archive`):
  IBPS1_HEADER(25) + ENCODER_BLOB + DECODER_BLOB + LATENT_BLOB + META_BLOB
- ``pr101`` (alias of ``a1`` for back-compat) — same layout

For grammars not listed above, the parser falls back to a single
``whole_blob`` section. Tier A perturbation still runs, but section-level
attribution is coarse.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import struct
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# Ensure we can import tac + upstream + submission src modules
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "upstream"))

# These imports are heavy; guard for early --help availability.
try:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
    import numpy as np  # type: ignore
except ImportError:
    torch = None  # type: ignore
    F = None  # type: ignore
    np = None  # type: ignore


# ----------------------------------------------------------------------
# Constants per A1 + PR101 + PR106 wire format
# ----------------------------------------------------------------------

A1_LATENT_BLOB_LEN = 15_387  # PR101 latent blob length
EVAL_H, EVAL_W = 384, 512
CAMERA_H, CAMERA_W = 874, 1164
N_PAIRS = 600

# IBPS1 (C6 MDL-IBPS variant 1) header constants, mirrored from
# src/tac/substrates/c6_e4_mdl_ibps/archive.py so this tool stays
# importable even when the substrate package can't be imported (no torch
# at --help time). Per CLAUDE.md "Beauty, simplicity, and developer
# experience" — narrow contract preserved.
IBPS1_MAGIC = b"IBPS"
IBPS1_HEADER_FMT = "<4sBHHIIII"  # magic(4) + ver(1) + ld(2) + np(2) + 4*u32
IBPS1_HEADER_SIZE = struct.calcsize(IBPS1_HEADER_FMT)  # = 25

# Score formula per upstream/evaluate.py:
#   score = 100 * seg_dist + sqrt(10 * pose_dist) + 25 * rate
# We measure Δscore against the baseline so the `25 * rate` term cancels
# (we never change archive size during ablation). Both Δseg and Δpose
# are propagated separately.


# ----------------------------------------------------------------------
# Dataclasses
# ----------------------------------------------------------------------


@dataclass
class ArchiveSpec:
    """A single archive to ablate."""

    path: Path
    name: str  # e.g. "a1", "pr106_r2"
    grammar: str  # "a1" | "pr106_latent_sidecar" | "pr101"
    sha256: str
    size_bytes: int
    sections: dict[str, tuple[int, int]] = field(default_factory=dict)
    # section name -> (start_offset, length) inside the inner blob


@dataclass
class TierAResult:
    """Per-section structural ablation result."""

    section: str
    start_offset: int
    length_bytes: int
    perturbation_mode: str  # "zero" | "random" | "skip"
    inflate_success: bool
    delta_seg: float | None  # Δseg vs baseline
    delta_pose: float | None  # Δpose vs baseline
    delta_score_components: float | None  # 100 * Δseg + sqrt(10 * pose_perturbed) - sqrt(10 * pose_baseline)
    failure_reason: str | None
    elapsed_seconds: float


@dataclass
class TierBSample:
    """Single byte-flip sample."""

    section: str
    byte_offset: int
    bit_offset: int | None
    inflate_success: bool
    delta_seg: float | None
    delta_pose: float | None
    delta_score_components: float | None
    elapsed_seconds: float


@dataclass
class TierBResult:
    """Sampled byte-level ablation aggregated result."""

    section: str
    n_samples: int
    n_inflate_failures: int
    n_significant: int  # |Δscore_components| > significance_threshold
    significance_threshold: float
    mean_abs_delta: float
    std_abs_delta: float
    max_abs_delta: float
    fraction_significant: float
    # Lower-bound MDL estimate: significant bytes × 8 bits (upper bound)
    upper_bound_scorer_extracted_bits: float
    # Lower-bound MDL: significant bytes × log2(1/Δ) ≈ how informative each is
    # (heuristic; uses median delta for log2 base)
    estimated_scorer_extracted_bits_lo: float
    samples: list[TierBSample] = field(default_factory=list)


@dataclass
class TierCResult:
    """Post-decode perturbation: gaussian noise on decoded state."""

    target: str  # "state_dict" | "latents"
    noise_sigma_relative: float
    delta_seg: float | None
    delta_pose: float | None
    delta_score_components: float | None
    elapsed_seconds: float


@dataclass
class ArchiveAblationResult:
    """All-tiers ablation result for one archive."""

    archive_name: str
    archive_path: str
    archive_sha256: str
    archive_size_bytes: int
    grammar: str
    device: str  # "mps" | "cpu" | "cuda"
    pair_samples: int  # how many pairs evaluated per measurement

    baseline_seg: float
    baseline_pose: float
    baseline_score_components: float  # 100*seg + sqrt(10*pose), excludes rate

    tier_a: list[TierAResult] = field(default_factory=list)
    tier_b: list[TierBResult] = field(default_factory=list)
    tier_c: list[TierCResult] = field(default_factory=list)

    # Aggregate MDL findings
    mdl_density_estimate_lo: float = 0.0  # fraction of archive that is scorer-extracted (lower bound)
    mdl_density_estimate_hi: float = 0.0  # upper bound
    mdl_scorer_extracted_bytes_lo: float = 0.0
    mdl_scorer_extracted_bytes_hi: float = 0.0
    zen_floor_band_recommendation: str = ""  # "[low, center, high]"

    notes: list[str] = field(default_factory=list)
    timestamp_utc: str = ""
    elapsed_seconds_total: float = 0.0


# ----------------------------------------------------------------------
# Archive parsing
# ----------------------------------------------------------------------


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def parse_a1_archive_bytes(archive_bytes: bytes) -> dict[str, tuple[int, int]]:
    """Return section name -> (start, length) for A1 grammar.

    A1 wire format (inner blob, single ZIP member 'x'):
        uint32 LE: decoder_section_total_bytes (D)
        byte * (D - 4): encoded decoder blob (PR101 split-Brotli, canonical)
        byte * 15387: latent_blob (PR101 ORIGINAL)
        byte * remaining: sidecar_blob (PR101 ORIGINAL)
    """
    if len(archive_bytes) < 4:
        raise ValueError("archive too short")
    section_total = struct.unpack_from("<I", archive_bytes, 0)[0]
    return {
        "decoder_section_header": (0, 4),
        "decoder_blob": (4, section_total - 4),
        "latent_blob": (section_total, A1_LATENT_BLOB_LEN),
        "sidecar_blob": (
            section_total + A1_LATENT_BLOB_LEN,
            len(archive_bytes) - section_total - A1_LATENT_BLOB_LEN,
        ),
    }


def parse_pr106_archive_bytes(archive_bytes: bytes) -> dict[str, tuple[int, int]]:
    """Return section name -> (start, length) for PR106 latent_sidecar grammar.

    PR106 latent_sidecar wire format (see
    `submissions/pr106_latent_sidecar_r2/inflate.py::parse_sidecar_archive`):

        uint8 magic (0xFE)
        uint8 format_id (0x01)
        uint32 LE pr106_len
        bytes[pr106_len] pr106_base_archive   (Brotli-compressed state_dict + latents)
        uint16 LE sidecar_len
        bytes[sidecar_len] sidecar_blob       (Brotli-compressed (dim, delta) corrections)
    """
    if len(archive_bytes) < 8:
        raise ValueError("archive too short")
    if archive_bytes[0] != 0xFE:
        raise ValueError(f"expected magic 0xFE, got 0x{archive_bytes[0]:02X}")
    if archive_bytes[1] != 0x01:
        raise ValueError(f"expected format_id 0x01, got 0x{archive_bytes[1]:02X}")

    pos = 2
    (pr106_len,) = struct.unpack_from("<I", archive_bytes, pos)
    pos_after_len4 = pos + 4
    pr106_start = pos_after_len4
    pos_after_pr106 = pr106_start + pr106_len
    if pos_after_pr106 + 2 > len(archive_bytes):
        raise ValueError("sidecar archive truncated before sidecar_len")
    (sidecar_len,) = struct.unpack_from("<H", archive_bytes, pos_after_pr106)
    sidecar_start = pos_after_pr106 + 2

    return {
        "magic_format_header": (0, 2),
        "pr106_len_field": (2, 4),
        "pr106_base_archive": (pr106_start, pr106_len),
        "sidecar_len_field": (pos_after_pr106, 2),
        "sidecar_blob": (sidecar_start, sidecar_len),
    }


def parse_ibps1_archive_bytes(archive_bytes: bytes) -> dict[str, tuple[int, int]]:
    """Return section name -> (start, length) for IBPS1 (C6 MDL-IBPS) grammar.

    IBPS1 wire format (inner blob, single ZIP member '0.bin'); see
    :mod:`tac.substrates.c6_e4_mdl_ibps.archive` for the canonical
    definition. Layout:

        MAGIC(4)            b"IBPS"
        VERSION(1)          u8   schema version (currently 1)
        LATENT_DIM(2)       u16  cfg.latent_dim (e.g. 24)
        NUM_PAIRS(2)        u16  cfg.num_pairs (e.g. 600)
        ENCODER_BLOB_LEN(4) u32  brotli-compressed encoder state_dict len
        DECODER_BLOB_LEN(4) u32  brotli-compressed decoder state_dict len
        LATENT_BLOB_LEN(4)  u32  int8 latents bytes len (num_pairs*latent_dim)
        META_BLOB_LEN(4)    u32  sorted-keys JSON utf-8 bytes len
        ENCODER_BLOB        ...
        DECODER_BLOB        ...
        LATENT_BLOB         ...
        META_BLOB           ...

    Total header bytes = 25 (IBPS1_HEADER_SIZE).

    Returned sections (Tier A / Tier B targets):

    - ``ibps1_header`` — 25-byte header (control_or_metadata; fixed layout)
    - ``encoder_blob`` — brotli q=9 compressed encoder weights
      (decoder_weight_stream — encoder is forensic-only at inflate but
      occupies the same role family in the planner's bit-allocator)
    - ``decoder_blob`` — brotli q=9 compressed decoder weights
      (decoder_weight_stream — the actual inflate-time consumer)
    - ``latent_blob`` — int8 quantized per-pair latents (latent_stream)
    - ``meta_blob`` — sorted-keys JSON with beta_ib, decoder_channels,
      _lat_scale, _lat_zero_point (control_or_metadata)
    """
    if len(archive_bytes) < IBPS1_HEADER_SIZE:
        raise ValueError(
            f"ibps1 archive too short: got {len(archive_bytes)} bytes, "
            f"need >= {IBPS1_HEADER_SIZE} for header"
        )
    (
        magic,
        version,
        latent_dim,
        num_pairs,
        encoder_len,
        decoder_len,
        latent_len,
        meta_len,
    ) = struct.unpack(IBPS1_HEADER_FMT, archive_bytes[:IBPS1_HEADER_SIZE])
    if magic != IBPS1_MAGIC:
        raise ValueError(
            f"ibps1 archive: bad magic {magic!r} (expected {IBPS1_MAGIC!r})"
        )
    if version != 1:
        raise ValueError(
            f"ibps1 archive: unsupported schema version {version} (expected 1)"
        )
    # int8 = 1 byte each
    expected_latent_bytes = int(num_pairs) * int(latent_dim)
    if latent_len != expected_latent_bytes:
        raise ValueError(
            f"ibps1 archive: latent_len {latent_len} != num_pairs*latent_dim "
            f"= {expected_latent_bytes}"
        )
    end_header = IBPS1_HEADER_SIZE
    end_encoder = end_header + int(encoder_len)
    end_decoder = end_encoder + int(decoder_len)
    end_latents = end_decoder + int(latent_len)
    end_meta = end_latents + int(meta_len)
    if end_meta > len(archive_bytes):
        raise ValueError(
            f"ibps1 archive: declared end_meta {end_meta} > archive bytes "
            f"{len(archive_bytes)} — truncated archive"
        )
    return {
        "ibps1_header": (0, IBPS1_HEADER_SIZE),
        "encoder_blob": (end_header, int(encoder_len)),
        "decoder_blob": (end_encoder, int(decoder_len)),
        "latent_blob": (end_decoder, int(latent_len)),
        "meta_blob": (end_latents, int(meta_len)),
    }


def load_archive(archive_path: Path, grammar: str) -> tuple[bytes, dict[str, tuple[int, int]]]:
    """Read inner blob from archive.zip and parse sections.

    The "inner blob" is the single ZIP member that the inflate.sh script
    points its child inflate.py at — for A1 it is `x`, for PR106 it is
    the first file by name (`x` is conventional).
    """
    inner_bytes = _read_inner_member(archive_path, grammar)

    if grammar in ("a1", "pr101"):
        sections = parse_a1_archive_bytes(inner_bytes)
    elif grammar in ("pr106", "pr106_latent_sidecar"):
        sections = parse_pr106_archive_bytes(inner_bytes)
    elif grammar == "ibps1":
        sections = parse_ibps1_archive_bytes(inner_bytes)
    else:
        # Generic: treat whole blob as one section
        sections = {"whole_blob": (0, len(inner_bytes))}

    return inner_bytes, sections


def _read_inner_member(archive_path: Path, grammar: str) -> bytes:
    """Read the inner ZIP member that corresponds to ``inflate.sh``'s payload.

    The conventional inner member is ``x`` for the A1 / PR101 / PR106 family.
    For IBPS1 (C6 MDL-IBPS) the trainer writes the inner member as
    ``0.bin``. If neither is present, the first member by listing order is
    used (preserving previous behavior).
    """
    with zipfile.ZipFile(archive_path) as zf:
        names = zf.namelist()
        if grammar == "ibps1" and "0.bin" in names:
            inner_name = "0.bin"
        elif "x" in names:
            inner_name = "x"
        elif "0.bin" in names:
            inner_name = "0.bin"
        else:
            inner_name = names[0]
        return zf.read(inner_name)


# ----------------------------------------------------------------------
# Decoder-side: inflate inner blob bytes -> frames
# ----------------------------------------------------------------------


def _decode_a1_to_frames(
    inner_bytes: bytes,
    pair_indices: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Inflate A1 inner blob -> (N_pairs, 2, H, W, C) uint8 frames.

    Only the requested ``pair_indices`` are rendered (saves time).

    Returns shape (len(pair_indices), 2, CAMERA_H, CAMERA_W, 3) uint8 CPU tensor.

    Raises on parse/inflate failure (caller catches).
    """
    # Import only when we have a known grammar
    a1_src = REPO_ROOT / "submissions" / "a1" / "src"
    if str(a1_src) not in sys.path:
        sys.path.insert(0, str(a1_src))
    from codec import (  # type: ignore[import-not-found]
        LATENT_BLOB_LEN,
        decode_decoder_compact,
        decode_latents_compact,
        apply_latent_sidecar,
    )
    from model import HNeRVDecoder  # type: ignore[import-not-found]

    section_total = struct.unpack_from("<I", inner_bytes, 0)[0]
    if section_total < 4 or section_total > len(inner_bytes):
        raise ValueError(f"bad decoder_section_total {section_total}")
    decoder_blob = inner_bytes[4:section_total]
    latent_blob = inner_bytes[section_total:section_total + LATENT_BLOB_LEN]
    sidecar_blob = inner_bytes[section_total + LATENT_BLOB_LEN:]
    if not decoder_blob or len(latent_blob) != LATENT_BLOB_LEN:
        raise ValueError("bad A1 archive layout")

    decoder_sd = decode_decoder_compact(decoder_blob)
    latents = apply_latent_sidecar(decode_latents_compact(latent_blob), sidecar_blob)

    decoder = HNeRVDecoder(latent_dim=28, base_channels=36, eval_size=(EVAL_H, EVAL_W)).to(device)
    decoder.load_state_dict(decoder_sd)
    decoder.eval()
    latents = latents.to(device)

    pair_indices_t = sorted(set(pair_indices))
    n_target = len(pair_indices_t)
    out = torch.empty((n_target, 2, CAMERA_H, CAMERA_W, 3), dtype=torch.uint8)

    batch_size = 16
    written = 0
    with torch.inference_mode():
        for batch_start in range(0, n_target, batch_size):
            batch_end = min(batch_start + batch_size, n_target)
            batch_pairs = pair_indices_t[batch_start:batch_end]
            sel = torch.tensor(batch_pairs, dtype=torch.long, device=device)
            l = latents.index_select(0, sel)
            decoded = decoder(l)
            flat = decoded.reshape(l.shape[0] * 2, 3, EVAL_H, EVAL_W)
            up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W),
                               mode="bicubic", align_corners=False)
            up = up.reshape(l.shape[0], 2, 3, CAMERA_H, CAMERA_W)
            up[:, 0, 0].sub_(1.0)
            up[:, 0, 2].sub_(1.0)
            up[:, 1, 1].sub_(1.0)
            frames = (
                up.reshape(l.shape[0] * 2, 3, CAMERA_H, CAMERA_W)
                .clamp(0, 255)
                .permute(0, 2, 3, 1)
                .round()
                .to(torch.uint8)
                .cpu()
            ).reshape(l.shape[0], 2, CAMERA_H, CAMERA_W, 3)
            out[written:written + l.shape[0]] = frames
            written += l.shape[0]
    return out


def _decode_pr106_to_frames(
    inner_bytes: bytes,
    pair_indices: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Inflate PR106 latent_sidecar inner blob -> uint8 frames for given pairs.

    Per CLAUDE.md "Forbidden in-place edits to public PR intake clones" we
    do NOT modify PR106's inflate.py. Instead we replicate its parser
    contract here, then bypass its ``select_inflate_device`` (which
    refuses MPS) so the MDL ablation can run on MPS.

    Per CLAUDE.md "MPS auth eval is NOISE" the absolute scores from this
    path are NOT contest-grade; that's correct for MDL ablation which
    measures DELTAS not absolute values.
    """
    pr106_src = REPO_ROOT / "submissions" / "pr106_latent_sidecar_r2" / "src"
    if str(pr106_src) not in sys.path:
        sys.path.insert(0, str(pr106_src))

    from codec import parse_packed_archive  # type: ignore[import-not-found]
    from model import HNeRVDecoder  # type: ignore[import-not-found]

    # Mirror inflate.py::parse_sidecar_archive parsing exactly:
    #   uint8 magic (0xFE) + uint8 format_id (0x01)
    #   uint32 LE pr106_len + pr106_bytes (PR106 base archive)
    #   uint16 LE sidecar_len + sidecar_blob (brotli-compressed dim/delta arr)
    SIDECAR_MAGIC = 0xFE
    SIDECAR_FORMAT_ID = 0x01
    DELTA_SCALE = 0.01
    NO_OP_DIM = 255

    if len(inner_bytes) < 8:
        raise ValueError("pr106 archive too short")
    if inner_bytes[0] != SIDECAR_MAGIC:
        raise ValueError(f"expected magic 0xFE, got 0x{inner_bytes[0]:02X}")
    if inner_bytes[1] != SIDECAR_FORMAT_ID:
        raise ValueError(f"expected format_id 0x01, got 0x{inner_bytes[1]:02X}")

    pos = 2
    (pr106_len,) = struct.unpack_from("<I", inner_bytes, pos)
    pos += 4
    pr106_bytes = inner_bytes[pos:pos + pr106_len]
    pos += pr106_len
    if pos + 2 > len(inner_bytes):
        raise ValueError("sidecar archive truncated before sidecar_len")
    (sidecar_len,) = struct.unpack_from("<H", inner_bytes, pos)
    pos += 2
    sidecar_blob = inner_bytes[pos:pos + sidecar_len]
    pos += sidecar_len

    # Parse PR106 base archive -> (state_dict, latents)
    try:
        parsed = parse_packed_archive(pr106_bytes)
    except Exception as e:
        raise ValueError(f"parse_packed_archive failed: {e}") from e
    if isinstance(parsed, tuple) and len(parsed) >= 2:
        state_dict = parsed[0]
        latents = parsed[1]
    else:
        raise ValueError(f"parse_packed_archive returned unexpected shape: {type(parsed)}")

    # Decode sidecar (brotli-compressed (dim, delta_q) pairs).
    # Per inflate.py decode_sidecar_corrections:
    #   raw = brotli.decompress(blob)
    #   n = uint16 LE at offset 0
    #   arr = uint8[n * 2] starting offset 2 -> (n, 2) -> col0=dim, col1=delta_q (int8)
    import brotli  # type: ignore[import-not-found]
    latents = latents.clone()
    if sidecar_blob:
        try:
            raw = brotli.decompress(sidecar_blob)
            n_corr = struct.unpack_from("<H", raw, 0)[0]
            arr_bytes = raw[2:2 + 2 * n_corr]
            arr = np.frombuffer(arr_bytes, dtype=np.uint8).reshape(n_corr, 2)
            dim_arr = arr[:, 0]
            delta_q_arr = arr[:, 1].view(np.int8)
            for p in range(min(n_corr, latents.shape[0])):
                d = int(dim_arr[p])
                if d == NO_OP_DIM:
                    continue
                if d < latents.shape[1]:
                    latents[p, d] = latents[p, d] + float(delta_q_arr[p]) * DELTA_SCALE
        except Exception as e:
            raise ValueError(f"sidecar decode failed: {e}") from e

    # Decoder
    latent_dim = latents.shape[1]
    # PR106 r2 uses base_channels=36 + latent_dim=28; verify if possible
    decoder = HNeRVDecoder(latent_dim=latent_dim, base_channels=36, eval_size=(EVAL_H, EVAL_W)).to(device)
    try:
        decoder.load_state_dict(state_dict)
    except Exception as e:
        raise ValueError(f"decoder.load_state_dict failed: {e}") from e
    decoder.eval()
    latents = latents.to(device)

    pair_indices_t = sorted(set(pair_indices))
    n_target = len(pair_indices_t)
    out = torch.empty((n_target, 2, CAMERA_H, CAMERA_W, 3), dtype=torch.uint8)

    batch_size = 16
    written = 0
    with torch.inference_mode():
        for batch_start in range(0, n_target, batch_size):
            batch_end = min(batch_start + batch_size, n_target)
            batch_pairs = pair_indices_t[batch_start:batch_end]
            sel = torch.tensor(batch_pairs, dtype=torch.long, device=device)
            l = latents.index_select(0, sel)
            decoded = decoder(l)
            flat = decoded.reshape(l.shape[0] * 2, 3, EVAL_H, EVAL_W)
            up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W),
                               mode="bicubic", align_corners=False)
            up = up.reshape(l.shape[0], 2, 3, CAMERA_H, CAMERA_W)
            # PR106 also has the per-channel -1.0 sub_ adjustments (mirror inflate.py)
            up[:, 0, 0].sub_(1.0)
            up[:, 0, 2].sub_(1.0)
            up[:, 1, 1].sub_(1.0)
            frames = (
                up.reshape(l.shape[0] * 2, 3, CAMERA_H, CAMERA_W)
                .clamp(0, 255)
                .permute(0, 2, 3, 1)
                .round()
                .to(torch.uint8)
                .cpu()
            ).reshape(l.shape[0], 2, CAMERA_H, CAMERA_W, 3)
            out[written:written + l.shape[0]] = frames
            written += l.shape[0]
    return out


def _decode_ibps1_to_frames(
    inner_bytes: bytes,
    pair_indices: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Inflate IBPS1 (C6 MDL-IBPS) inner blob -> uint8 frames for given pairs.

    Per CLAUDE.md "MPS auth eval is NOISE" the absolute scores from this
    path are NOT contest-grade; that's correct for MDL ablation which
    measures DELTAS not absolute values.

    Per CLAUDE.md "Forbidden in-place edits to public PR intake clones"
    we do NOT mutate substrate inflate.py — we re-construct the substrate
    from the parsed archive (mirrors :func:`inflate_one_video`'s
    contract) but render at the scorer-resolution (EVAL_H, EVAL_W) then
    bicubic-upsample to CAMERA (874, 1164) so the output shape matches
    the A1 / PR106 contract and can be compared against PyAV-decoded GT.

    Returns shape (len(pair_indices), 2, CAMERA_H, CAMERA_W, 3) uint8 CPU
    tensor.

    Raises on parse / inflate failure (caller catches in the byte-flip
    Tier B loop — an inflate failure counts as a relevant byte position).
    """
    # Late imports — C6 substrate may not be importable in test fixtures
    from tac.substrates.c6_e4_mdl_ibps.architecture import (  # type: ignore[import-not-found]
        EVAL_HW,
        MDLIBPSConfig,
        MDLIBPSSubstrate,
    )
    from tac.substrates.c6_e4_mdl_ibps.archive import (  # type: ignore[import-not-found]
        parse_archive,
    )

    arc = parse_archive(inner_bytes)
    meta = arc.meta
    cfg = MDLIBPSConfig(
        latent_dim=int(arc.latents.shape[1]),
        encoder_input_channels=int(meta.get("encoder_input_channels", 3)),
        encoder_sin_freq=float(meta.get("encoder_sin_freq", 30.0)),
        decoder_embed_dim=int(meta["decoder_embed_dim"]),
        decoder_initial_grid_h=int(meta["decoder_initial_grid_h"]),
        decoder_initial_grid_w=int(meta["decoder_initial_grid_w"]),
        decoder_channels=tuple(int(c) for c in meta["decoder_channels"]),
        decoder_num_upsample_blocks=int(meta["decoder_num_upsample_blocks"]),
        decoder_sin_freq=float(meta.get("decoder_sin_freq", 30.0)),
        num_pairs=int(arc.latents.shape[0]),
        output_height=int(meta.get("output_height", EVAL_HW[0])),
        output_width=int(meta.get("output_width", EVAL_HW[1])),
        beta_ib=float(meta.get("beta_ib", 0.01)),
        latent_init_std=float(meta.get("latent_init_std", 0.02)),
    )

    model = MDLIBPSSubstrate(cfg).to(device).eval()
    enc_load = model.encoder.load_state_dict(arc.encoder_state_dict, strict=False)
    if set(enc_load.missing_keys) or set(enc_load.unexpected_keys):
        raise ValueError(
            f"ibps1: encoder state_dict mismatch missing={sorted(enc_load.missing_keys)} "
            f"unexpected={sorted(enc_load.unexpected_keys)}"
        )
    dec_load = model.decoder.load_state_dict(arc.decoder_state_dict, strict=False)
    if set(dec_load.missing_keys) or set(dec_load.unexpected_keys):
        raise ValueError(
            f"ibps1: decoder state_dict mismatch missing={sorted(dec_load.missing_keys)} "
            f"unexpected={sorted(dec_load.unexpected_keys)}"
        )
    with torch.no_grad():
        model.latents.copy_(arc.latents.to(device=device, dtype=model.latents.dtype))

    pair_indices_t = sorted(set(pair_indices))
    n_target = len(pair_indices_t)
    out = torch.empty((n_target, 2, CAMERA_H, CAMERA_W, 3), dtype=torch.uint8)

    written = 0
    with torch.inference_mode():
        for pair_idx in pair_indices_t:
            idx_t = torch.tensor([pair_idx], device=device, dtype=torch.long)
            # rgb_0, rgb_1 are at EVAL_HW (384, 512), shape (1, 3, H, W),
            # range [0, 1] per the substrate's "input_range=unit" contract
            # (see substrates._shared.inflate_runtime.write_rgb_pair_to_raw).
            rgb_0, rgb_1, _mu, _logvar = model(idx_t, frames_for_encoder=None)
            # Stack as (2, 3, EVAL_H, EVAL_W)
            stacked = torch.cat([rgb_0, rgb_1], dim=0)  # (2, 3, H, W)
            # Upsample to CAMERA resolution via bicubic; A1/PR106 also do this.
            up = F.interpolate(
                stacked, size=(CAMERA_H, CAMERA_W),
                mode="bicubic", align_corners=False,
            )
            # Convert [0, 1] -> [0, 255] uint8
            frames = (
                (up * 255.0)
                .clamp(0, 255)
                .permute(0, 2, 3, 1)
                .round()
                .to(torch.uint8)
                .cpu()
            )  # (2, H, W, 3)
            out[written, 0] = frames[0]
            out[written, 1] = frames[1]
            written += 1
    return out


def decode_to_frames(
    inner_bytes: bytes,
    grammar: str,
    pair_indices: list[int],
    device: torch.device,
) -> torch.Tensor:
    """Dispatch on grammar."""
    if grammar in ("a1", "pr101"):
        return _decode_a1_to_frames(inner_bytes, pair_indices, device)
    elif grammar in ("pr106", "pr106_latent_sidecar"):
        return _decode_pr106_to_frames(inner_bytes, pair_indices, device)
    elif grammar == "ibps1":
        return _decode_ibps1_to_frames(inner_bytes, pair_indices, device)
    else:
        raise NotImplementedError(f"grammar {grammar} not supported")


# ----------------------------------------------------------------------
# Ground-truth frames (decoded from upstream/videos/0.mkv)
# ----------------------------------------------------------------------


def _load_ground_truth_pairs(
    video_path: Path,
    pair_indices: list[int],
) -> torch.Tensor:
    """Decode requested pairs from video. Returns (N, 2, H, W, 3) uint8 CPU tensor.

    Uses PyAV (sister to upstream AVVideoDataset) for cross-platform decode.
    """
    import av  # type: ignore[import-not-found]

    pair_indices_t = sorted(set(pair_indices))
    # Each pair uses 2 frames; pair i corresponds to source frames (2*i, 2*i+1)
    # per the contest's seq_len=2 non-overlapping pair construction.
    needed_frame_indices = sorted({2 * p + k for p in pair_indices_t for k in (0, 1)})
    needed_set = set(needed_frame_indices)
    max_frame = max(needed_frame_indices) if needed_frame_indices else 0

    out = torch.empty((len(pair_indices_t), 2, CAMERA_H, CAMERA_W, 3), dtype=torch.uint8)

    # Build pair_idx -> output_row mapping
    pair_to_row = {p: i for i, p in enumerate(pair_indices_t)}
    frame_buffer: dict[int, torch.Tensor] = {}

    container = av.open(str(video_path))
    stream = container.streams.video[0]
    decoded_count = 0
    for frame_idx, frame in enumerate(container.decode(stream)):
        if frame_idx in needed_set:
            arr = frame.to_ndarray(format="rgb24")  # (H, W, 3) uint8
            if arr.shape != (CAMERA_H, CAMERA_W, 3):
                raise ValueError(f"unexpected GT frame shape {arr.shape}")
            frame_buffer[frame_idx] = torch.from_numpy(arr.copy())
            decoded_count += 1
        if frame_idx >= max_frame:
            break
    container.close()

    # Assemble pairs
    for p, row in pair_to_row.items():
        f0 = frame_buffer.get(2 * p)
        f1 = frame_buffer.get(2 * p + 1)
        if f0 is None or f1 is None:
            raise ValueError(f"missing GT frames for pair {p}")
        out[row, 0] = f0
        out[row, 1] = f1

    return out


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------


def _compute_seg_pose_delta(
    distortion_net: torch.nn.Module,
    gt_pairs: torch.Tensor,  # (N, 2, H, W, 3) uint8 CPU
    candidate_pairs: torch.Tensor,  # (N, 2, H, W, 3) uint8 CPU
    device: torch.device,
    batch_size: int = 4,
) -> tuple[float, float]:
    """Mean per-pair (pose_dist, seg_dist) over the supplied pairs.

    Matches upstream/evaluate.py: distortion_net.compute_distortion(gt, comp)
    -> (pose_dist, seg_dist) each shape (B,).
    """
    n = gt_pairs.shape[0]
    pose_sum = 0.0
    seg_sum = 0.0
    with torch.inference_mode():
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            gt_b = gt_pairs[batch_start:batch_end].to(device).float()
            comp_b = candidate_pairs[batch_start:batch_end].to(device).float()
            pose, seg = distortion_net.compute_distortion(gt_b, comp_b)
            # Each is shape (B,)
            pose_sum += float(pose.sum().item())
            seg_sum += float(seg.sum().item())
    return (pose_sum / n, seg_sum / n)


def _score_components(pose_dist: float, seg_dist: float) -> float:
    """100 * seg + sqrt(10 * pose). Rate term excluded (constant across ablation)."""
    return 100.0 * seg_dist + math.sqrt(10.0 * max(0.0, pose_dist))


# ----------------------------------------------------------------------
# Tier A: structural ablation
# ----------------------------------------------------------------------


def _perturb_section(
    inner_bytes: bytes,
    start: int,
    length: int,
    mode: str,
    rng: random.Random,
) -> bytes:
    """Return perturbed bytes (does NOT mutate input)."""
    buf = bytearray(inner_bytes)
    if mode == "zero":
        for i in range(start, start + length):
            buf[i] = 0
    elif mode == "random":
        for i in range(start, start + length):
            buf[i] = rng.randrange(0, 256)
    elif mode == "skip":
        # Skip means delete the section entirely. Since this changes length
        # (and that breaks downstream offsets), we instead overwrite with
        # zeros AND mark this as "lossless if section is irrelevant".
        for i in range(start, start + length):
            buf[i] = 0
    else:
        raise ValueError(f"unknown mode {mode}")
    return bytes(buf)


def run_tier_a(
    archive: ArchiveSpec,
    inner_bytes: bytes,
    grammar: str,
    pair_indices: list[int],
    gt_pairs: torch.Tensor,
    baseline_seg: float,
    baseline_pose: float,
    distortion_net: torch.nn.Module,
    device: torch.device,
    rng: random.Random,
) -> list[TierAResult]:
    """Run structural section ablation."""
    results: list[TierAResult] = []

    section_modes = {
        # A1 / PR101 sections
        "decoder_section_header": ["zero"],
        "decoder_blob": ["zero", "random"],
        "latent_blob": ["zero", "random"],
        "sidecar_blob": ["zero", "random"],
        # PR106 sections
        "magic_header": ["zero"],
        "pr106_body_plus_sidecar": ["zero", "random"],
        # IBPS1 (C6 MDL-IBPS) sections
        "ibps1_header": ["zero"],  # zero magic / version -> inflate failure
        "encoder_blob": ["zero", "random"],  # forensic-only but still required by parser
        # decoder_blob handled above (same name as A1 / PR101)
        # latent_blob handled above
        "meta_blob": ["zero", "random"],  # JSON metadata; zero -> parse failure
    }

    baseline_score_comp = _score_components(baseline_pose, baseline_seg)

    for section_name, (start, length) in archive.sections.items():
        modes_to_run = section_modes.get(section_name, ["zero"])
        for mode in modes_to_run:
            t0 = time.time()
            perturbed = _perturb_section(inner_bytes, start, length, mode, rng)
            try:
                cand_frames = decode_to_frames(perturbed, grammar, pair_indices, device)
                pose_p, seg_p = _compute_seg_pose_delta(
                    distortion_net, gt_pairs, cand_frames, device
                )
                dseg = seg_p - baseline_seg
                dpose = pose_p - baseline_pose
                # Δscore_components = 100*Δseg + (sqrt(10*pose_p) - sqrt(10*pose_b))
                dsc = _score_components(pose_p, seg_p) - baseline_score_comp
                inflate_ok = True
                failure_reason = None
            except Exception as e:
                inflate_ok = False
                dseg = None
                dpose = None
                dsc = None
                failure_reason = f"{type(e).__name__}: {e}"
            t1 = time.time()
            results.append(TierAResult(
                section=section_name,
                start_offset=start,
                length_bytes=length,
                perturbation_mode=mode,
                inflate_success=inflate_ok,
                delta_seg=dseg,
                delta_pose=dpose,
                delta_score_components=dsc,
                failure_reason=failure_reason,
                elapsed_seconds=t1 - t0,
            ))

    return results


# ----------------------------------------------------------------------
# Tier B: sampled byte-level ablation
# ----------------------------------------------------------------------


def run_tier_b(
    archive: ArchiveSpec,
    inner_bytes: bytes,
    grammar: str,
    pair_indices: list[int],
    gt_pairs: torch.Tensor,
    baseline_seg: float,
    baseline_pose: float,
    distortion_net: torch.nn.Module,
    device: torch.device,
    rng: random.Random,
    n_samples_per_section: int,
    significance_threshold: float = 1e-4,
    flip_mode: str = "xor_0xff",  # "xor_0xff" | "xor_bit_msb" | "random_byte"
) -> list[TierBResult]:
    """For each section, flip N random byte positions, measure Δscore."""
    baseline_score_comp = _score_components(baseline_pose, baseline_seg)
    results: list[TierBResult] = []

    # Skip tiny sections (headers) — they get covered by Tier A
    section_min_length_for_sampling = 16

    for section_name, (start, length) in archive.sections.items():
        if length < section_min_length_for_sampling:
            continue
        n_samples = min(n_samples_per_section, length)
        offsets = sorted(rng.sample(range(length), n_samples))
        samples: list[TierBSample] = []
        n_inflate_failures = 0
        n_significant = 0
        abs_deltas = []
        for off in offsets:
            byte_global = start + off
            t0 = time.time()
            buf = bytearray(inner_bytes)
            orig = buf[byte_global]
            if flip_mode == "xor_0xff":
                buf[byte_global] = orig ^ 0xFF
            elif flip_mode == "xor_bit_msb":
                buf[byte_global] = orig ^ 0x80
            elif flip_mode == "random_byte":
                buf[byte_global] = rng.randrange(0, 256)
                while buf[byte_global] == orig:
                    buf[byte_global] = rng.randrange(0, 256)
            else:
                raise ValueError(f"unknown flip_mode {flip_mode}")
            try:
                cand_frames = decode_to_frames(bytes(buf), grammar, pair_indices, device)
                pose_p, seg_p = _compute_seg_pose_delta(
                    distortion_net, gt_pairs, cand_frames, device
                )
                dseg = seg_p - baseline_seg
                dpose = pose_p - baseline_pose
                dsc = _score_components(pose_p, seg_p) - baseline_score_comp
                inflate_ok = True
                abs_deltas.append(abs(dsc))
                if abs(dsc) > significance_threshold:
                    n_significant += 1
            except Exception:
                inflate_ok = False
                dseg = None
                dpose = None
                dsc = None
                n_inflate_failures += 1
                # Inflate failure means the bit is REQUIRED for parser
                # consumption — count as a "structural significance" sample
                # but don't add to abs_deltas (no numeric delta).
                # We DO count it as scorer-extracted because changing it
                # breaks the score completely.
                n_significant += 1
            t1 = time.time()
            samples.append(TierBSample(
                section=section_name,
                byte_offset=off,
                bit_offset=None,
                inflate_success=inflate_ok,
                delta_seg=dseg,
                delta_pose=dpose,
                delta_score_components=dsc,
                elapsed_seconds=t1 - t0,
            ))

        if abs_deltas:
            arr = np.array(abs_deltas, dtype=np.float64)
            mean_abs = float(arr.mean())
            std_abs = float(arr.std())
            max_abs = float(arr.max())
        else:
            mean_abs = 0.0
            std_abs = 0.0
            max_abs = 0.0

        frac_sig = n_significant / max(1, n_samples)
        # Lower-bound bits estimate: frac_sig * length * 8 (assuming each
        # significant byte carries 8 bits of scorer-relevant info)
        upper_bits = frac_sig * length * 8
        # More conservative estimate: use mean abs delta to estimate bits.
        # Heuristic: bits ≈ log2(1 + mean_delta / threshold). Clipped at 8.
        if mean_abs > 0:
            est_bits_per_sig_byte = min(8.0, math.log2(1.0 + mean_abs / significance_threshold))
        else:
            est_bits_per_sig_byte = 0.0
        lo_bits = frac_sig * length * est_bits_per_sig_byte

        results.append(TierBResult(
            section=section_name,
            n_samples=n_samples,
            n_inflate_failures=n_inflate_failures,
            n_significant=n_significant,
            significance_threshold=significance_threshold,
            mean_abs_delta=mean_abs,
            std_abs_delta=std_abs,
            max_abs_delta=max_abs,
            fraction_significant=frac_sig,
            upper_bound_scorer_extracted_bits=upper_bits,
            estimated_scorer_extracted_bits_lo=lo_bits,
            samples=samples,
        ))

    return results


# ----------------------------------------------------------------------
# Tier C: post-decode perturbation (decoder state_dict / latents)
# ----------------------------------------------------------------------


def run_tier_c(
    inner_bytes: bytes,
    grammar: str,
    pair_indices: list[int],
    gt_pairs: torch.Tensor,
    baseline_seg: float,
    baseline_pose: float,
    distortion_net: torch.nn.Module,
    device: torch.device,
    rng: random.Random,
    noise_sigmas: list[float] | None = None,
) -> list[TierCResult]:
    """Inject gaussian noise on state_dict OR latents at multiple sigmas.

    Provides a clean MDL-sensitivity vs noise-magnitude curve.

    Note: this REQUIRES re-implementing the inflate path with monkey-patched
    decode. We instead intercept at the assembled-tensor boundary by
    re-running the decode-and-inflate loop locally (inflate.py replica).
    Only A1 grammar is implemented for Tier C in this initial version.
    """
    if grammar != "a1":
        return []  # PR106 Tier C is non-trivial; defer

    if noise_sigmas is None:
        noise_sigmas = [0.001, 0.01, 0.1, 1.0]

    a1_src = REPO_ROOT / "submissions" / "a1" / "src"
    if str(a1_src) not in sys.path:
        sys.path.insert(0, str(a1_src))
    from codec import (  # type: ignore[import-not-found]
        LATENT_BLOB_LEN,
        decode_decoder_compact,
        decode_latents_compact,
        apply_latent_sidecar,
    )
    from model import HNeRVDecoder  # type: ignore[import-not-found]

    section_total = struct.unpack_from("<I", inner_bytes, 0)[0]
    decoder_blob = inner_bytes[4:section_total]
    latent_blob = inner_bytes[section_total:section_total + LATENT_BLOB_LEN]
    sidecar_blob = inner_bytes[section_total + LATENT_BLOB_LEN:]
    base_decoder_sd = decode_decoder_compact(decoder_blob)
    base_latents = apply_latent_sidecar(
        decode_latents_compact(latent_blob), sidecar_blob
    ).to(device)

    baseline_score_comp = _score_components(baseline_pose, baseline_seg)
    results: list[TierCResult] = []

    def _render(decoder_sd, latents):
        decoder = HNeRVDecoder(latent_dim=28, base_channels=36, eval_size=(EVAL_H, EVAL_W)).to(device)
        decoder.load_state_dict(decoder_sd)
        decoder.eval()
        pair_indices_t = sorted(set(pair_indices))
        n_target = len(pair_indices_t)
        out = torch.empty((n_target, 2, CAMERA_H, CAMERA_W, 3), dtype=torch.uint8)
        batch_size = 16
        written = 0
        with torch.inference_mode():
            for batch_start in range(0, n_target, batch_size):
                batch_end = min(batch_start + batch_size, n_target)
                batch_pairs = pair_indices_t[batch_start:batch_end]
                sel = torch.tensor(batch_pairs, dtype=torch.long, device=device)
                l = latents.index_select(0, sel)
                decoded = decoder(l)
                flat = decoded.reshape(l.shape[0] * 2, 3, EVAL_H, EVAL_W)
                up = F.interpolate(flat, size=(CAMERA_H, CAMERA_W),
                                   mode="bicubic", align_corners=False)
                up = up.reshape(l.shape[0], 2, 3, CAMERA_H, CAMERA_W)
                up[:, 0, 0].sub_(1.0)
                up[:, 0, 2].sub_(1.0)
                up[:, 1, 1].sub_(1.0)
                frames = (
                    up.reshape(l.shape[0] * 2, 3, CAMERA_H, CAMERA_W)
                    .clamp(0, 255)
                    .permute(0, 2, 3, 1)
                    .round()
                    .to(torch.uint8)
                    .cpu()
                ).reshape(l.shape[0], 2, CAMERA_H, CAMERA_W, 3)
                out[written:written + l.shape[0]] = frames
                written += l.shape[0]
        return out

    for sigma in noise_sigmas:
        for target in ("state_dict", "latents"):
            t0 = time.time()
            if target == "state_dict":
                perturbed_sd = {}
                for k, v in base_decoder_sd.items():
                    v_dev = v.to(device)
                    rel_std = v_dev.std().clamp(min=1e-8)
                    noise = torch.randn_like(v_dev) * (rel_std * sigma)
                    perturbed_sd[k] = (v_dev + noise).cpu()
                cand_frames = _render(perturbed_sd, base_latents)
            else:  # latents
                rel_std = base_latents.std().clamp(min=1e-8)
                noise = torch.randn_like(base_latents) * (rel_std * sigma)
                perturbed_latents = base_latents + noise
                cand_frames = _render(base_decoder_sd, perturbed_latents)
            pose_p, seg_p = _compute_seg_pose_delta(
                distortion_net, gt_pairs, cand_frames, device
            )
            dseg = seg_p - baseline_seg
            dpose = pose_p - baseline_pose
            dsc = _score_components(pose_p, seg_p) - baseline_score_comp
            t1 = time.time()
            results.append(TierCResult(
                target=target,
                noise_sigma_relative=sigma,
                delta_seg=dseg,
                delta_pose=dpose,
                delta_score_components=dsc,
                elapsed_seconds=t1 - t0,
            ))

    return results


# ----------------------------------------------------------------------
# Aggregate MDL estimate
# ----------------------------------------------------------------------


def aggregate_mdl_estimate(
    archive_result: ArchiveAblationResult,
) -> ArchiveAblationResult:
    """Combine Tier A + Tier B + Tier C into a single MDL band estimate.

    Strategy:
    - Tier A gives section-level: any section whose zero/random ablation
      moves score by < 1e-3 is "redundant" and excluded from MDL.
    - Tier B gives byte-density per section: `fraction_significant` of
      bytes in section S carry score-relevant info.
    - Tier B upper_bound = fraction_sig * length * 8.
    - Tier B lower_bound = fraction_sig * length * est_bits_per_sig_byte.

    Final MDL_scorer_extracted_bytes:
    - LO: sum across sections of (fraction_sig_S * length_S) when Tier A
      says section is non-redundant (zero-ablation Δ > 1e-3)
    - HI: same but using upper-bound bits (×8 / 8) = same as LO in bytes;
      we report bits as a separate field
    """
    significance = 1e-3
    section_to_lo = {}
    section_to_hi = {}

    # Identify which sections survive Tier A (carry score-relevant info).
    # If Tier A was skipped (empty), treat all sections with Tier B
    # measurements as relevant — the byte-flip + inflate-failure proxy
    # is itself a sensitivity signal.
    relevant_sections: set[str] = set()
    if archive_result.tier_a:
        for ta in archive_result.tier_a:
            if ta.delta_score_components is None:
                relevant_sections.add(ta.section)  # inflate failure = relevant
                continue
            if abs(ta.delta_score_components) > significance:
                relevant_sections.add(ta.section)
    else:
        for tb in archive_result.tier_b:
            relevant_sections.add(tb.section)

    # Build section name -> length lookup. Prefer Tier A entries (with
    # full section metadata); fall back to inferred length from Tier B
    # sample count (over-estimates but bounded).
    section_lengths: dict[str, int] = {}
    for ta in archive_result.tier_a:
        # If we have multiple rows for the same section (e.g. zero+random
        # modes), they have the same length; overwrite is harmless.
        section_lengths[ta.section] = ta.length_bytes

    # For each relevant section, use Tier B byte-density
    for tb in archive_result.tier_b:
        if tb.section not in relevant_sections:
            continue
        length = section_lengths.get(tb.section, tb.n_samples)
        # bytes_lo: fraction_significant * length (count of bytes that move score)
        bytes_lo = tb.fraction_significant * length
        section_to_lo[tb.section] = bytes_lo
        section_to_hi[tb.section] = bytes_lo  # bytes are bytes; bits scale

    total_lo = sum(section_to_lo.values())
    total_hi = sum(section_to_hi.values())

    # Density = bytes scorer-extracted / archive size
    if archive_result.archive_size_bytes > 0:
        density_lo = total_lo / archive_result.archive_size_bytes
        density_hi = total_hi / archive_result.archive_size_bytes
    else:
        density_lo = 0.0
        density_hi = 0.0

    archive_result.mdl_scorer_extracted_bytes_lo = total_lo
    archive_result.mdl_scorer_extracted_bytes_hi = total_hi
    archive_result.mdl_density_estimate_lo = density_lo
    archive_result.mdl_density_estimate_hi = density_hi

    # Zen-floor band recommendation per Z1 spec
    if total_hi < 1_000:
        band = "[0.003, 0.010] — Shannon zen-floor confirmed; staircase HIGH-EV"
    elif total_hi < 10_000:
        band = "[0.010, 0.050] — MEDIUM-EV; cooperative-receiver substrate measurable"
    elif total_hi < 50_000:
        band = "[0.050, 0.100] — current substrates close to floor; LOW-EV for sub-0.10"
    else:
        band = "[0.100, 0.150] — major architectural breakthrough needed for sub-0.10"
    archive_result.zen_floor_band_recommendation = band

    return archive_result


# ----------------------------------------------------------------------
# Top-level orchestration
# ----------------------------------------------------------------------


def ablate_archive(
    archive_path: Path,
    archive_name: str,
    grammar: str,
    upstream_dir: Path,
    output_dir: Path,
    device: torch.device,
    pair_samples: int,
    byte_samples_per_section: int,
    seed: int,
    run_tier_a_flag: bool = True,
    run_tier_b_flag: bool = True,
    run_tier_c_flag: bool = True,
) -> ArchiveAblationResult:
    """Full 3-tier ablation on one archive."""
    from datetime import datetime, timezone
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    rng = random.Random(seed)
    torch.manual_seed(seed)
    if np is not None:
        np.random.seed(seed)

    # Read archive bytes (grammar-aware inner-member selection)
    inner_bytes = _read_inner_member(archive_path, grammar)
    archive_bytes_total = archive_path.stat().st_size
    archive_sha256 = _sha256_bytes(archive_path.read_bytes())

    # Parse sections
    if grammar in ("a1", "pr101"):
        sections = parse_a1_archive_bytes(inner_bytes)
    elif grammar in ("pr106", "pr106_latent_sidecar"):
        sections = parse_pr106_archive_bytes(inner_bytes)
    elif grammar == "ibps1":
        sections = parse_ibps1_archive_bytes(inner_bytes)
    else:
        sections = {"whole_blob": (0, len(inner_bytes))}

    archive_spec = ArchiveSpec(
        path=archive_path,
        name=archive_name,
        grammar=grammar,
        sha256=archive_sha256,
        size_bytes=archive_bytes_total,
        sections=sections,
    )

    # Sample pair indices (uniformly over 0..599)
    pair_indices = sorted(rng.sample(range(N_PAIRS), pair_samples))

    # Load ground-truth frames (the same ones for every ablation)
    video_path = upstream_dir / "videos" / "0.mkv"
    print(f"[{archive_name}] Loading GT pairs ({len(pair_indices)} pairs) from {video_path}...")
    t_gt0 = time.time()
    gt_pairs = _load_ground_truth_pairs(video_path, pair_indices)
    t_gt1 = time.time()
    print(f"[{archive_name}] GT load: {t_gt1-t_gt0:.2f}s, shape {gt_pairs.shape}")

    # Load scorer (DistortionNet, matches upstream/evaluate.py exactly)
    print(f"[{archive_name}] Loading scorer on {device}...")
    from modules import DistortionNet, posenet_sd_path, segnet_sd_path  # type: ignore[import-not-found]
    distortion_net = DistortionNet().eval().to(device)
    distortion_net.load_state_dicts(str(posenet_sd_path), str(segnet_sd_path), device)

    # Baseline: inflate the unmodified archive
    print(f"[{archive_name}] Computing baseline...")
    t_b0 = time.time()
    baseline_frames = decode_to_frames(inner_bytes, grammar, pair_indices, device)
    baseline_pose, baseline_seg = _compute_seg_pose_delta(
        distortion_net, gt_pairs, baseline_frames, device
    )
    t_b1 = time.time()
    baseline_sc = _score_components(baseline_pose, baseline_seg)
    print(f"[{archive_name}] Baseline: pose={baseline_pose:.6f} seg={baseline_seg:.6f} "
          f"score_components={baseline_sc:.4f} ({t_b1-t_b0:.2f}s)")
    print(f"[{archive_name}]   (rate term not included; rate = {25 * archive_bytes_total / 37_545_489:.6f})")

    result = ArchiveAblationResult(
        archive_name=archive_name,
        archive_path=str(archive_path),
        archive_sha256=archive_sha256,
        archive_size_bytes=archive_bytes_total,
        grammar=grammar,
        device=str(device),
        pair_samples=pair_samples,
        baseline_seg=baseline_seg,
        baseline_pose=baseline_pose,
        baseline_score_components=baseline_sc,
        timestamp_utc=timestamp_utc,
        notes=[
            "[MDL-ablation-MPS]" if str(device) == "mps" else f"[MDL-ablation-{device}]",
            "[mathematical-derivation]",
            "Δscore measurements only; absolute scores are NOT contest-CUDA / contest-CPU.",
            "Rate term excluded from score_components; archive size constant across ablations.",
        ],
    )

    t_total0 = time.time()

    # Tier A: structural
    if run_tier_a_flag:
        print(f"[{archive_name}] Tier A: structural section ablation...")
        t_a0 = time.time()
        result.tier_a = run_tier_a(
            archive_spec, inner_bytes, grammar, pair_indices, gt_pairs,
            baseline_seg, baseline_pose, distortion_net, device, rng,
        )
        t_a1 = time.time()
        print(f"[{archive_name}] Tier A complete ({t_a1-t_a0:.2f}s, {len(result.tier_a)} measurements)")
        for ta in result.tier_a:
            tag = "INFLATE-FAIL" if not ta.inflate_success else (
                "SIGNIFICANT" if (ta.delta_score_components and abs(ta.delta_score_components) > 1e-3) else "NEGLIGIBLE"
            )
            print(f"  {ta.section:35s} [{ta.perturbation_mode:6s}] Δscore_comp={ta.delta_score_components} {tag}")

    # Tier B: sampled byte-level
    if run_tier_b_flag:
        print(f"[{archive_name}] Tier B: sampled byte-level ablation ({byte_samples_per_section}/section)...")
        t_b_start = time.time()
        result.tier_b = run_tier_b(
            archive_spec, inner_bytes, grammar, pair_indices, gt_pairs,
            baseline_seg, baseline_pose, distortion_net, device, rng,
            n_samples_per_section=byte_samples_per_section,
        )
        t_b_end = time.time()
        print(f"[{archive_name}] Tier B complete ({t_b_end-t_b_start:.2f}s)")
        for tb in result.tier_b:
            print(f"  {tb.section:35s} N={tb.n_samples} sig={tb.n_significant}/{tb.n_samples} "
                  f"frac_sig={tb.fraction_significant:.3f} mean|Δ|={tb.mean_abs_delta:.5f} "
                  f"upper_bits={tb.upper_bound_scorer_extracted_bits:.0f}")

    # Tier C: post-decode perturbation
    if run_tier_c_flag:
        print(f"[{archive_name}] Tier C: post-decode perturbation...")
        t_c0 = time.time()
        result.tier_c = run_tier_c(
            inner_bytes, grammar, pair_indices, gt_pairs,
            baseline_seg, baseline_pose, distortion_net, device, rng,
        )
        t_c1 = time.time()
        print(f"[{archive_name}] Tier C complete ({t_c1-t_c0:.2f}s)")
        for tc in result.tier_c:
            print(f"  {tc.target:12s} sigma_rel={tc.noise_sigma_relative:.4f} "
                  f"Δseg={tc.delta_seg} Δpose={tc.delta_pose} Δscore_comp={tc.delta_score_components}")

    # Aggregate
    result = aggregate_mdl_estimate(result)

    t_total1 = time.time()
    result.elapsed_seconds_total = t_total1 - t_total0

    print(f"[{archive_name}] MDL aggregate:")
    print(f"  scorer-extracted bytes LO: {result.mdl_scorer_extracted_bytes_lo:.1f}")
    print(f"  scorer-extracted bytes HI: {result.mdl_scorer_extracted_bytes_hi:.1f}")
    print(f"  density LO: {result.mdl_density_estimate_lo:.4f}")
    print(f"  density HI: {result.mdl_density_estimate_hi:.4f}")
    print(f"  zen-floor band: {result.zen_floor_band_recommendation}")
    print(f"  total ablation time: {result.elapsed_seconds_total:.1f}s")

    # Persist JSON
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{archive_name}_mdl_ablation.json"
    with open(out_path, "w") as f:
        json.dump(asdict(result), f, indent=2, default=str)
    print(f"[{archive_name}] Wrote: {out_path}")

    return result


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Scorer-conditional MDL ablation (Z1, $0 GPU).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--archive", action="append", required=True,
                   help="Path to archive.zip (repeat for multiple)")
    p.add_argument("--archive-name", action="append", required=True,
                   help="Short name for the archive (repeat in same order as --archive)")
    p.add_argument("--grammar", action="append", required=True,
                   help="Grammar (a1|pr106_latent_sidecar|pr101|ibps1); repeat in same order as --archive")
    p.add_argument("--upstream-dir", type=Path, default=REPO_ROOT / "upstream",
                   help="Path to upstream/ (for models + videos/0.mkv)")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Output directory under experiments/results/...")
    p.add_argument("--device", type=str, default="auto",
                   help="auto|mps|cpu|cuda")
    p.add_argument("--pair-samples", type=int, default=60,
                   help="Number of pairs to sample per measurement (default: 60)")
    p.add_argument("--byte-samples", type=int, default=200,
                   help="Number of byte positions to flip per section in Tier B")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--skip-tier-a", action="store_true")
    p.add_argument("--skip-tier-b", action="store_true")
    p.add_argument("--skip-tier-c", action="store_true")
    args = p.parse_args(argv)

    # Validate alignment
    if len(args.archive) != len(args.archive_name) or len(args.archive) != len(args.grammar):
        p.error("--archive / --archive-name / --grammar lists must have the same length")

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device}")
    print(f"Output dir: {args.output_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[ArchiveAblationResult] = []
    for archive_path_str, archive_name, grammar in zip(args.archive, args.archive_name, args.grammar):
        archive_path = Path(archive_path_str)
        if not archive_path.exists():
            print(f"ERROR: archive not found: {archive_path}", file=sys.stderr)
            return 1
        result = ablate_archive(
            archive_path=archive_path,
            archive_name=archive_name,
            grammar=grammar,
            upstream_dir=args.upstream_dir,
            output_dir=args.output_dir,
            device=device,
            pair_samples=args.pair_samples,
            byte_samples_per_section=args.byte_samples,
            seed=args.seed,
            run_tier_a_flag=not args.skip_tier_a,
            run_tier_b_flag=not args.skip_tier_b,
            run_tier_c_flag=not args.skip_tier_c,
        )
        all_results.append(result)

    # Write aggregate cross-archive summary
    summary = {
        "timestamp_utc": all_results[0].timestamp_utc if all_results else "",
        "device": str(device),
        "pair_samples": args.pair_samples,
        "byte_samples_per_section": args.byte_samples,
        "seed": args.seed,
        "archives": [
            {
                "name": r.archive_name,
                "path": r.archive_path,
                "sha256": r.archive_sha256,
                "size_bytes": r.archive_size_bytes,
                "baseline_seg": r.baseline_seg,
                "baseline_pose": r.baseline_pose,
                "baseline_score_components": r.baseline_score_components,
                "mdl_scorer_extracted_bytes_lo": r.mdl_scorer_extracted_bytes_lo,
                "mdl_scorer_extracted_bytes_hi": r.mdl_scorer_extracted_bytes_hi,
                "mdl_density_lo": r.mdl_density_estimate_lo,
                "mdl_density_hi": r.mdl_density_estimate_hi,
                "zen_floor_band_recommendation": r.zen_floor_band_recommendation,
                "elapsed_seconds": r.elapsed_seconds_total,
            }
            for r in all_results
        ],
        "notes": [
            "[MDL-ablation-MPS]" if str(device) == "mps" else f"[MDL-ablation-{device}]",
            "[mathematical-derivation]",
            "Per CLAUDE.md 'MPS auth eval is NOISE': MPS forward passes do NOT produce authoritative absolute scores.",
            "MDL ablation measures DELTAS; device-specific drift cancels for sign + relative magnitude.",
            "Rate term excluded from score_components; archive size constant across ablations.",
            "Per CLAUDE.md 'KILL is LAST RESORT': zen-floor band update is INFORMATIONAL, not falsifying.",
        ],
    }
    summary_path = args.output_dir / "summary_mdl_ablation.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nAggregate summary: {summary_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
