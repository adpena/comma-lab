# SPDX-License-Identifier: MIT
"""WAVE-3 PARSER-SAFE METHODOLOGY EXTENSION smoke (LOCAL macOS-CPU).

Per PARSER-SAFE SUBSET SMOKE landing (commit ``e3e198c9f``) Top-3
operator-routable #1 verbatim: apply the null-byte parser-safe analysis
methodology to 4 canonical IN-DOMAIN substrates (DP1 + VQ-VAE +
grayscale_lut + ATW V2). Sister extension of
``tools/run_parser_safe_subset_smoke.py`` which covered ONLY the fec6
frontier substrate.

The sister smoke's META-LESSON was: null-gradient is NECESSARY but NOT
SUFFICIENT for byte replaceability — replaceability ALSO requires the
byte be downstream of parser dispatch (i.e., NOT inside a
Brotli/LZMA/Huffman bitstream or a struct-packed wrapper field). On
fec6 the parser-safe subset was structurally EMPTY because all bytes
were inside compressed streams or struct fields.

This subagent (Phase 1) statically classifies each substrate's archive
sections into 4 region categories: struct_field / brotli_stream /
raw_byte_section / json_metadata. Phase 2 constructs a "parser-safe
subset" as the union of raw_byte_section regions (the only regions where
mutation does NOT corrupt the parser's bit-accurate decoder). Phase 3
classifies the parser-safe subset for SCORE-RELEVANCE — bytes in raw
sections may be parser-safe but still score-affecting (decoder
side-information; codebook indices; CDF tables). Phase 4 emits a
canonical equation #26 update if the substrate's parser-safe subset
reveals a NEW IN-DOMAIN context.

Per CLAUDE.md "MPS auth eval is NOISE" + "Submission auth eval — BOTH
CPU AND CUDA" non-negotiables: macOS-CPU smoke is observability-only;
contest-CPU paired Linux x86_64 anchor required for any promotion.
This smoke does NOT run paid GPU; it is STATIC analysis only (no
contest_auth_eval invocation).

Catalog #270 tool dispatch scope per "tac stays clean" + canonical
dispatch optimization protocol.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import struct
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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


class ParserSafeMethodologyExtensionError(RuntimeError):
    """Raised when the methodology extension pipeline cannot complete."""


# Region kind taxonomy. Parser-safe candidates are RAW_BYTE_SECTION
# (uncompressed bytes the parser reads byte-by-byte, e.g. int8 latent
# residuals; int16 codebook indices; fp16 CDF tables). All other kinds
# are parser-essential: byte mutation corrupts the parser's bit-accurate
# decoder OR the JSON parser's UTF-8 grammar.
KIND_STRUCT_FIELD = "struct_field"
KIND_BROTLI_STREAM = "brotli_stream"
KIND_RAW_BYTE_SECTION = "raw_byte_section"
KIND_JSON_METADATA = "json_metadata"

PARSER_ESSENTIAL_KINDS = (KIND_STRUCT_FIELD, KIND_BROTLI_STREAM, KIND_JSON_METADATA)
PARSER_SAFE_KINDS = (KIND_RAW_BYTE_SECTION,)

# Score-relevance taxonomy for parser-safe sections (Phase 3).
SCORE_RELEVANCE_SCORE_AFFECTING = "score_affecting"
SCORE_RELEVANCE_SCORE_OPAQUE = "score_opaque"
SCORE_RELEVANCE_UNKNOWN = "unknown"


@dataclass(frozen=True)
class SubstrateRegion:
    """One archive region's static classification."""

    region_name: str
    start_byte: int
    end_byte: int  # exclusive
    parser_kind: str  # one of KIND_*
    parser_essential: bool
    score_relevance: str  # one of SCORE_RELEVANCE_*
    role: str  # canonical optimization-role per *_SECTION_ROLES dicts
    rationale: str

    def size(self) -> int:
        return self.end_byte - self.start_byte

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SubstrateClassification:
    """One substrate's full region-level classification."""

    substrate_id: str
    archive_magic: str
    archive_bytes: int
    archive_sha256: str
    regions: tuple[SubstrateRegion, ...]
    parser_safe_subset_total_bytes: int
    parser_safe_score_affecting_bytes: int
    parser_safe_score_opaque_bytes: int
    parser_safe_unknown_bytes: int
    canonical_equation_26_eligible_contexts: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "substrate_id": self.substrate_id,
            "archive_magic": self.archive_magic,
            "archive_bytes": self.archive_bytes,
            "archive_sha256": self.archive_sha256,
            "regions": [r.as_dict() for r in self.regions],
            "parser_safe_subset_total_bytes": self.parser_safe_subset_total_bytes,
            "parser_safe_score_affecting_bytes": self.parser_safe_score_affecting_bytes,
            "parser_safe_score_opaque_bytes": self.parser_safe_score_opaque_bytes,
            "parser_safe_unknown_bytes": self.parser_safe_unknown_bytes,
            "canonical_equation_26_eligible_contexts": list(
                self.canonical_equation_26_eligible_contexts
            ),
        }


def _synthesize_dp1_archive() -> tuple[bytes, str]:
    """Synthesize a minimal valid DP1 archive (header + 4 sections)."""
    import numpy as np
    import torch
    from tac.substrates.pretrained_driving_prior.archive import pack_archive
    from tac.substrates.pretrained_driving_prior.codebook import (
        DashcamCodebook,
        LANE_CURVATURE_PCA_SHAPE,
        ROAD_PLANE_BASIS_SHAPE,
        SKY_HORIZON_PROFILE_SHAPE,
        VEHICLE_APPEARANCE_BASIS_SHAPE,
    )

    cb = DashcamCodebook(
        road_plane_basis=np.zeros(ROAD_PLANE_BASIS_SHAPE, dtype=np.int8),
        sky_horizon_profile=np.zeros(SKY_HORIZON_PROFILE_SHAPE, dtype=np.int8),
        lane_curvature_pca=np.zeros(LANE_CURVATURE_PCA_SHAPE, dtype=np.float16),
        vehicle_appearance_basis=np.zeros(
            VEHICLE_APPEARANCE_BASIS_SHAPE, dtype=np.int8
        ),
        metadata={
            "road_plane_scale": 1.0,
            "sky_horizon_scale": 1.0,
            "vehicle_scale": 1.0,
            "dataset_provenance": "Comma2k19",
            "distillation_version": 1,
            "license_tags": "MIT",
        },
    )
    sd = {"renderer.weight": torch.zeros(8, 8)}
    residual = bytes([0] * (2 * 4))
    blob = pack_archive(
        cb,
        sd,
        residual,
        {"a": 1},
        num_pairs=2,
        output_height=384,
        output_width=512,
        per_pair_bytes=4,
    )
    return blob, hashlib.sha256(blob).hexdigest()


def _synthesize_vqv1_archive() -> tuple[bytes, str]:
    """Synthesize a minimal valid VQV1 archive."""
    import torch
    from tac.substrates.vq_vae.archive import pack_archive

    sd = {"codebook": torch.zeros(8, 8), "decoder.weight": torch.zeros(8, 8)}
    indices = torch.zeros(2, 2, 4, 6, dtype=torch.int64)
    blob = pack_archive(sd, indices, {"a": 1}, codebook_size=8, embedding_dim=8)
    return blob, hashlib.sha256(blob).hexdigest()


def _synthesize_glv1_archive() -> tuple[bytes, str]:
    """Synthesize a minimal valid GLV1 archive."""
    import torch
    from tac.substrates.grayscale_lut.archive import pack_archive

    sd = {
        "pair_embedding": torch.zeros(2, 8),
        "stem.weight": torch.zeros(8, 8),
        "blocks.0.weight": torch.zeros(8, 8),
        "head_rgb_0.weight": torch.zeros(8, 8),
        "head_rgb_1.weight": torch.zeros(8, 8),
    }
    gs = torch.zeros(2, 1, 8, 12, dtype=torch.uint8)
    blob = pack_archive(
        sd,
        gs,
        {"a": 1},
        num_pairs=2,
        grayscale_downsample=4,
        embedding_dim=8,
        output_height=384,
        output_width=512,
    )
    return blob, hashlib.sha256(blob).hexdigest()


def _synthesize_atw2_archive() -> tuple[bytes, str]:
    """Synthesize a minimal valid ATW2 archive."""
    import torch
    from tac.substrates.atw_codec_v2.archive import pack_archive

    enc = {"e.weight": torch.zeros(8, 8)}
    dec = {"d.weight": torch.zeros(8, 8)}
    wzh = {"wz.weight": torch.zeros(8, 8)}
    dst = {"g1.weight": torch.zeros(5, 8)}
    latents = torch.zeros(2, 4)
    prior = torch.zeros(2, 16)
    cdf = torch.zeros(5, 256)
    blob = pack_archive(enc, dec, wzh, dst, latents, prior, cdf, {"a": 1})
    return blob, hashlib.sha256(blob).hexdigest()


def classify_dp1(archive_bytes: bytes) -> SubstrateClassification:
    """Classify DP1 archive sections per the inflate parser grammar."""
    from tac.substrates.pretrained_driving_prior.archive import (
        DP1_HEADER_SIZE,
        DP1_SECTION_ROLES,
        parse_dp1_archive_bytes,
    )

    sections = parse_dp1_archive_bytes(archive_bytes)
    regions: list[SubstrateRegion] = []
    # All DP1 sections are parser-essential:
    # header = struct_field; codebook/renderer/residual = brotli; meta = json
    sec_kind = {
        "dp1_header": KIND_STRUCT_FIELD,
        "codebook_blob": KIND_BROTLI_STREAM,
        "renderer_blob": KIND_BROTLI_STREAM,
        "residual_blob": KIND_BROTLI_STREAM,
        "meta_blob": KIND_JSON_METADATA,
    }
    sec_rationale = {
        "dp1_header": "28-byte struct (MAGIC + VERSION + 5 u16 + 4 u32). Byte mutation breaks struct.unpack.",
        "codebook_blob": "serialize_codebook output (sectioned brotli streams). Byte mutation breaks brotli.decompress.",
        "renderer_blob": "brotli(pickle(fp16 state_dict)). Byte mutation breaks brotli + pickle.",
        "residual_blob": "brotli(int8 per-pair residual). Byte mutation breaks brotli.",
        "meta_blob": "sorted-keys utf-8 JSON. Byte mutation breaks JSON parser.",
    }
    for name, (start, length) in sections.items():
        kind = sec_kind[name]
        regions.append(
            SubstrateRegion(
                region_name=name,
                start_byte=start,
                end_byte=start + length,
                parser_kind=kind,
                parser_essential=(kind in PARSER_ESSENTIAL_KINDS),
                # All DP1 sections parser-essential -> score relevance irrelevant
                # for parser-safe subset; tag UNKNOWN for non-safe regions.
                score_relevance=SCORE_RELEVANCE_UNKNOWN,
                role=DP1_SECTION_ROLES.get(name, "control_or_metadata"),
                rationale=sec_rationale[name],
            )
        )
    assert DP1_HEADER_SIZE == 28
    return _aggregate_substrate(
        substrate_id="dp1_pretrained_driving_prior",
        archive_magic="DP1\\x00",
        archive_bytes_len=len(archive_bytes),
        archive_sha256=hashlib.sha256(archive_bytes).hexdigest(),
        regions=tuple(regions),
        canonical_equation_26_eligible_contexts=(
            "dp1_codebook_bytes",  # already in _INCLUDED_CONTEXTS
            "comma2k19_ood_derived_basis_replacement",  # already in _INCLUDED_CONTEXTS
        ),
    )


def classify_vqv1(archive_bytes: bytes) -> SubstrateClassification:
    """Classify VQV1 archive sections per the inflate parser grammar."""
    if archive_bytes[:4] != b"VQV1":
        raise ParserSafeMethodologyExtensionError(
            f"VQV1 magic mismatch: {archive_bytes[:4]!r}"
        )
    VQV1_HEADER_FMT = "<4sBHHHHHIII"
    VQV1_HEADER_SIZE = struct.calcsize(VQV1_HEADER_FMT)
    hdr = struct.unpack(VQV1_HEADER_FMT, archive_bytes[:VQV1_HEADER_SIZE])
    decoder_len, indices_len, meta_len = int(hdr[7]), int(hdr[8]), int(hdr[9])

    regions: list[SubstrateRegion] = []
    cursor = 0
    regions.append(
        SubstrateRegion(
            region_name="vqv1_header",
            start_byte=cursor,
            end_byte=cursor + VQV1_HEADER_SIZE,
            parser_kind=KIND_STRUCT_FIELD,
            parser_essential=True,
            score_relevance=SCORE_RELEVANCE_UNKNOWN,
            role="control_or_metadata",
            rationale="27-byte struct (MAGIC + VERSION + 5 u16 + 3 u32). Byte mutation breaks struct.unpack.",
        )
    )
    cursor += VQV1_HEADER_SIZE
    regions.append(
        SubstrateRegion(
            region_name="decoder_blob",
            start_byte=cursor,
            end_byte=cursor + decoder_len,
            parser_kind=KIND_BROTLI_STREAM,
            parser_essential=True,
            score_relevance=SCORE_RELEVANCE_UNKNOWN,
            role="decoder_weight_stream",
            rationale="brotli(pickle(fp16 state_dict)) of codebook + decoder.*. Byte mutation breaks brotli.",
        )
    )
    cursor += decoder_len
    # KEY REGION: indices_blob is RAW int16 (no brotli). Parser reads
    # byte-by-byte; mutation does NOT corrupt parser. BUT the indices
    # select codebook entries -> decoder output -> score; mutation IS
    # score-affecting (the decoder dequantizes from indices to embedded
    # vectors and renders).
    regions.append(
        SubstrateRegion(
            region_name="indices_blob",
            start_byte=cursor,
            end_byte=cursor + indices_len,
            parser_kind=KIND_RAW_BYTE_SECTION,
            parser_essential=False,
            score_relevance=SCORE_RELEVANCE_SCORE_AFFECTING,
            role="latent_stream",
            rationale=(
                "RAW int16 packed (num_pairs * 2 * H_GRID * W_GRID * 2). "
                "Parser-safe (no brotli/LZMA) but indices SELECT codebook "
                "entries which feed the decoder render path -> "
                "score-affecting at inflate time. Catalog #220 sister "
                "anchor: operational mechanism active."
            ),
        )
    )
    cursor += indices_len
    regions.append(
        SubstrateRegion(
            region_name="meta_blob",
            start_byte=cursor,
            end_byte=cursor + meta_len,
            parser_kind=KIND_JSON_METADATA,
            parser_essential=True,
            score_relevance=SCORE_RELEVANCE_UNKNOWN,
            role="control_or_metadata",
            rationale="separators=(',', ':') sorted-keys utf-8 JSON. Byte mutation breaks JSON parser.",
        )
    )

    return _aggregate_substrate(
        substrate_id="vq_vae",
        archive_magic="VQV1",
        archive_bytes_len=len(archive_bytes),
        archive_sha256=hashlib.sha256(archive_bytes).hexdigest(),
        regions=tuple(regions),
        canonical_equation_26_eligible_contexts=(
            # Indices are codebook lookup indices, which fits the canonical
            # "procedural_codebook_as_lookup_table" or
            # "intermediate_transform_quantizer" already-INCLUDED contexts.
            # But the indices themselves are score-affecting -> direct
            # mutation falls into the residual-hybrid or
            # direct-byte-substitution side of the domain refinement.
            "intermediate_transform_quantizer",  # already in _INCLUDED_CONTEXTS
            "procedural_codebook_as_lookup_table",  # already in _INCLUDED_CONTEXTS
        ),
    )


def classify_glv1(archive_bytes: bytes) -> SubstrateClassification:
    """Classify GLV1 archive sections per the inflate parser grammar."""
    if archive_bytes[:4] != b"GLV1":
        raise ParserSafeMethodologyExtensionError(
            f"GLV1 magic mismatch: {archive_bytes[:4]!r}"
        )
    GLV1_HEADER_FMT = "<4sBHHHBHHHIII"
    GLV1_HEADER_SIZE = struct.calcsize(GLV1_HEADER_FMT)
    hdr = struct.unpack(GLV1_HEADER_FMT, archive_bytes[:GLV1_HEADER_SIZE])
    decoder_len, grayscale_len, meta_len = int(hdr[9]), int(hdr[10]), int(hdr[11])

    regions: list[SubstrateRegion] = []
    cursor = 0
    regions.append(
        SubstrateRegion(
            region_name="glv1_header",
            start_byte=cursor,
            end_byte=cursor + GLV1_HEADER_SIZE,
            parser_kind=KIND_STRUCT_FIELD,
            parser_essential=True,
            score_relevance=SCORE_RELEVANCE_UNKNOWN,
            role="control_or_metadata",
            rationale="30-byte struct. Byte mutation breaks struct.unpack.",
        )
    )
    cursor += GLV1_HEADER_SIZE
    regions.append(
        SubstrateRegion(
            region_name="decoder_blob",
            start_byte=cursor,
            end_byte=cursor + decoder_len,
            parser_kind=KIND_BROTLI_STREAM,
            parser_essential=True,
            score_relevance=SCORE_RELEVANCE_UNKNOWN,
            role="decoder_weight_stream",
            rationale="brotli(pickle(fp16 state_dict)). Byte mutation breaks brotli.",
        )
    )
    cursor += decoder_len
    regions.append(
        SubstrateRegion(
            region_name="grayscale_blob",
            start_byte=cursor,
            end_byte=cursor + grayscale_len,
            parser_kind=KIND_BROTLI_STREAM,
            parser_essential=True,
            score_relevance=SCORE_RELEVANCE_UNKNOWN,
            role="latent_stream",
            rationale="brotli(uint8 grayscale stream). Byte mutation breaks brotli.",
        )
    )
    cursor += grayscale_len
    regions.append(
        SubstrateRegion(
            region_name="meta_blob",
            start_byte=cursor,
            end_byte=cursor + meta_len,
            parser_kind=KIND_JSON_METADATA,
            parser_essential=True,
            score_relevance=SCORE_RELEVANCE_UNKNOWN,
            role="control_or_metadata",
            rationale="separators=(',', ':') sorted-keys utf-8 JSON. Byte mutation breaks JSON parser.",
        )
    )

    return _aggregate_substrate(
        substrate_id="grayscale_lut",
        archive_magic="GLV1",
        archive_bytes_len=len(archive_bytes),
        archive_sha256=hashlib.sha256(archive_bytes).hexdigest(),
        regions=tuple(regions),
        canonical_equation_26_eligible_contexts=(
            # GLV1 has no current raw bytes; the canonical chroma LUT
            # context applies only if a future GLV2 variant adds a
            # parser-visible LUT section (per design memo distillation
            # procedural variant).
            "chroma_lut_replacement",  # already in _INCLUDED_CONTEXTS (future GLV2)
        ),
    )


def classify_atw2(archive_bytes: bytes) -> SubstrateClassification:
    """Classify ATW2 archive sections per the inflate parser grammar."""
    from tac.substrates.atw_codec_v2.archive import (
        ATW2_HEADER_SIZE,
        ATW2_SECTION_ROLES,
        parse_atw2_archive_bytes,
    )

    sections = parse_atw2_archive_bytes(archive_bytes)
    regions: list[SubstrateRegion] = []

    # ATW2 has 4 brotli sections + 3 RAW sections + 1 header + 1 meta JSON.
    raw_section_names = {
        "latent_residual_blob": (
            "int8 z_residual",
            "RAW int8 packed (num_pairs * latent_dim). Parser-safe (no brotli) "
            "but z_residual feeds Wyner-Ziv side-info head reconstruction -> "
            "score-affecting at inflate time. Catalog #220 sister anchor: "
            "operational mechanism active.",
            SCORE_RELEVANCE_SCORE_AFFECTING,
            "atw_v2_codec_quantizer_lut",  # canonical context already in _INCLUDED_CONTEXTS
        ),
        "class_prior_table_blob": (
            "fp16 class prior table",
            "RAW fp16 (num_pairs * scorer_class_prior_dim * 2). Parser-safe "
            "(no brotli) but consumed by inflate-time decoder as side info "
            "-> score-affecting.",
            SCORE_RELEVANCE_SCORE_AFFECTING,
            "atw_v2_codec_quantizer_lut",
        ),
        "cdf_table_blob": (
            "fp16 B3 scorer-conditional CDF table",
            "RAW fp16 (num_classes * num_symbols * 2). Parser-safe but CDF "
            "table feeds B3 entropy-coding decoder -> score-affecting.",
            SCORE_RELEVANCE_SCORE_AFFECTING,
            "atw_v2_codec_quantizer_lut",
        ),
    }
    brotli_section_names = {
        "encoder_blob",
        "decoder_blob",
        "wz_head_blob",
        "distill_head_blob",
    }
    header_section_names = {"atw2_header"}
    meta_section_names = {"meta_blob"}

    for name, (start, length) in sections.items():
        if name in raw_section_names:
            _, rationale, score_rel, _ = raw_section_names[name]
            regions.append(
                SubstrateRegion(
                    region_name=name,
                    start_byte=start,
                    end_byte=start + length,
                    parser_kind=KIND_RAW_BYTE_SECTION,
                    parser_essential=False,
                    score_relevance=score_rel,
                    role=ATW2_SECTION_ROLES.get(name, "decoder_side_information"),
                    rationale=rationale,
                )
            )
        elif name in brotli_section_names:
            regions.append(
                SubstrateRegion(
                    region_name=name,
                    start_byte=start,
                    end_byte=start + length,
                    parser_kind=KIND_BROTLI_STREAM,
                    parser_essential=True,
                    score_relevance=SCORE_RELEVANCE_UNKNOWN,
                    role=ATW2_SECTION_ROLES.get(name, "decoder_weight_stream"),
                    rationale="brotli q=9 length-prefixed fp16 state_dict. Byte mutation breaks brotli.",
                )
            )
        elif name in header_section_names:
            regions.append(
                SubstrateRegion(
                    region_name=name,
                    start_byte=start,
                    end_byte=start + length,
                    parser_kind=KIND_STRUCT_FIELD,
                    parser_essential=True,
                    score_relevance=SCORE_RELEVANCE_UNKNOWN,
                    role="control_or_metadata",
                    rationale="48-byte struct. Byte mutation breaks struct.unpack.",
                )
            )
        elif name in meta_section_names:
            regions.append(
                SubstrateRegion(
                    region_name=name,
                    start_byte=start,
                    end_byte=start + length,
                    parser_kind=KIND_JSON_METADATA,
                    parser_essential=True,
                    score_relevance=SCORE_RELEVANCE_UNKNOWN,
                    role="control_or_metadata",
                    rationale="sorted-keys utf-8 JSON with atw_v2_codec_meta. Byte mutation breaks JSON parser.",
                )
            )
        else:
            raise ParserSafeMethodologyExtensionError(
                f"ATW2 unknown section {name!r}; grammar drift"
            )

    assert ATW2_HEADER_SIZE == 48
    return _aggregate_substrate(
        substrate_id="atw_codec_v2",
        archive_magic="ATW2",
        archive_bytes_len=len(archive_bytes),
        archive_sha256=hashlib.sha256(archive_bytes).hexdigest(),
        regions=tuple(regions),
        canonical_equation_26_eligible_contexts=(
            "atw_v2_codec_quantizer_lut",  # already in _INCLUDED_CONTEXTS
        ),
    )


def _aggregate_substrate(
    substrate_id: str,
    archive_magic: str,
    archive_bytes_len: int,
    archive_sha256: str,
    regions: tuple[SubstrateRegion, ...],
    canonical_equation_26_eligible_contexts: tuple[str, ...],
) -> SubstrateClassification:
    """Build the per-substrate aggregate row from region-level results."""
    parser_safe_total = 0
    parser_safe_score_affecting = 0
    parser_safe_score_opaque = 0
    parser_safe_unknown = 0
    for r in regions:
        if r.parser_kind in PARSER_SAFE_KINDS:
            parser_safe_total += r.size()
            if r.score_relevance == SCORE_RELEVANCE_SCORE_AFFECTING:
                parser_safe_score_affecting += r.size()
            elif r.score_relevance == SCORE_RELEVANCE_SCORE_OPAQUE:
                parser_safe_score_opaque += r.size()
            else:
                parser_safe_unknown += r.size()
    return SubstrateClassification(
        substrate_id=substrate_id,
        archive_magic=archive_magic,
        archive_bytes=archive_bytes_len,
        archive_sha256=archive_sha256,
        regions=regions,
        parser_safe_subset_total_bytes=parser_safe_total,
        parser_safe_score_affecting_bytes=parser_safe_score_affecting,
        parser_safe_score_opaque_bytes=parser_safe_score_opaque,
        parser_safe_unknown_bytes=parser_safe_unknown,
        canonical_equation_26_eligible_contexts=canonical_equation_26_eligible_contexts,
    )


def classify_all_substrates() -> list[SubstrateClassification]:
    """Run Phase 1-3 classification for all 4 in-scope substrates."""
    results: list[SubstrateClassification] = []
    dp1_bytes, _ = _synthesize_dp1_archive()
    results.append(classify_dp1(dp1_bytes))
    vqv1_bytes, _ = _synthesize_vqv1_archive()
    results.append(classify_vqv1(vqv1_bytes))
    glv1_bytes, _ = _synthesize_glv1_archive()
    results.append(classify_glv1(glv1_bytes))
    atw2_bytes, _ = _synthesize_atw2_archive()
    results.append(classify_atw2(atw2_bytes))
    return results


def compute_aggregate_verdict(
    results: list[SubstrateClassification],
) -> tuple[str, str]:
    """Phase 4 — aggregate verdict across the 4 substrates.

    Verdict taxonomy:
    * METHODOLOGY_EXTENSION_ALL_EMPTY — every substrate has parser_safe_total = 0;
      REINFORCE Catalog #344 _EXCLUDED_CONTEXTS extension; structural
      exhaustion of canonical equation #26 IN-DOMAIN scope on the existing
      4 substrates.
    * METHODOLOGY_EXTENSION_MIXED_PARSER_SAFE_BUT_SCORE_AFFECTING — at least
      one substrate has non-empty parser-safe subset, but ALL parser-safe
      bytes are score-affecting (decoder side-information). Per CLAUDE.md
      "HNeRV / leaderboard-implementation parity discipline" L6 score-domain
      Lagrangian: parser-safe-but-score-affecting bytes are NOT canonical
      equation #26 IN-DOMAIN candidates (the equation predicts REPLACEMENT
      savings for score-OPAQUE bytes). The right canonical equation surface
      is the existing IN-DOMAIN context (e.g. atw_v2_codec_quantizer_lut /
      intermediate_transform_quantizer) which already covers procedural-
      replacement of these decoder-side-information bytes WHEN the
      substrate's training co-optimizes the replacement.
    * METHODOLOGY_EXTENSION_PARSER_SAFE_AND_SCORE_OPAQUE — at least one
      substrate has parser-safe + score-opaque bytes; STRONG NEW canonical
      equation #26 context candidate; route per CLAUDE.md "Forbidden
      premature KILL" + sister Catalog #325 per-substrate symposium.
    """

    total_parser_safe = sum(r.parser_safe_subset_total_bytes for r in results)
    total_score_affecting = sum(
        r.parser_safe_score_affecting_bytes for r in results
    )
    total_score_opaque = sum(r.parser_safe_score_opaque_bytes for r in results)

    if total_parser_safe == 0:
        return (
            "METHODOLOGY_EXTENSION_ALL_EMPTY",
            "All 4 in-scope substrates have ZERO parser-safe bytes outside "
            "compressed streams or struct fields. REINFORCES Catalog #344 "
            "_EXCLUDED_CONTEXTS extension: no NEW IN-DOMAIN canonical "
            "equation #26 contexts surface from these substrates beyond "
            "the already-included set (intermediate_transform_quantizer / "
            "chroma_lut_replacement / nscs06_v8_chroma_lut / "
            "atw_v2_codec_quantizer_lut / tt5l_transformer_tokens / "
            "dp1_codebook_bytes / class_anchor_replacement / etc.). "
            "Structural exhaustion of the direct-byte-substitution surface "
            "on these substrates; alternative paths require architectural "
            "rescope (e.g. expose a new parser-visible LUT section by "
            "design).",
        )
    if total_score_opaque > 0:
        return (
            "METHODOLOGY_EXTENSION_PARSER_SAFE_AND_SCORE_OPAQUE",
            f"{total_score_opaque} parser-safe + score-opaque bytes found "
            "across in-scope substrates. STRONG NEW canonical equation "
            "#26 context candidate. Per CLAUDE.md sister Catalog #325 "
            "per-substrate symposium + Catalog #324 post-training Tier-C "
            "validation BEFORE any paid dispatch. Operator-routable: name "
            "the substrate + section + propose new context token.",
        )
    return (
        "METHODOLOGY_EXTENSION_MIXED_PARSER_SAFE_BUT_SCORE_AFFECTING",
        f"{total_parser_safe} parser-safe bytes found across in-scope "
        f"substrates; ALL {total_score_affecting} bytes are score-"
        "affecting (decoder side-information; codebook indices / latent "
        "residuals / CDF tables). Per CLAUDE.md HNeRV parity discipline "
        "L6 (score-domain Lagrangian): parser-safe-but-score-affecting "
        "bytes are NOT canonical equation #26 IN-DOMAIN candidates for "
        "direct byte substitution (the equation predicts REPLACEMENT "
        "savings for score-OPAQUE bytes only). The right surface is the "
        "existing IN-DOMAIN context (atw_v2_codec_quantizer_lut / "
        "intermediate_transform_quantizer / procedural_codebook_as_lookup_table) "
        "which already covers procedural-replacement of these bytes WHEN "
        "the substrate's training co-optimizes the replacement (sister "
        "PROCEDURAL VARIANT BUILDs land this surface; this smoke confirms "
        "STATIC structural compatibility).",
    )


def _write_smoke_result_json(
    output_dir: Path,
    results: list[SubstrateClassification],
    verdict_label: str,
    verdict_rationale: str,
) -> Path:
    """Emit smoke_result.json with full provenance per Catalog #323."""
    # Provenance archive sha is the union (concat) of the 4 substrate sha
    # prefixes; canonical helper requires a single non-empty archive sha.
    # Use the first substrate's sha as the canonical anchor (DP1).
    primary_sha = results[0].archive_sha256

    try:
        source_path_str = str(output_dir.relative_to(REPO_ROOT))
    except ValueError:
        source_path_str = str(output_dir)

    provenance = build_provenance_for_macos_cpu_advisory(
        archive_sha256=primary_sha,
        source_path=source_path_str,
    )
    payload = {
        "schema_version": "parser_safe_methodology_extension_smoke_v1_20260520",
        "smoke_at_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
        "in_scope_substrates": [
            r.substrate_id for r in results
        ],
        "substrate_count": len(results),
        "per_substrate_classification": [r.as_dict() for r in results],
        "aggregate_parser_safe_total_bytes": sum(
            r.parser_safe_subset_total_bytes for r in results
        ),
        "aggregate_parser_safe_score_affecting_bytes": sum(
            r.parser_safe_score_affecting_bytes for r in results
        ),
        "aggregate_parser_safe_score_opaque_bytes": sum(
            r.parser_safe_score_opaque_bytes for r in results
        ),
        "aggregate_parser_safe_unknown_bytes": sum(
            r.parser_safe_unknown_bytes for r in results
        ),
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
            "#220 substrate L1+ operational mechanism",
            "#229 premise verification",
            "#272 byte-mutation smoke methodology",
            "#287 placeholder-rationale rejection",
            "#323 canonical Provenance umbrella",
            "#344 canonical equation cross-ref",
            "#356 per-axis decomposition contract",
        ],
        "canonical_equation_cross_ref": (
            "procedural_codebook_from_seed_compression_savings_v1 (Catalog "
            "#344 registry #26); extends sister PARSER-SAFE SUBSET SMOKE "
            "methodology (commit e3e198c9f) to 4 IN-DOMAIN substrates"
        ),
        "sister_smoke_context": {
            "sister_smoke_commit": "e3e198c9f",
            "sister_smoke_substrate": "fec6 PR101 frontier",
            "sister_smoke_verdict": "PARSER_SAFE_SUBSET_EMPTY",
            "sister_smoke_meta_lesson": (
                "null-gradient is NECESSARY but NOT SUFFICIENT for byte "
                "replaceability; this extension generalizes the methodology "
                "from a single-substrate verdict to a 4-substrate "
                "comparative classification"
            ),
            "this_smoke_extension_role": (
                "static structural classification across 4 IN-DOMAIN "
                "substrates per Top-3 #1 directive"
            ),
        },
    }
    out_path = output_dir / "smoke_result.json"
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    return out_path


def _write_smoke_result_md(
    output_dir: Path,
    results: list[SubstrateClassification],
    verdict_label: str,
    verdict_rationale: str,
) -> Path:
    """Emit human-readable smoke_result.md."""
    lines = [
        "<!-- HISTORICAL_SCORE_LITERAL_OK:macos_cpu_advisory_smoke_not_score_truth_parser_safe_methodology_extension_2026-05-20 -->",
        "# WAVE-3 PARSER-SAFE METHODOLOGY EXTENSION smoke",
        "",
        f"- **smoke_at_utc**: {_dt.datetime.now(tz=_dt.timezone.utc).isoformat()}",
        f"- **in-scope substrates**: {len(results)} ({', '.join(r.substrate_id for r in results)})",
        "- **axis tag**: `[macOS-CPU advisory]` (NEVER promotable per Catalog #192)",
        "- **$ spent**: $0 (LOCAL static analysis — no contest_auth_eval)",
        "",
        "## Per-substrate region classification",
        "",
    ]
    for r in results:
        lines.extend(
            [
                f"### {r.substrate_id} (magic={r.archive_magic}, archive {r.archive_bytes:,} bytes, sha[:16]={r.archive_sha256[:16]})",
                "",
                "| region | byte range | size | parser_kind | parser_essential | score_relevance | role |",
                "|---|---|---:|---|---|---|---|",
            ]
        )
        for region in r.regions:
            lines.append(
                f"| `{region.region_name}` | "
                f"[{region.start_byte}, {region.end_byte}) | "
                f"{region.size():,} | {region.parser_kind} | "
                f"{region.parser_essential} | {region.score_relevance} | "
                f"{region.role} |"
            )
        lines.extend(
            [
                "",
                f"**Parser-safe subset**: {r.parser_safe_subset_total_bytes:,} bytes "
                f"(score-affecting: {r.parser_safe_score_affecting_bytes:,}, "
                f"score-opaque: {r.parser_safe_score_opaque_bytes:,}, "
                f"unknown: {r.parser_safe_unknown_bytes:,})",
                "",
                f"**Canonical equation #26 eligible contexts**: "
                f"{list(r.canonical_equation_26_eligible_contexts) or '(none)'}",
                "",
            ]
        )

    lines.extend(
        [
            "## Comparative table",
            "",
            "| substrate | archive bytes | parser-safe bytes | score-affecting | score-opaque | unknown |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for r in results:
        lines.append(
            f"| `{r.substrate_id}` | {r.archive_bytes:,} | "
            f"{r.parser_safe_subset_total_bytes:,} | "
            f"{r.parser_safe_score_affecting_bytes:,} | "
            f"{r.parser_safe_score_opaque_bytes:,} | "
            f"{r.parser_safe_unknown_bytes:,} |"
        )
    lines.append("")

    lines.extend(
        [
            "## Aggregate verdict",
            "",
            f"**Verdict**: `{verdict_label}`",
            "",
            f"**Rationale**: {verdict_rationale}",
            "",
            "## Sister smoke cascade context",
            "",
            "- Sister smoke commit: `e3e198c9f` (PARSER-SAFE SUBSET SMOKE on fec6 frontier — PARSER_SAFE_SUBSET_EMPTY)",
            "- Sister META-LESSON: null-gradient is NECESSARY but NOT SUFFICIENT for byte replaceability",
            "- This extension's role: static structural classification across 4 IN-DOMAIN substrates",
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
            "WAVE-3 PARSER-SAFE METHODOLOGY EXTENSION smoke (LOCAL "
            "macOS-CPU). Extends sister PARSER-SAFE SUBSET SMOKE (commit "
            "e3e198c9f) to 4 IN-DOMAIN substrates: DP1 + VQ-VAE + "
            "grayscale_lut + ATW V2. STATIC analysis only; no paid "
            "GPU; no contest_auth_eval. Per CLAUDE.md MPS auth eval "
            "non-negotiable: macOS-CPU is observability-only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: experiments/results/parser_safe_methodology_extension_smoke_<utc>/)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Plan only; do not write artifacts",
    )
    args = parser.parse_args(argv)

    results = classify_all_substrates()
    verdict_label, verdict_rationale = compute_aggregate_verdict(results)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "in_scope_substrates": [r.substrate_id for r in results],
                    "aggregate_parser_safe_total_bytes": sum(
                        r.parser_safe_subset_total_bytes for r in results
                    ),
                    "aggregate_parser_safe_score_affecting_bytes": sum(
                        r.parser_safe_score_affecting_bytes for r in results
                    ),
                    "aggregate_parser_safe_score_opaque_bytes": sum(
                        r.parser_safe_score_opaque_bytes for r in results
                    ),
                    "verdict_label": verdict_label,
                },
                indent=2,
            )
        )
        return 0

    utc = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_dir = args.output_dir or (
        REPO_ROOT
        / f"experiments/results/parser_safe_methodology_extension_smoke_{utc}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = _write_smoke_result_json(
        output_dir, results, verdict_label, verdict_rationale
    )
    md_path = _write_smoke_result_md(
        output_dir, results, verdict_label, verdict_rationale
    )

    print("")
    print("=== PARSER-SAFE METHODOLOGY EXTENSION SMOKE RESULT ===")
    print(f"In-scope substrates: {len(results)}")
    print(
        f"Aggregate parser-safe total: "
        f"{sum(r.parser_safe_subset_total_bytes for r in results)} bytes"
    )
    print(
        f"  score-affecting: "
        f"{sum(r.parser_safe_score_affecting_bytes for r in results)}"
    )
    print(
        f"  score-opaque: "
        f"{sum(r.parser_safe_score_opaque_bytes for r in results)}"
    )
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
