# SPDX-License-Identifier: MIT
"""Parser-visible procedural replacement surface matrix.

This module is the small source-grounded bridge between the FEC6
``PARSER_SAFE_SUBSET_EMPTY`` result and the next procedural-codebook
substrate work. It separates three different concepts that were easy to
conflate in prose:

* raw-byte parser safety: mutating bytes cannot break the container parser;
* whole-section replacement: a parser can intentionally swap an entire
  section for seed-derived content;
* score safety: still unproven until a paired scorer smoke lands.

The matrix is advisory. It produces no score claim and no promotion-ready
state; it only ranks where canonical equation #26 can plausibly apply next.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from tac.canonical_equations.procedural_codebook_savings import (
    CANONICAL_RATE_DENOM_BYTES,
    CANONICAL_RATE_MULTIPLIER,
    validate_context_is_in_domain,
)


DEFAULT_SEED_BYTES = 32


@dataclass(frozen=True)
class ProceduralReplacementSurface:
    """One candidate region for seed-derived procedural replacement."""

    substrate_id: str
    surface_id: str
    context: str
    candidate_status: str
    current_archive_surface: bool
    parser_visible: bool
    raw_byte_mutation_parse_safe: bool
    whole_section_replacement_surface: bool
    requires_archive_adapter: bool
    original_payload_bytes: int
    seed_bytes: int
    evidence_grade: str
    byte_basis: str
    blocker: str
    rationale: str

    @property
    def context_in_domain(self) -> bool:
        return validate_context_is_in_domain(self.context, raise_on_excluded=False)

    @property
    def predicted_bytes_saved(self) -> int:
        if self.candidate_status == "BLOCKED_PARSER_SAFE_SUBSET_EMPTY":
            return 0
        if self.original_payload_bytes <= self.seed_bytes:
            return 0
        return self.original_payload_bytes - self.seed_bytes

    @property
    def predicted_delta_s(self) -> float:
        return (
            -CANONICAL_RATE_MULTIPLIER
            * self.predicted_bytes_saved
            / CANONICAL_RATE_DENOM_BYTES
        )

    @property
    def actionable_now(self) -> bool:
        return (
            self.context_in_domain
            and self.parser_visible
            and self.whole_section_replacement_surface
            and self.candidate_status in {"READY_TO_PAIR_SMOKE", "DESIGN_READY_DEFERRED"}
        )

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["context_in_domain"] = self.context_in_domain
        payload["predicted_bytes_saved"] = self.predicted_bytes_saved
        payload["predicted_delta_s"] = self.predicted_delta_s
        payload["actionable_now"] = self.actionable_now
        payload["score_claim"] = False
        payload["promotion_eligible"] = False
        payload["rank_or_kill_eligible"] = False
        payload["ready_for_exact_eval_dispatch"] = False
        return payload


def build_default_surface_matrix() -> tuple[ProceduralReplacementSurface, ...]:
    """Return the current source-grounded procedural surface matrix.

    Byte counts are intentionally conservative and labeled by ``byte_basis``.
    Empirical/runtime score movement still requires paired contest-CUDA and
    contest-CPU auth-eval; these rows are only routing signals.
    """

    return (
        ProceduralReplacementSurface(
            substrate_id="atw_codec_v2",
            surface_id="cdf_table_blob",
            context="atw_v2_codec_quantizer_lut",
            candidate_status="DESIGN_READY_DEFERRED",
            current_archive_surface=True,
            parser_visible=True,
            raw_byte_mutation_parse_safe=True,
            whole_section_replacement_surface=True,
            requires_archive_adapter=False,
            original_payload_bytes=5 * 256 * 2,
            seed_bytes=DEFAULT_SEED_BYTES,
            evidence_grade="source_formula_empirical_bytes",
            byte_basis="ATW2 cdf_classes * cdf_symbols * fp16_bytes = 5*256*2",
            blocker="D4 predecessor verdict and Variant-C scoping still gate paid dispatch",
            rationale=(
                "ATW2 stores cdf_table_blob as a raw fp16 table after parser dispatch; "
                "it is the cleanest current parser-visible equation #26 surface."
            ),
        ),
        ProceduralReplacementSurface(
            substrate_id="pretrained_driving_prior_dp1",
            surface_id="codebook_blob",
            context="comma2k19_ood_derived_basis_replacement",
            candidate_status="READY_TO_PAIR_SMOKE",
            current_archive_surface=True,
            parser_visible=True,
            raw_byte_mutation_parse_safe=False,
            whole_section_replacement_surface=True,
            requires_archive_adapter=True,
            original_payload_bytes=4096,
            seed_bytes=DEFAULT_SEED_BYTES,
            evidence_grade="design_default_raw_codebook_bytes",
            byte_basis="DP1 procedural variant default codebook shape (1024,4) uint8",
            blocker="paired contest-CUDA + contest-CPU smoke not yet landed",
            rationale=(
                "DP1 exposes a top-level codebook section. Raw bytes are brotli-coded, "
                "so arbitrary mutation is not parser-safe, but whole-section seed "
                "replacement is the intended procedural adapter path."
            ),
        ),
        ProceduralReplacementSurface(
            substrate_id="vq_vae",
            surface_id="codebook_inside_decoder_blob",
            context="intermediate_transform_quantizer",
            candidate_status="ADAPTER_REQUIRED",
            current_archive_surface=False,
            parser_visible=False,
            raw_byte_mutation_parse_safe=False,
            whole_section_replacement_surface=False,
            requires_archive_adapter=True,
            original_payload_bytes=512 * 8 * 2,
            seed_bytes=DEFAULT_SEED_BYTES,
            evidence_grade="source_formula_empirical_bytes",
            byte_basis="VQ-VAE codebook_size * embedding_dim * fp16_bytes = 512*8*2",
            blocker="canonical VQV1 stores codebook inside brotli'd decoder_blob; procedural envelope must be used",
            rationale=(
                "The VQ-VAE codebook is a strong equation #26 context, but not a "
                "raw parser-visible region in canonical VQV1. It must move through "
                "the procedural decoder-blob envelope, not null-byte mutation."
            ),
        ),
        ProceduralReplacementSurface(
            substrate_id="atw_codec_v2",
            surface_id="class_prior_table_blob",
            context="class_anchor_replacement",
            candidate_status="REQUIRES_SIGNAL_PRESERVATION_PROBE",
            current_archive_surface=True,
            parser_visible=True,
            raw_byte_mutation_parse_safe=True,
            whole_section_replacement_surface=True,
            requires_archive_adapter=False,
            original_payload_bytes=600 * 16 * 2,
            seed_bytes=DEFAULT_SEED_BYTES,
            evidence_grade="source_formula_default_bytes",
            byte_basis="ATW2 num_pairs * scorer_class_prior_dim * fp16_bytes = 600*16*2",
            blocker="large byte upside but high signal-risk; probe after cdf_table_blob",
            rationale=(
                "The class prior table is parser-visible and large, but likely carries "
                "score-relevant conditioning. It should be probed only with explicit "
                "signal-preservation gates."
            ),
        ),
        ProceduralReplacementSurface(
            substrate_id="grayscale_lut_glv1",
            surface_id="chroma_lut",
            context="chroma_lut_replacement",
            candidate_status="BLOCKED_NO_CURRENT_SURFACE",
            current_archive_surface=False,
            parser_visible=False,
            raw_byte_mutation_parse_safe=False,
            whole_section_replacement_surface=False,
            requires_archive_adapter=True,
            original_payload_bytes=256,
            seed_bytes=DEFAULT_SEED_BYTES,
            evidence_grade="design_target_not_current_glv1",
            byte_basis="future GLV2 explicit chroma LUT target; current GLV1 has no such section",
            blocker="GLV2 explicit LUT grammar required before any byte-savings claim",
            rationale=(
                "The grayscale-LUT procedural scaffold is an envelope today. It is "
                "not yet a parser-visible replacement surface in current GLV1."
            ),
        ),
        ProceduralReplacementSurface(
            substrate_id="pr101_fec6_frontier",
            surface_id="master_gradient_null_bytes",
            context="master_gradient_null_byte_replacement_with_arbitrary_constant",
            candidate_status="BLOCKED_PARSER_SAFE_SUBSET_EMPTY",
            current_archive_surface=True,
            parser_visible=False,
            raw_byte_mutation_parse_safe=False,
            whole_section_replacement_surface=False,
            requires_archive_adapter=True,
            original_payload_bytes=16_292,
            seed_bytes=DEFAULT_SEED_BYTES,
            evidence_grade="macos_cpu_advisory_parser_safe_subset_smoke",
            byte_basis="parser-safe subset smoke: 0/16292 null bytes parser-safe",
            blocker="all null-gradient bytes are parser-essential in FP11/Brotli/LZMA/Huffman grammar",
            rationale=(
                "Negative control from the FEC6 smoke. Null-gradient alone does not "
                "make a byte replaceable."
            ),
        ),
    )


def rank_surface_matrix(
    surfaces: tuple[ProceduralReplacementSurface, ...] | None = None,
) -> list[ProceduralReplacementSurface]:
    """Rank surfaces by actionable status, then predicted rate-axis upside."""

    rows = list(surfaces if surfaces is not None else build_default_surface_matrix())
    return sorted(
        rows,
        key=lambda s: (
            not s.actionable_now,
            s.candidate_status != "READY_TO_PAIR_SMOKE",
            -s.predicted_bytes_saved,
            s.substrate_id,
            s.surface_id,
        ),
    )


def build_surface_matrix_payload() -> dict[str, Any]:
    """Return a JSON-serializable advisory payload."""

    rows = [surface.as_dict() for surface in rank_surface_matrix()]
    return {
        "schema_version": "procedural_replacement_surface_matrix_v1_20260521",
        "axis_tag": "[predicted]",
        "evidence_grade": "source_grounded_static_planner",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "canonical_rate_denom_bytes": CANONICAL_RATE_DENOM_BYTES,
        "canonical_rate_multiplier": CANONICAL_RATE_MULTIPLIER,
        "seed_bytes_default": DEFAULT_SEED_BYTES,
        "surfaces": rows,
    }


__all__ = [
    "DEFAULT_SEED_BYTES",
    "ProceduralReplacementSurface",
    "build_default_surface_matrix",
    "build_surface_matrix_payload",
    "rank_surface_matrix",
]
