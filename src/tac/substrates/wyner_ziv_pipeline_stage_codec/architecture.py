# SPDX-License-Identifier: MIT
"""Wyner-Ziv pipeline-stage codec architecture (L0 SCAFFOLD).

MLX-first per CLAUDE.md 8th MLX-FIRST standing directive 2026-05-26 +
INDIVIDUALLY-FRACTAL per 11th standing directive 2026-05-27. The architecture
is intentionally THIN at L0: this substrate's distinguishing-feature is the
pipeline-stage primitive insertion (routes through
``tac.codec.wyner_ziv_layer``), NOT a novel neural backbone. The L1 trainer
will wire MLX training + emit archive bytes; the L0 scaffold declares the
canonical interfaces + the substrate-distinguishing primitives per
Catalog #272 distinguishing-feature integration contract.

Per Catalog #220 substrate L1+ scaffold operational mechanism: at L0 the
mechanism is RESEARCH_ONLY (declared but not yet wired). The L1 trainer will
flip to OPERATIONAL by routing real pre-entropy bytes through the primitive
and producing archive bytes that inflate.py operationally consumes.

Public interface
================

* :class:`WynerZivPipelineStageCodecArchitecture` — frozen-config wrapper
  around the canonical primitive. Holds (intercept_location, side_info_source,
  side_info_max_bytes, main_codec, compression_codec_for_side). NOT a
  neural network at L0 (the neural component lands at L1 when this substrate
  wraps an existing base substrate's pre-entropy stage).
* :func:`encode_pre_entropy_via_pipeline_stage_codec` — encoder entry point.
  Takes raw pre-entropy bytes + Y, returns (main_compressed, side_compressed_baked,
  result). Routes through ``tac.codec.wyner_ziv_layer.insert_wyner_ziv_layer``.
* :func:`reconstruct_pre_entropy_via_pipeline_stage_codec` — decoder entry
  point. Takes (main, side, Y, config), returns reconstructed pre_entropy_bytes.
  Routes through ``tac.codec.wyner_ziv_layer.reconstruct_from_wyner_ziv_layer``.
* :func:`report_stage_byte_counts` — per-stage byte-count reporter for the
  observability surface (Catalog #305 ``inspectable_per_layer`` facet).
"""

from __future__ import annotations

import dataclasses
from typing import Any

from tac.codec.wyner_ziv_layer import (
    InterceptLocation,
    WynerZivLayerConfig,
    WynerZivLayerResult,
    insert_wyner_ziv_layer,
    reconstruct_from_wyner_ziv_layer,
)


__all__ = (
    "WynerZivPipelineStageCodecArchitecture",
    "encode_pre_entropy_via_pipeline_stage_codec",
    "reconstruct_pre_entropy_via_pipeline_stage_codec",
    "report_stage_byte_counts",
    "DEFAULT_INTERCEPT_LOCATION",
    "DEFAULT_SIDE_INFO_SOURCE",
    "DEFAULT_SIDE_INFO_MAX_BYTES",
)


# Canonical defaults for the L0 SCAFFOLD. The substrate-distinguishing
# decision per Catalog #272: this substrate operates at the
# STATE_DICT_SERIALIZATION intercept (the highest-leverage point per the
# sister primitive's empirical anchors) with Comma2k19 side-info source
# (decoder-side Y derived via the canonical Comma2k19LocalCache per Catalog
# #213). Per Wyner 1976 R(D|Y), encoder-decoder MUST agree on Y derivation.
DEFAULT_INTERCEPT_LOCATION: InterceptLocation = InterceptLocation.STATE_DICT_SERIALIZATION
DEFAULT_SIDE_INFO_SOURCE: str = "Comma2k19"
DEFAULT_SIDE_INFO_MAX_BYTES: int = 4096  # 4 KB inflate.py-baked side stream budget


@dataclasses.dataclass(frozen=True)
class WynerZivPipelineStageCodecArchitecture:
    """Frozen-config wrapper around the canonical Wyner-Ziv pipeline-stage primitive.

    Per Catalog #241 ``SubstrateContract`` invariant (parser_section_manifest
    declared in ``__init__.py``): the architecture exposes the canonical
    primitive's interface as a substrate-scope wrapper. At L0 the wrapper is
    a thin pass-through; at L1+ the trainer adds substrate-specific routing
    (e.g. per-pair Y derivation from PoseNet output, per-frame composition
    with a base substrate's pre-entropy stage).

    Attributes:
        intercept_location: where in the wrapped substrate's pipeline this
            WZ stage is inserted (per :class:`InterceptLocation`).
        side_info_source: canonical Y-derivation source per the primitive's
            taxonomy ("Comma2k19" / "ImageNet" / "torch_defaults" /
            "math_constants"; "scorer_compressed" requires explicit operator
            attestation per Catalog #320).
        side_info_max_bytes: budget for the inflate.py-baked side stream
            (per HNeRV parity L4 inflate runtime closure).
        main_codec: codec for the main stream (post-WZ-split → archive).
        compression_codec_for_side: codec for the side stream
            (baked into inflate.py per HNeRV parity L4 ≤200 LOC).
        deterministic_seed: seed for split-deterministic encoding (per
            Catalog #305 ``diff_able_across_runs`` facet).
    """

    intercept_location: InterceptLocation = DEFAULT_INTERCEPT_LOCATION
    side_info_source: str = DEFAULT_SIDE_INFO_SOURCE
    side_info_max_bytes: int = DEFAULT_SIDE_INFO_MAX_BYTES
    main_codec: str = "lzma"
    compression_codec_for_side: str = "lzma"
    deterministic_seed: int = 0

    def to_primitive_config(self) -> WynerZivLayerConfig:
        """Return a frozen :class:`WynerZivLayerConfig` honoring this architecture's choices.

        The conversion is intentionally explicit so the substrate-scope
        wrapper and the canonical primitive share a single source of truth
        for the WZ insertion contract.
        """
        return WynerZivLayerConfig(
            intercept_location=self.intercept_location,
            side_info_source=self.side_info_source,
            side_info_max_bytes=self.side_info_max_bytes,
            main_codec=self.main_codec,
            compression_codec_for_side=self.compression_codec_for_side,
            deterministic_seed=self.deterministic_seed,
        )


def encode_pre_entropy_via_pipeline_stage_codec(
    *,
    pre_entropy_bytes: bytes,
    side_info_y: bytes,
    architecture: WynerZivPipelineStageCodecArchitecture | None = None,
) -> WynerZivLayerResult:
    """Encode pre-entropy bytes via the canonical Wyner-Ziv pipeline-stage primitive.

    This is the canonical entry point for the encoder path of this substrate.
    Routes through ``tac.codec.wyner_ziv_layer.insert_wyner_ziv_layer`` per
    Catalog #335 canonical contract auto-discovery + Catalog #336 invocation
    discipline.

    Per CLAUDE.md "Forbidden device-selection defaults" + CLAUDE.md
    "Apples-to-apples evidence discipline": the canonical primitive's
    result ships with ``evidence_grade="predicted"`` + ``score_claim=False``
    + ``promotion_eligible=False`` defaults. Promotion to a contest score
    claim requires paired contest-CUDA + contest-CPU auth-eval anchors per
    CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-
    COMPLIANT HARDWARE".

    Args:
        pre_entropy_bytes: the bytes the wrapped substrate's existing pipeline
            would have fed into its entropy coder.
        side_info_y: the side information Y available to the decoder at
            inflate time (must equal the side_info_y passed at encode time
            per the contest determinism requirement).
        architecture: optional :class:`WynerZivPipelineStageCodecArchitecture`
            instance; defaults to canonical L0 defaults.

    Returns:
        :class:`WynerZivLayerResult` with byte counts + score-savings estimate
        + LOC overhead + decoder complexity estimate + canonical Provenance.
    """
    arch = architecture if architecture is not None else WynerZivPipelineStageCodecArchitecture()
    config = arch.to_primitive_config()
    return insert_wyner_ziv_layer(
        pre_entropy_bytes=pre_entropy_bytes,
        side_info_y=side_info_y,
        config=config,
    )


def reconstruct_pre_entropy_via_pipeline_stage_codec(
    *,
    main_compressed: bytes,
    side_compressed_baked: bytes,
    side_info_y: bytes,
    architecture: WynerZivPipelineStageCodecArchitecture | None = None,
) -> bytes:
    """Reconstruct pre-entropy bytes from (main, side, Y) at decoder side.

    Routes through ``tac.codec.wyner_ziv_layer.reconstruct_from_wyner_ziv_layer``.
    The reconstruction MUST be byte-identical to the source pre_entropy_bytes
    per the contest determinism requirement; the regression test
    ``tests/test_wyner_ziv_pipeline_stage_codec_smoke.py::
    test_encode_decode_roundtrip_byte_identical`` pins this invariant.

    Args:
        main_compressed: compressed main stream from the archive.
        side_compressed_baked: compressed side stream baked into inflate.py.
        side_info_y: side info available at inflate time (must equal the
            side_info_y passed at encode time per the contest determinism
            requirement).
        architecture: optional :class:`WynerZivPipelineStageCodecArchitecture`
            instance; MUST equal the architecture used at encode time per the
            primitive's contract.

    Returns:
        The original pre_entropy_bytes (byte-identical to encoder input).
    """
    arch = architecture if architecture is not None else WynerZivPipelineStageCodecArchitecture()
    config = arch.to_primitive_config()
    return reconstruct_from_wyner_ziv_layer(
        main_compressed=main_compressed,
        side_compressed_baked=side_compressed_baked,
        side_info_y=side_info_y,
        config=config,
    )


def report_stage_byte_counts(result: WynerZivLayerResult) -> dict[str, Any]:
    """Per-stage byte-count report for the observability surface.

    Per Catalog #305 ``inspectable_per_layer`` + ``decomposable_per_signal``
    facets: every Wyner-Ziv stage exposes its byte-budget contribution so a
    reviewer can decompose total archive bytes into per-stage attributions.

    Args:
        result: :class:`WynerZivLayerResult` from
            :func:`encode_pre_entropy_via_pipeline_stage_codec`.

    Returns:
        Dict with per-stage byte counts + score-savings estimate + LOC
        overhead + decoder complexity estimate + canonical
        Provenance-eligible (commit-archive-side-info-sha256) tuple.
    """
    return {
        "intercept_location": result.intercept_location.value,
        "main_bytes_raw": result.main_bytes_raw,
        "main_bytes_compressed": result.main_bytes_compressed,
        "side_bytes_raw": result.side_bytes_raw,
        "side_bytes_compressed_baked": result.side_bytes_compressed_baked,
        "score_savings_estimate": result.score_savings_estimate,
        "inflate_py_loc_added": result.inflate_py_loc_added,
        "decoder_complexity_estimate_seconds": result.decoder_complexity_estimate_seconds,
        "main_bytes_sha256": result.main_bytes_sha256,
        "side_info_sha256": result.side_info_sha256,
        "evidence_grade": result.evidence_grade,
        "score_claim": result.score_claim,
        "promotion_eligible": result.promotion_eligible,
    }
