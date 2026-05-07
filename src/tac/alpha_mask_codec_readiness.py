"""Fail-closed readiness contracts for alpha mask codec training harnesses.

This module does not train NeRV, wavelet, VQ-VAE, or grayscale-LUT mask
codecs. It records the minimum deterministic decode-validation contract that
must exist before a harness run can be treated as a reusable alpha-mask
artifact. Exact CUDA auth eval remains the only score authority.

ROUNDTRIP_NOT_REQUIRED: this module is a contract REGISTRY (frozen dataclass
records describing per-family decode-validation requirements), not a codec
that quantizes/encodes/decodes data. It does not have a quantize() function
to roundtrip; the actual NeRV/wavelet/VQ-VAE/grayscale-LUT codec
implementations live in src/tac/{nerv,wavelet,vqvae,grayscale_lut}_mask_codec.py
and own their own roundtrip tests. The Check 46 filename glob (`*codec*.py`)
matches this readiness file by accident.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from tac.optimization.research_basis import research_basis_ids_for_family

SUPPORTED_ALPHA_MASK_CODEC_FAMILIES: tuple[str, ...] = (
    "nerv",
    "wavelet",
    "vqvae",
    "grayscale_lut",
)


@dataclass(frozen=True)
class AlphaMaskCodecFamilyContract:
    codec_family: str
    local_implementation_surface: str
    decoder_entrypoint: str
    mathematical_constraint: str
    scorer_term_targeted: str
    research_basis_ids: tuple[str, ...]
    fail_closed_criteria: tuple[str, ...]


ALPHA_MASK_CODEC_FAMILY_CONTRACTS: dict[str, AlphaMaskCodecFamilyContract] = {
    "nerv": AlphaMaskCodecFamilyContract(
        codec_family="nerv",
        local_implementation_surface="src/tac/nerv_mask_codec.py",
        decoder_entrypoint="tac.nerv_mask_codec.decode_nerv_codec",
        mathematical_constraint=(
            "decoded class tensor y must match declared shape/dtype/range and "
            "be compared against source mask x before any NeRV artifact is "
            "eligible for training-harness reuse"
        ),
        scorer_term_targeted="seg",
        research_basis_ids=tuple(research_basis_ids_for_family("alpha")),
        fail_closed_criteria=(
            "refuse_if_decode_validation_missing",
            "refuse_if_decoder_entrypoint_mismatch",
            "refuse_if_decoded_shape_or_dtype_missing",
            "refuse_if_class_range_not_validated",
            "refuse_if_source_or_decoded_sha256_missing",
            "refuse_if_charged_bytes_missing",
        ),
    ),
    "wavelet": AlphaMaskCodecFamilyContract(
        codec_family="wavelet",
        local_implementation_surface="src/tac/wavelet_mask_codec.py",
        decoder_entrypoint="tac.wavelet_mask_codec.decode_wavelet_codec",
        mathematical_constraint=(
            "multi-scale sub-band payload must inverse-decode to the declared "
            "mask tensor before any foveated/wavelet atom can affect alpha "
            "training readiness"
        ),
        scorer_term_targeted="seg",
        research_basis_ids=tuple(
            dict.fromkeys(
                [
                    *research_basis_ids_for_family("wavelet"),
                    *research_basis_ids_for_family("foveation"),
                ]
            )
        ),
        fail_closed_criteria=(
            "refuse_if_decode_validation_missing",
            "refuse_if_decoder_entrypoint_mismatch",
            "refuse_if_decoded_shape_or_dtype_missing",
            "refuse_if_class_range_not_validated",
            "refuse_if_source_or_decoded_sha256_missing",
            "refuse_if_charged_bytes_missing",
        ),
    ),
    "vqvae": AlphaMaskCodecFamilyContract(
        codec_family="vqvae",
        local_implementation_surface="src/tac/vqvae_mask_codec.py",
        decoder_entrypoint="tac.vqvae_mask_codec.decode_vqvae_codec",
        mathematical_constraint=(
            "codebook plus index stream must decode without sidecars to the "
            "declared mask tensor; side-informed quantization metadata is "
            "compress-time provenance only"
        ),
        scorer_term_targeted="seg",
        research_basis_ids=tuple(research_basis_ids_for_family("alpha")),
        fail_closed_criteria=(
            "refuse_if_decode_validation_missing",
            "refuse_if_decoder_entrypoint_mismatch",
            "refuse_if_decoded_shape_or_dtype_missing",
            "refuse_if_class_range_not_validated",
            "refuse_if_source_or_decoded_sha256_missing",
            "refuse_if_charged_bytes_missing",
        ),
    ),
    "grayscale_lut": AlphaMaskCodecFamilyContract(
        codec_family="grayscale_lut",
        local_implementation_surface="src/tac/mask_grayscale_lut.py",
        decoder_entrypoint="tac.mask_grayscale_lut.decode_grayscale_to_classes",
        mathematical_constraint=(
            "nearest-LUT grayscale projection must roundtrip class IDs with "
            "declared LUT provenance before the stream can stand in for masks"
        ),
        scorer_term_targeted="seg",
        research_basis_ids=tuple(research_basis_ids_for_family("categorical")),
        fail_closed_criteria=(
            "refuse_if_decode_validation_missing",
            "refuse_if_decoder_entrypoint_mismatch",
            "refuse_if_decoded_shape_or_dtype_missing",
            "refuse_if_class_range_not_validated",
            "refuse_if_source_or_decoded_sha256_missing",
            "refuse_if_charged_bytes_missing",
        ),
    ),
}


def _missing_or_false(mapping: Mapping[str, Any], key: str) -> bool:
    return key not in mapping or mapping[key] is False or mapping[key] is None


def _decode_validation_blockers(
    *,
    family: AlphaMaskCodecFamilyContract,
    bytes_charged: int | None,
    decode_validation: Mapping[str, Any] | None,
) -> list[str]:
    blockers: list[str] = []
    if bytes_charged is None or int(bytes_charged) < 0:
        blockers.append("charged_bytes_missing")
    if decode_validation is None:
        blockers.append("decode_validation_missing")
        return blockers

    if decode_validation.get("status") != "passed":
        blockers.append("decode_validation_status_not_passed")
    if decode_validation.get("decoder_entrypoint") != family.decoder_entrypoint:
        blockers.append("decoder_entrypoint_mismatch")
    for key in (
        "decoded_shape",
        "decoded_dtype",
        "source_mask_sha256",
        "decoded_mask_sha256",
    ):
        if _missing_or_false(decode_validation, key):
            blockers.append(f"{key}_missing")
    if decode_validation.get("shape_matches_expected") is not True:
        blockers.append("shape_matches_expected_not_true")
    if decode_validation.get("class_range_valid") is not True:
        blockers.append("class_range_valid_not_true")
    if decode_validation.get("sidecars_required") is True:
        blockers.append("sidecars_required")
    return blockers


def build_alpha_mask_training_readiness_contract(
    *,
    codec_family: str,
    bytes_charged: int | None = None,
    decode_validation: Mapping[str, Any] | None = None,
    source_prior: str = "local_contract",
    evidence_grade: str = "empirical",
) -> dict[str, Any]:
    """Return a deterministic readiness manifest for one alpha mask codec.

    A codec is ready for the deterministic training harness only when its
    decode validation has already passed, charged bytes are declared, class
    range and shape are checked, and both source/decoded mask hashes exist.
    The returned manifest always blocks exact-eval dispatch; this is a harness
    readiness contract, not a candidate archive or score claim.
    """
    if codec_family not in ALPHA_MASK_CODEC_FAMILY_CONTRACTS:
        raise ValueError(
            f"unsupported alpha mask codec family {codec_family!r}; expected "
            f"one of {SUPPORTED_ALPHA_MASK_CODEC_FAMILIES}"
        )
    family = ALPHA_MASK_CODEC_FAMILY_CONTRACTS[codec_family]
    blockers = _decode_validation_blockers(
        family=family,
        bytes_charged=bytes_charged,
        decode_validation=decode_validation,
    )
    training_ready = not blockers
    return {
        "schema_version": 1,
        "contract": "alpha_mask_training_harness_readiness_v1",
        "codec_family": codec_family,
        "source_prior": source_prior,
        "mathematical_constraint": family.mathematical_constraint,
        "scorer_term_targeted": family.scorer_term_targeted,
        "research_basis_ids": family.research_basis_ids,
        "bytes_charged": bytes_charged,
        "local_implementation_surface": family.local_implementation_surface,
        "decoder_entrypoint": family.decoder_entrypoint,
        "decode_validation": dict(decode_validation or {}),
        "evidence_grade": evidence_grade,
        "ready_for_training_harness": training_ready,
        "ready_for_exact_eval_dispatch": False,
        "score_claim": False,
        "promotion_eligible": False,
        "dispatch_blockers": tuple(
            dict.fromkeys(
                [
                    *blockers,
                    "exact_archive_rebuild_missing",
                    "exact_cuda_auth_eval_missing",
                    "operator_dispatch_claim_missing",
                ]
            )
        ),
        "fail_closed_criteria": family.fail_closed_criteria,
    }


__all__ = [
    "ALPHA_MASK_CODEC_FAMILY_CONTRACTS",
    "SUPPORTED_ALPHA_MASK_CODEC_FAMILIES",
    "AlphaMaskCodecFamilyContract",
    "build_alpha_mask_training_readiness_contract",
]
