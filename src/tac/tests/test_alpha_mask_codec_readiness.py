from __future__ import annotations

import json

import pytest

from tac.alpha_mask_codec_readiness import (
    ALPHA_MASK_CODEC_FAMILY_CONTRACTS,
    SUPPORTED_ALPHA_MASK_CODEC_FAMILIES,
    build_alpha_mask_training_readiness_contract,
)


def test_alpha_mask_readiness_defaults_fail_closed_for_all_families() -> None:
    for family in SUPPORTED_ALPHA_MASK_CODEC_FAMILIES:
        manifest = build_alpha_mask_training_readiness_contract(codec_family=family)

        assert manifest["ready_for_training_harness"] is False
        assert manifest["ready_for_exact_eval_dispatch"] is False
        assert manifest["score_claim"] is False
        assert manifest["bytes_charged"] is None
        assert "decode_validation_missing" in manifest["dispatch_blockers"]
        assert "charged_bytes_missing" in manifest["dispatch_blockers"]
        assert manifest["decoder_entrypoint"] == (
            ALPHA_MASK_CODEC_FAMILY_CONTRACTS[family].decoder_entrypoint
        )
        assert manifest["research_basis_ids"]
        assert manifest["scorer_term_targeted"] == "seg"
        json.dumps(manifest, sort_keys=True)


def test_alpha_mask_readiness_passes_only_with_complete_decode_validation() -> None:
    validation = {
        "status": "passed",
        "decoder_entrypoint": "tac.wavelet_mask_codec.decode_wavelet_codec",
        "decoded_shape": [1200, 384, 512],
        "decoded_dtype": "torch.uint8",
        "source_mask_sha256": "0" * 64,
        "decoded_mask_sha256": "1" * 64,
        "shape_matches_expected": True,
        "class_range_valid": True,
        "sidecars_required": False,
    }

    manifest = build_alpha_mask_training_readiness_contract(
        codec_family="wavelet",
        bytes_charged=12345,
        decode_validation=validation,
        source_prior="wavelet_foveation_prior",
        evidence_grade="empirical",
    )

    assert manifest["ready_for_training_harness"] is True
    assert manifest["ready_for_exact_eval_dispatch"] is False
    assert manifest["bytes_charged"] == 12345
    assert manifest["source_prior"] == "wavelet_foveation_prior"
    assert "foveated_telepresence_2025" in manifest["research_basis_ids"]
    assert "exact_cuda_auth_eval_missing" in manifest["dispatch_blockers"]
    assert "decode_validation_missing" not in manifest["dispatch_blockers"]


def test_alpha_mask_readiness_rejects_decoder_mismatch() -> None:
    validation = {
        "status": "passed",
        "decoder_entrypoint": "wrong.decoder",
        "decoded_shape": [1200, 384, 512],
        "decoded_dtype": "torch.uint8",
        "source_mask_sha256": "0" * 64,
        "decoded_mask_sha256": "1" * 64,
        "shape_matches_expected": True,
        "class_range_valid": True,
        "sidecars_required": False,
    }

    manifest = build_alpha_mask_training_readiness_contract(
        codec_family="grayscale_lut",
        bytes_charged=1200,
        decode_validation=validation,
    )

    assert manifest["ready_for_training_harness"] is False
    assert "decoder_entrypoint_mismatch" in manifest["dispatch_blockers"]


def test_alpha_mask_readiness_rejects_unknown_family() -> None:
    with pytest.raises(ValueError, match="unsupported alpha mask codec family"):
        build_alpha_mask_training_readiness_contract(codec_family="cool_name_only")
