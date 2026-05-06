"""Canonical contract for categorical mask-compression research surfaces."""

from __future__ import annotations

from typing import Any

from tac.semantic_label_contract import (
    CONTEST_SEGNET_CLASSES,
    NUM_CONTEST_SEGNET_CLASSES,
    SELFCOMP_CLASS_TO_GRAY,
    SEMANTIC_QUANTIZATION_DEFAULT_BITS,
    validate_contest_class_table,
)

SCHEMA_VERSION = 1


def build_categorical_compression_contract() -> dict[str, Any]:
    """Return the non-dispatch contract for QMA9/CLaDE/SPADE/openpilot work."""

    validate_contest_class_table()
    classes = [
        {
            "class_id": item.class_id,
            "comma10k_id": item.comma10k_id,
            "name": item.name,
            "comma10k_color": item.comma10k_color,
            "selfcomp_gray": SELFCOMP_CLASS_TO_GRAY[item.class_id],
            "default_quant_bits": SEMANTIC_QUANTIZATION_DEFAULT_BITS[item.class_id],
        }
        for item in CONTEST_SEGNET_CLASSES
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "categorical_compression_contract",
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "promotion_eligible": False,
        "class_count": NUM_CONTEST_SEGNET_CLASSES,
        "classes": classes,
        "conditioning_families": {
            "qma9": {
                "role": "charged semantic mask stream or replacement mask grammar",
                "class_id_contract": "contest_zero_based_comma10k_order",
                "decoder_required": True,
                "charged_bytes_required": True,
            },
            "clade_spade": {
                "role": "class-conditioned renderer/residual modulation",
                "class_id_contract": "contest_zero_based_comma10k_order",
                "parameters_must_be_charged": True,
                "no_untracked_label_remap": True,
            },
            "openpilot_priors": {
                "role": "proposal ranking or charged side information",
                "allowed_uncharged_use": "compression_time_atom_ranking_only",
                "inflate_runtime_use_requires_charged_payload": True,
            },
        },
        "charged_byte_contract": {
            "every_decoder_table_is_archive_member": True,
            "every_label_remap_is_archive_member": True,
            "every_conditioning_weight_or_codebook_is_archive_member": True,
            "sidecars_outside_archive_forbidden": True,
        },
        "no_op_controls": (
            "decode_reencode_identity_control",
            "label_permutation_fail_closed_control",
            "charged_member_presence_control",
            "runtime_consumes_conditioning_control",
        ),
        "stacking_contract": {
            "representation": "semantic class ids or class-conditioned residual atoms",
            "prediction": "openpilot/camera priors may rank atoms but do not score",
            "quantization": "per-class quantization must use canonical class ids",
            "hyperprior": "learned class/context tables must be charged",
            "arithmetic": "entropy stream must declare frequency/codebook custody",
            "pack": "archive member manifest must name every categorical payload",
        },
        "dispatch_blockers": (
            "planning_contract_not_archive_candidate",
            "requires_byte_closed_decoder_or_runtime_consumer",
            "requires_no_op_controls_before_exact_eval",
            "requires_exact_cuda_auth_eval_before_score_claim",
        ),
    }


__all__ = ["SCHEMA_VERSION", "build_categorical_compression_contract"]
