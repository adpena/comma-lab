"""Charged label-prior payload manifest for categorical candidates."""

from __future__ import annotations

from typing import Any

from tac.categorical_openpilot_mask_prior_contract import (
    RUNTIME_LABEL_CONTRACT,
    audit_categorical_openpilot_mask_priors,
)
from tac.semantic_label_contract import (
    CONTEST_SEGNET_CLASS_NAME_TUPLE,
    CONTEST_SEGNET_CLASSES,
    SELFCOMP_CLASS_TO_GRAY,
)

SCHEMA_VERSION = 1
LABEL_PRIOR_PAYLOAD_MANIFEST_KIND = "categorical_label_prior_payload_manifest"
LABEL_PRIOR_PAYLOAD_MANIFEST_CONTRACT = "categorical_label_prior_payload_manifest_v1"
LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER = "label_prior_payload_manifest.json"
LABEL_PRIOR_PAYLOAD_MANIFEST_ROLE = "label_prior_payload_manifest"


def _charged_name_list(charged_member_links: Any) -> list[str]:
    if not isinstance(charged_member_links, list):
        return []
    return [
        record.get("name", "")
        for record in charged_member_links
        if isinstance(record, dict) and isinstance(record.get("name"), str)
    ]


def canonical_categorical_label_prior_class_rows() -> list[dict[str, Any]]:
    """Return the byte-closed semantic rows required in label-prior manifests."""

    return [
        {
            "class_id": item.class_id,
            "comma10k_id": item.comma10k_id,
            "name": item.name,
            "selfcomp_gray": SELFCOMP_CLASS_TO_GRAY[item.class_id],
        }
        for item in CONTEST_SEGNET_CLASSES
    ]


def build_categorical_label_prior_payload_manifest(
    *,
    source_archive_sha256: str,
    payload_member: str,
    payload_member_sha256: str,
    class_codebook_member: str,
    class_codebook_sha256: str,
    conditioning_priors: Any,
) -> dict[str, Any]:
    """Build the deterministic charged label/openpilot prior manifest."""

    charged_member_links = [
        {
            "name": payload_member,
            "role": "categorical_payload",
            "sha256": payload_member_sha256,
        },
        {
            "name": class_codebook_member,
            "role": "decoder_table",
            "sha256": class_codebook_sha256,
        },
    ]
    normalized_priors = conditioning_priors if isinstance(conditioning_priors, list) else []
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": LABEL_PRIOR_PAYLOAD_MANIFEST_KIND,
        "label_prior_payload_manifest_contract": LABEL_PRIOR_PAYLOAD_MANIFEST_CONTRACT,
        "score_claim": False,
        "dispatch_attempted": False,
        "ready_for_exact_eval_dispatch": False,
        "source_archive_sha256": source_archive_sha256,
        "label_contract": RUNTIME_LABEL_CONTRACT,
        "semantic_class_order": list(CONTEST_SEGNET_CLASS_NAME_TUPLE),
        "selfcomp_gray_codebook": [
            SELFCOMP_CLASS_TO_GRAY[index] for index in range(len(SELFCOMP_CLASS_TO_GRAY))
        ],
        "class_rows": canonical_categorical_label_prior_class_rows(),
        "charged_member_links": charged_member_links,
        "conditioning_priors": normalized_priors,
        "conditioning_prior_contract": audit_categorical_openpilot_mask_priors(
            normalized_priors,
            charged_member_names=_charged_name_list(charged_member_links),
            charged_members=charged_member_links,
        ),
        "required_no_op_controls": [
            "decode_reencode_identity_control",
            "label_permutation_fail_closed_control",
            "charged_member_presence_control",
            "runtime_consumes_conditioning_control",
        ],
    }


__all__ = [
    "LABEL_PRIOR_PAYLOAD_MANIFEST_CONTRACT",
    "LABEL_PRIOR_PAYLOAD_MANIFEST_KIND",
    "LABEL_PRIOR_PAYLOAD_MANIFEST_MEMBER",
    "LABEL_PRIOR_PAYLOAD_MANIFEST_ROLE",
    "RUNTIME_LABEL_CONTRACT",
    "SCHEMA_VERSION",
    "build_categorical_label_prior_payload_manifest",
    "canonical_categorical_label_prior_class_rows",
]
