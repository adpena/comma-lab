# SPDX-License-Identifier: MIT
"""Canonical schema names for reusable materializer contracts."""

from __future__ import annotations

ARCHIVE_SECTION_ENTROPY_RECODE_SCHEMA = "archive_section_entropy_recode_candidate.v1"
ARCHIVE_ZIP_REPACK_SCHEMA = "archive_zip_repack_candidate.v1"
PACKET_MEMBER_RECOMPRESS_SCHEMA = "packet_member_recompress_candidate.v1"
PACKET_MEMBER_MERGE_SCHEMA = "packet_member_merge_candidate.v1"
PACKET_MEMBER_ZIP_HEADER_ELIDE_SCHEMA = "packet_member_zip_header_elide_candidate.v1"
RENDERER_PAYLOAD_DFL1_SCHEMA = "renderer_payload_dfl1_candidate.v1"
TENSOR_FACTORIZE_SCHEMA = "tensor_factorize_candidate.v1"
BYTE_RANGE_ENTROPY_RECODE_VERIFIED_SCHEMA = "byte_range_entropy_recode_verified_candidate.v1"

FAMILY_AGNOSTIC_MATERIALIZER_CANDIDATE_SCHEMAS = frozenset(
    {
        ARCHIVE_SECTION_ENTROPY_RECODE_SCHEMA,
        ARCHIVE_ZIP_REPACK_SCHEMA,
        PACKET_MEMBER_RECOMPRESS_SCHEMA,
        PACKET_MEMBER_MERGE_SCHEMA,
        PACKET_MEMBER_ZIP_HEADER_ELIDE_SCHEMA,
        RENDERER_PAYLOAD_DFL1_SCHEMA,
        TENSOR_FACTORIZE_SCHEMA,
    }
)

MATERIALIZER_RECEIVER_FEEDBACK_ARTIFACT_SCHEMAS = frozenset(
    {
        *FAMILY_AGNOSTIC_MATERIALIZER_CANDIDATE_SCHEMAS,
        BYTE_RANGE_ENTROPY_RECODE_VERIFIED_SCHEMA,
    }
)
