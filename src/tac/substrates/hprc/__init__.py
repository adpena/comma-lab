# SPDX-License-Identifier: MIT
"""HPRC: Hierarchical Predictive Receiver Codec.

HPRC is the rate-first substrate lane that turns PR95/HNeRV control lessons,
Z8/HPC teacher surfaces, learned residual tokenization, and scorer-conditioned
allocation into one byte-closed archive contract.
"""

from __future__ import annotations

from tac.substrates.hprc.archive import (
    HPRC_MAGIC,
    HPRC_SCHEMA_VERSION,
    HprcPacket,
    HprcPacketConfig,
    HprcSection,
    HprcSectionKind,
    pack_hprc_packet,
    parse_hprc_packet,
)
from tac.substrates.hprc.archive_candidate import (
    HPRC_ARCHIVE_BOUND_ADAPTER_ID,
    HPRC_ARCHIVE_CANDIDATE_FAMILY,
    HPRC_ARCHIVE_TRANSFORM_KIND,
    build_hprc_section_mutation_proof,
    build_minimal_hprc_v0_packet,
    export_hprc_archive_bytes,
)
from tac.substrates.hprc.lineage import (
    HPRC_FAMILY_BINDINGS,
    HPRC_OPTIMIZATION_LEVERS,
    HprcFamilyBinding,
    HprcOptimizationLever,
    HprcRole,
    get_binding,
    hprc_campaign_manifest,
    primary_rate_collapse_candidates,
    residual_sidecar_candidates,
)
from tac.substrates.hprc.pr95_adapter import (
    PR95_HNERV_DECODER_FAMILY_ID,
    PR95_RGB_COLOR_TRANSFORM_ID,
    Pr95HprcControlPacket,
    build_pr95_hprc_control_packet,
    parse_pr95_hnerv_payload,
)

__all__ = [
    "HPRC_ARCHIVE_BOUND_ADAPTER_ID",
    "HPRC_ARCHIVE_CANDIDATE_FAMILY",
    "HPRC_ARCHIVE_TRANSFORM_KIND",
    "HPRC_FAMILY_BINDINGS",
    "HPRC_MAGIC",
    "HPRC_OPTIMIZATION_LEVERS",
    "HPRC_SCHEMA_VERSION",
    "PR95_HNERV_DECODER_FAMILY_ID",
    "PR95_RGB_COLOR_TRANSFORM_ID",
    "HprcFamilyBinding",
    "HprcOptimizationLever",
    "HprcPacket",
    "HprcPacketConfig",
    "HprcRole",
    "HprcSection",
    "HprcSectionKind",
    "Pr95HprcControlPacket",
    "build_hprc_section_mutation_proof",
    "build_minimal_hprc_v0_packet",
    "build_pr95_hprc_control_packet",
    "export_hprc_archive_bytes",
    "get_binding",
    "hprc_campaign_manifest",
    "pack_hprc_packet",
    "parse_hprc_packet",
    "parse_pr95_hnerv_payload",
    "primary_rate_collapse_candidates",
    "residual_sidecar_candidates",
]
