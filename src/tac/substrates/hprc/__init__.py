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
from tac.substrates.hprc.campaign import (
    HPRC_CAMPAIGN_MANIFEST_SCHEMA,
    HPRC_EXACT_READINESS_REFUSAL_SCHEMA,
    HPRC_V0_EXACT_READINESS_BLOCKERS,
    HprcCampaignRunResult,
    build_hprc_campaign_manifest,
    build_hprc_exact_readiness_refusal,
    materialize_minimal_hprc_campaign,
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
from tac.substrates.hprc.resolution_contract import (
    CONTEST_FRAME_COUNT,
    CONTEST_PAIR_COUNT,
    HPRC_RESOLUTION_CONTRACT_SCHEMA,
    POSENET_PAIR_FRAME_COUNT,
    POSENET_YUV6_CHANNEL_COUNT,
    RGB_CHANNEL_COUNT,
    hprc_resolution_contract,
)
from tac.substrates.hprc.training_adapter import (
    HPRC_LONG_TRAINING_ARCHIVE_EXPORT_SCHEMA,
    HPRC_LONG_TRAINING_SUBSTRATE_ID,
    HprcCompactReceiverLongTrainingAdapter,
    HprcCompactReceiverTrainingModel,
    HprcGainBounds,
)

__all__ = [
    "CONTEST_FRAME_COUNT",
    "CONTEST_PAIR_COUNT",
    "HPRC_ARCHIVE_BOUND_ADAPTER_ID",
    "HPRC_ARCHIVE_CANDIDATE_FAMILY",
    "HPRC_ARCHIVE_TRANSFORM_KIND",
    "HPRC_CAMPAIGN_MANIFEST_SCHEMA",
    "HPRC_EXACT_READINESS_REFUSAL_SCHEMA",
    "HPRC_FAMILY_BINDINGS",
    "HPRC_LONG_TRAINING_ARCHIVE_EXPORT_SCHEMA",
    "HPRC_LONG_TRAINING_SUBSTRATE_ID",
    "HPRC_MAGIC",
    "HPRC_OPTIMIZATION_LEVERS",
    "HPRC_RESOLUTION_CONTRACT_SCHEMA",
    "HPRC_SCHEMA_VERSION",
    "HPRC_V0_EXACT_READINESS_BLOCKERS",
    "POSENET_PAIR_FRAME_COUNT",
    "POSENET_YUV6_CHANNEL_COUNT",
    "PR95_HNERV_DECODER_FAMILY_ID",
    "PR95_RGB_COLOR_TRANSFORM_ID",
    "RGB_CHANNEL_COUNT",
    "HprcCampaignRunResult",
    "HprcCompactReceiverLongTrainingAdapter",
    "HprcCompactReceiverTrainingModel",
    "HprcFamilyBinding",
    "HprcGainBounds",
    "HprcOptimizationLever",
    "HprcPacket",
    "HprcPacketConfig",
    "HprcRole",
    "HprcSection",
    "HprcSectionKind",
    "Pr95HprcControlPacket",
    "build_hprc_campaign_manifest",
    "build_hprc_exact_readiness_refusal",
    "build_hprc_section_mutation_proof",
    "build_minimal_hprc_v0_packet",
    "build_pr95_hprc_control_packet",
    "export_hprc_archive_bytes",
    "get_binding",
    "hprc_campaign_manifest",
    "hprc_resolution_contract",
    "materialize_minimal_hprc_campaign",
    "pack_hprc_packet",
    "parse_hprc_packet",
    "parse_pr95_hnerv_payload",
    "primary_rate_collapse_candidates",
    "residual_sidecar_candidates",
]
