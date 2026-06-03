# SPDX-License-Identifier: MIT
"""Dynamic per-video adaptation contracts."""

from tac.adaptation.dynamic_byte_allocator import (
    DynamicByteAllocatorError,
    build_dynamic_byte_atom_ledger,
    select_hard_pairs,
)
from tac.adaptation.hard_pair_indices import (
    HardPairIndicesError,
    load_pair_indices_file,
    merge_pair_indices,
    normalize_pair_indices,
    pair_indices_from_mapping,
    parse_pair_indices_csv,
    validate_pair_indices_in_range,
)
from tac.adaptation.video_telemetry import (
    DynamicVideoTelemetryError,
    TelemetryPairRow,
    build_dynamic_video_telemetry,
    telemetry_to_hard_pair_indices,
    write_hard_pair_indices_file,
)

__all__ = [
    "DynamicByteAllocatorError",
    "DynamicVideoTelemetryError",
    "HardPairIndicesError",
    "TelemetryPairRow",
    "build_dynamic_byte_atom_ledger",
    "build_dynamic_video_telemetry",
    "load_pair_indices_file",
    "merge_pair_indices",
    "normalize_pair_indices",
    "pair_indices_from_mapping",
    "parse_pair_indices_csv",
    "select_hard_pairs",
    "telemetry_to_hard_pair_indices",
    "validate_pair_indices_in_range",
    "write_hard_pair_indices_file",
]
