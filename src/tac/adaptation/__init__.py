# SPDX-License-Identifier: MIT
"""Dynamic per-video adaptation contracts."""

from tac.adaptation.dynamic_byte_allocator import (
    DynamicByteAllocatorError,
    build_dynamic_byte_atom_ledger,
    select_hard_pairs,
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
    "TelemetryPairRow",
    "build_dynamic_byte_atom_ledger",
    "build_dynamic_video_telemetry",
    "select_hard_pairs",
    "telemetry_to_hard_pair_indices",
    "write_hard_pair_indices_file",
]
