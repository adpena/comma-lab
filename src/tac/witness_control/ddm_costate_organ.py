# SPDX-License-Identifier: MIT
"""Compatibility re-export for the live top-level DDM costate organ.

New live consumers must import :mod:`tac.ddm_costate_organ` directly so Python
does not initialize the historical witness-control package first.
"""

from tac.ddm_costate_organ import (
    CHECKPOINT_SCHEMA,
    EVIDENCE_AXIS,
    LEGACY_AUTHORITY_OWED_ROWS,
    MATURITY,
    SCHEMA,
    SOURCE_SPECS,
    DdmCostateCheckpoint,
    SourceSpec,
    build_live_ddm_costate,
    digest_lines,
    discover_sources,
    rank_scheduler_blocks,
    register_ddm_costate_checkpoint,
    write_receipt_atomic,
)

__all__ = [
    "CHECKPOINT_SCHEMA",
    "EVIDENCE_AXIS",
    "LEGACY_AUTHORITY_OWED_ROWS",
    "MATURITY",
    "SCHEMA",
    "SOURCE_SPECS",
    "DdmCostateCheckpoint",
    "SourceSpec",
    "build_live_ddm_costate",
    "digest_lines",
    "discover_sources",
    "rank_scheduler_blocks",
    "register_ddm_costate_checkpoint",
    "write_receipt_atomic",
]
