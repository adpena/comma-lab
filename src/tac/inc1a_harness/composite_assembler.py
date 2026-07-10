# SPDX-License-Identifier: MIT
"""composite_assembler — DELEGATION SHIM to the canonical :mod:`tac.through_r.scaffold_assembler`.

increment-1a item 1 (SYNTHESIS_v2_v8 §A.4 / §B ``measure_1a``). The composite-argmax assembler
was PROMOTED to :mod:`tac.through_r.scaffold_assembler` (CANONICALIZATION UNIT 1, #388) as the
campaign's canonical "compose per-class fields into ONE argmax partition" home. This module
now RE-EXPORTS that canonical surface verbatim so every inc1a call site + test keeps working
unchanged — the public API (``CarrierField`` / ``CompositeResult`` / ``assemble_fields`` /
``compose_partition`` / ``reconcile_partition`` / ``BC_MODES`` / ``N_SEG_CLASSES`` /
``Inc1aAssemblerError``) is identical, and ``Inc1aAssemblerError`` remains the SAME class object
(aliased to ``ScaffoldAssemblerError`` in the canonical module) so ``except`` / ``pytest.raises``
call sites are unaffected.

There is exactly ONE implementation now (in through_r); this shim carries no logic.
"""

from __future__ import annotations

from tac.through_r.scaffold_assembler import (
    BC_MODES,
    N_SEG_CLASSES,
    CarrierField,
    CompositeResult,
    Inc1aAssemblerError,
    ScaffoldAssemblerError,
    assemble_fields,
    compose_partition,
    reconcile_partition,
)

__all__ = [
    "BC_MODES",
    "N_SEG_CLASSES",
    "CarrierField",
    "CompositeResult",
    "Inc1aAssemblerError",
    "ScaffoldAssemblerError",
    "assemble_fields",
    "compose_partition",
    "reconcile_partition",
]
