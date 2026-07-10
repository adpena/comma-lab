# SPDX-License-Identifier: MIT
"""tac.verdicts — canonical measurement-row schema + verdict-emission helper.

CANONICALIZATION UNIT 2 (task #388, operator GO 2026-07-09 "Go on building all").

WHY this package exists (the confound it extincts): verdict / measurement JSONs
were hand-rolled per subagent, so the same hooks kept firing for the same missing
fields — no ``verdict_scope`` (the verdict-scope ladder leg), a silently-zero /
absent noise floor (design philosophy P2), missing review-status metadata (MEMORY
L81 — recovery-written verdicts are UNREVIEWED by construction), and missing
composition rows (P12, operator 2026-07-09). This package is the ONE emission
surface: a typed, validated :class:`~tac.verdicts.measurement_row.MeasurementRow`
plus :func:`~tac.verdicts.emit.emit_verdict`, which REFUSES a claim that omits a
required honesty field rather than letting it ship and trip a hook downstream.

means != ends: a verdict JSON is a MEANS. Only a byte-closed n600 exact row
< 0.19110 from ``upstream/evaluate.py`` (contest-CPU/CUDA, NEVER MPS) moves the
pointer. This package makes the honest recording of that row cheap and structural.

# TODO(#389): the emitted verdict JSONs are the consumption surface for the
# session_bus (cross-agent verdict fan-in) and are cross-checked against the
# through_r measurement authority. Those wire-ins land in #389; this package
# deliberately imports NEITHER src/tac/through_r/ NOR src/tac/session_bus/
# (sibling units building in parallel) to keep the schema leaf-clean.
"""
from __future__ import annotations

from tac.verdicts.emit import (
    Composition,
    InteractionSign,
    ScopeLevel,
    VerdictEmitError,
    VerdictScope,
    emit_verdict,
)
from tac.verdicts.measurement_row import (
    AxisTag,
    MeasurementRow,
    MeasurementRowError,
    Provenance,
    ReviewStatus,
)

__all__ = [
    "AxisTag",
    "Composition",
    "InteractionSign",
    "MeasurementRow",
    "MeasurementRowError",
    "Provenance",
    "ReviewStatus",
    "ScopeLevel",
    "VerdictEmitError",
    "VerdictScope",
    "emit_verdict",
]
