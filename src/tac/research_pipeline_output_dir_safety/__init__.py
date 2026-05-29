# SPDX-License-Identifier: MIT
"""Canonical helper for research-pipeline tools writing under ``.omx/research/``.

This module is the canonical 2-landing-pattern sister of anti-pattern
``research_pipeline_tool_re_writes_historical_provenance_json_with_mutated_fields_v1``
registered 2026-05-28T22:41 in
``.omx/state/canonical_anti_patterns_registry.jsonl``.

Empirical anchor: preflight 2026-05-28T22:41 raised 77 in-place field
mutations across 3 dirs:

* ``pr95_mlx_runtime_consumption_queue_20260528T131513Z`` (24 mutations)
* ``repair_multi_archive_autonomous_live_psv3_fec6_20260528T055303Z`` (50)
* ``frontier_final_rate_attack_fp11_brotli_exec3_20260528Tlocal`` (3)

Per CLAUDE.md "Frontier scores are pointer-only" non-negotiable +
"Operator gates must be wired and used" non-negotiable + canonical Catalog
#110/#113 APPEND-ONLY HISTORICAL_PROVENANCE discipline + the canonical
``recover_lane_artifacts.py::_write_report`` reference pattern at
``tools/recover_lane_artifacts.py``: research-pipeline tools writing
``.omx/research/**/*.json`` MUST EITHER:

1. Write to a NEW dated subdir per invocation (the canonical structural
   per-invocation partition); OR
2. Emit append-only ``attempts[]`` schema per Catalog #110 pattern; OR
3. Carry an explicit operator opt-in flag.

The canonical helper :func:`validate_research_pipeline_output_dir` is the
runtime per-call validator (sister of the per-source STRICT preflight gate
referenced in the registered anti-pattern); together they extinct the bug
class at BOTH the runtime per-call surface AND the per-source surface.

Public API
----------
* :class:`ResearchPipelineOutputDirVerdict` — frozen dataclass with the
  per-validation outcome.
* :class:`OutputDirSafetyError` — typed exception raised when validation
  fails fail-closed.
* :func:`validate_research_pipeline_output_dir` — the canonical 4-cascade
  validator producing a typed verdict.
* :func:`enforce_research_pipeline_output_dir` — convenience wrapper that
  raises :class:`OutputDirSafetyError` on non-PROCEED verdicts.

Cross-references
----------------
* CLAUDE.md "Artifact lifecycle compliance" Catalog #113.
* CLAUDE.md "Forbidden in-place edits to public PR intake clones" sister.
* Catalog #110 (``check_recovery_metadata_append_only`` reference pattern).
* Catalog #287 (placeholder-rationale rejection sister discipline).
* ``tools/recover_lane_artifacts.py::_write_report`` (canonical append-only
  reference).
"""

from __future__ import annotations

from .output_dir_safety import (
    DEFAULT_DATED_SUFFIX_PATTERNS,
    DEFAULT_WAIVER_PLACEHOLDERS,
    HISTORICAL_PROVENANCE_JSON_NAMES,
    OutputDirSafetyError,
    ResearchPipelineOutputDirVerdict,
    enforce_research_pipeline_output_dir,
    is_dated_subdir,
    validate_research_pipeline_output_dir,
)

__all__ = [
    "DEFAULT_DATED_SUFFIX_PATTERNS",
    "DEFAULT_WAIVER_PLACEHOLDERS",
    "HISTORICAL_PROVENANCE_JSON_NAMES",
    "OutputDirSafetyError",
    "ResearchPipelineOutputDirVerdict",
    "enforce_research_pipeline_output_dir",
    "is_dated_subdir",
    "validate_research_pipeline_output_dir",
]
