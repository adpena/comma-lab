# SPDX-License-Identifier: MIT
"""Canonical MLX canonicalization audit helper package.

Per operator NON-NEGOTIABLE binding directive 2026-05-30 verbatim
*"we have a lot of MLX code we want to ensure it is canonicalized and no
duplicate code and compounding optimization and learning and coherent
codebase, remember our tinygrad primitives work that is underway perhaps
include that in the memo as well"* + *"all must be wired and integrated
and tested and individually fractally optimized for extreme synergy and
positive externalities"*.

Sister of :mod:`tac.local_acceleration.pr95_hnerv_mlx` (the canonical
PR95 HNeRV MLX core; 3012 LOC; 83 functions; consumed by 11 of 17
substrate MLX renderers) at the **audit + canonicalization-discipline
surface**. Where ``pr95_hnerv_mlx`` IS the canonical primitive surface,
THIS package enforces that future substrate MLX renderers ROUTE THROUGH
the canonical primitives instead of re-implementing them.

Per CLAUDE.md "UNIQUE-AND-COMPLETE-PER-METHOD operating mode" Catalog
#290 falling-rule list: per-substrate FORKS are accepted when canonical
adoption is empirically worse OR principled-mismatch. The audit helper
returns typed verdicts so future agents can route the falling-rule
decision per primitive per substrate at design-memo time.

Per CLAUDE.md "Forbidden premature KILL without research exhaustion":
classifications below are operator-routable migration paths NOT KILL
verdicts. Each detected duplicate-implementation finding carries
canonical reactivation criteria (consume canonical helper OR document
substrate-optimal FORK per Catalog #290).

Public API (narrow per CLAUDE.md "Beauty, simplicity, and developer
experience"):

  * :class:`MLXBearingFile` — per-file canonical record (path + LOC +
    primitive impl signatures)
  * :class:`MLXPrimitiveImpl` — per-primitive canonical record
    (function/class location + LOC + signature hash)
  * :class:`CanonicalDuplicationVerdict` — typed verdict
    (CANONICAL_ADOPTED / CANONICAL_EXTRACTION_RECOMMENDED /
    PRINCIPLED_FORK_HARD_EARNED / UNCLEAR_NEEDS_EMPIRICAL)
  * :class:`MigrationPlan` — typed migration plan per primitive
  * :func:`enumerate_mlx_bearing_files` — canonical enumerator
  * :func:`enumerate_mlx_primitive_implementations` — per-primitive scan
  * :func:`detect_canonical_duplication` — Catalog #290 falling-rule
    classifier
  * :func:`recommend_canonical_extraction` — operator-routable migration
    plan emitter

Cross-references:
  * Catalog #205 — sister inflate-time canonical helpers
  * Catalog #290 — UNIQUE-AND-COMPLETE-PER-METHOD falling-rule list
  * Catalog #335 — canonical cathedral consumer auto-discovery
  * Catalog #383 — STRICT preflight gate that this audit feeds
  * Catalog #371 — orphan-auto-trigger-stub sister discipline (this
    module has zero stubs; every helper has a working numpy reference
    path)
  * tac.framework_agnostic — sister at the training-time framework-
    selection surface
  * tac.local_acceleration.pr95_hnerv_mlx — THE canonical MLX core
  * tac.local_acceleration.tinygrad_bridge — sister at the tinygrad
    portability surface
"""
from __future__ import annotations

from tac.local_acceleration.mlx_canonical_audit.audit import (
    CANONICAL_PR95_HNERV_MLX_MODULE,
    CANONICAL_PRIMITIVE_EXTRACTORS,
    CanonicalDuplicationVerdict,
    DuplicationClassification,
    MigrationPlan,
    MLXBearingFile,
    MLXPrimitiveImpl,
    detect_canonical_duplication,
    enumerate_mlx_bearing_files,
    enumerate_mlx_primitive_implementations,
    recommend_canonical_extraction,
    summarize_audit_report,
)


__all__ = [
    "CANONICAL_PR95_HNERV_MLX_MODULE",
    "CANONICAL_PRIMITIVE_EXTRACTORS",
    "CanonicalDuplicationVerdict",
    "DuplicationClassification",
    "MigrationPlan",
    "MLXBearingFile",
    "MLXPrimitiveImpl",
    "detect_canonical_duplication",
    "enumerate_mlx_bearing_files",
    "enumerate_mlx_primitive_implementations",
    "recommend_canonical_extraction",
    "summarize_audit_report",
]
