# SPDX-License-Identifier: MIT
"""META-LIFT-4 UNIWARD canonical-application-surface invariant enumerator.

Per operator NON-NEGOTIABLE directive 2026-05-26 + just-validated 7th-order
PARADIGM-VALIDATED-AT-ENTROPY-CODED-SIDECAR landing (commit ``87bd1c355``;
43/72 nonempty bins; 576x dynamic range): UNIWARD has structural traction
at the canonical **ENTROPY-CODED + QUANTIZED + PER-SYMBOL-ROUTABLE** surface
(per Holub-Fridrich-Denemark 2014 + Sallee 2003 + Fridrich 2007 canonical).
The 7th-order verdict establishes the canonical Fridrich-natural application
domain invariant: every quantized + entropy-coded codebook substrate is a
potential UNIWARD-applicable surface.

This module is the META-LIFT (per the 11th standing directive ORDER
discipline: ONE canonical enumerator ACROSS substrates FIRST, then
per-substrate consumption SECOND). It iterates ALL known
canonical-application surfaces in our codebase (DCT analog / chroma LUT /
scorer class softmax / master-gradient per-byte / FEC selector indices /
VQ-VAE indices_blob / Wyner-Ziv codec layer) and emits a typed enumeration
ranked by predicted ΔS per the Cauchy-Schwarz bound per canonical equation
``cross_substrate_master_gradient_aggregate_ranking_taylor_savings_v1`` +
``per_pair_master_gradient_score_impact_taylor_v1``.

Mathematical grounding (per canonical equation #344 family + UNIWARD
canonical references):

  Per-surface UNIWARD applicability invariant (Holub-Fridrich-Denemark 2014
  + Fridrich 2007 + Sallee 2003 canonical):

    A surface S is UNIWARD-applicable iff ALL of:
        1. ENTROPY_CODED: surface bytes participate in entropy coding
           (arithmetic / range / Huffman / ANS / brotli with structural
           per-symbol routability)
        2. QUANTIZED: surface bytes are quantized integer symbols
           (NOT raw floats; NOT lossless dense streams)
        3. PER_SYMBOL_ROUTABLE: per-symbol bit allocation has direct
           control over per-symbol distortion (the canonical Fridrich
           STC routing condition)
        4. CANONICAL_FORMULA_GROUNDED: per-symbol distortion formula
           cites Holub-Fridrich-Denemark 2014 OR sister canonical
           (Sallee 2003 weighted-median; Fridrich 2007 inverse-Fisher)

  Per-surface ranking metric (canonical equation
  ``per_pair_master_gradient_score_impact_taylor_v1``):

    leverage_S = ||∇S_axis||_2 / sqrt(N_symbols_S) (per-symbol leverage)
    upper_bound_S = ||∇S_axis||_2 · ||Δθ_max||_2   (Cauchy-Schwarz at unit)

  Cross-surface ranking ordered by canonical Cauchy-Schwarz bound per
  contest axis (CPU vs CUDA per CLAUDE.md "Submission auth eval — BOTH
  CPU AND CUDA" non-negotiable + Catalog #127 custody validator).

Per CLAUDE.md "Apples-to-apples evidence discipline" + Catalog #341
non-promotable routing markers: every enumeration output is OBSERVABILITY-
ONLY by construction.

  - ``axis_tag = "[predicted]"``
  - ``score_claim = False``
  - ``promotable = False``
  - ``evidence_grade = "[predicted; uniward-canonical-application-surface-enumeration]"``

Promotion of an enumeration row to a contest score signal REQUIRES
paired-CUDA empirical anchor on the specific surface per CLAUDE.md
"Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
HARDWARE" non-negotiable.

Architecture (Catalog #230 sister-disjoint):

  - Inputs: per-substrate canonical-application-surface descriptors
    (compiled from substrate architecture / archive grammar / inflate
    runtime introspection); per-substrate master-gradient anchors loaded
    via :func:`tac.master_gradient_consumers.load_aggregate_gradient_from_anchor`
    when available for per-surface ranking
  - Outputs: ``UniwardInvariantEnumeration`` frozen dataclass persisted to
    fcntl-locked JSONL at
    ``.omx/state/uniward_invariant_enumerations.jsonl``
  - Discipline: Catalog #131 / #138 / #245 (fcntl-locked + strict-load +
    canonical 4-layer ledger) + #287 / #323 (placeholder rejection +
    canonical Provenance) + #341 (routing markers) + #356 (per-axis)

Sister of:

  - :mod:`tac.cross_substrate_master_gradient_analyzer` (META-LIFT-1
    `60acdc2d2`) — cross-substrate master-gradient ranking at the
    per-substrate aggregate-tensor surface; META-LIFT-4 is the META-sister
    at the per-canonical-application-surface invariant axis
  - :mod:`tac.pareto_polytope_unified_solver` (META-LIFT-2 `da803dd30`)
    — cross-substrate Dykstra alternating-projections at the Pareto
    polytope surface; META-LIFT-4 SHARES the per-axis Taylor projection
    primitive for ranking
  - :mod:`tac.uniward_texture` (Holub-Fridrich-Denemark 2014 canonical)
  - :mod:`tac.uniward_delta` (Yousfi 2017 detector-informed embedding)
  - :mod:`tac.substrates.uniward_per_pixel_distortion.nscs06_v8_chroma_lut_integration`
    (7th-order PARADIGM-VALIDATED-AT-ENTROPY-CODED-SIDECAR; the empirical
    anchor that motivated this META-LIFT)
  - :mod:`tac.master_gradient_consumers` — per-substrate consumer surface
    (READ-ONLY here; ``load_aggregate_gradient_from_anchor`` consumed when
    available)

The 6-hook wire-in declaration per Catalog #125:

  * Hook #1 SENSITIVITY_MAP — ACTIVE (per-surface per-axis Taylor
    projections feed :mod:`tac.sensitivity_map` axis_weights downstream)
  * Hook #2 PARETO_CONSTRAINT — N/A (canonical Pareto polytope lives in
    META-LIFT-2; this enumerator surfaces the canonical-application-surface
    invariant for downstream consumption)
  * Hook #3 BIT_ALLOCATOR — ACTIVE (per-surface UNIWARD applicability +
    ranked Cauchy-Schwarz bound feed the bit allocator priority cascade
    per CLAUDE.md "Bit-level deconstruction and entropy discipline")
  * Hook #4 CATHEDRAL_AUTOPILOT_DISPATCH — ACTIVE (sister consumer auto-
    discovered per Catalog #335 / #336 / #337 canonical contract)
  * Hook #5 CONTINUAL_LEARNING_POSTERIOR — ACTIVE (per-enumeration
    canonical posterior anchor via ``append_enumeration_locked``; sister of
    ``master_gradient.append_anchor_locked`` + ``cross_substrate_master_gradient_analyzer.append_analysis_locked``)
  * Hook #6 PROBE_DISAMBIGUATOR — ACTIVE (per-surface UNIWARD-applicable
    verdict IS the canonical disambiguator between Fridrich-natural surfaces
    vs raw-RGB application-domain mismatches; the 5th + 6th-order PARADIGM-
    NULL → 7th-order PARADIGM-VALIDATED transitions empirically validate
    this disambiguator)

Per CLAUDE.md "MLX-first numpy-portable individually-fractal standing
directive": pure-numpy at the enumeration time (no MLX or PyTorch
dependency); per-substrate descriptors are static metadata; per-substrate
ranking is loaded from cached gradient sidecars.

Per CLAUDE.md "Canonical equations + models registry" + Catalog #344:
this module enables (but does NOT yet register; FORMALIZATION_PENDING
until paired-CUDA empirical anchor lands) canonical equation
``uniward_canonical_application_surface_invariant_enumeration_v1``.
"""
from __future__ import annotations

from tac.uniward_invariant_enumerator.enumerator import (
    CANONICAL_EQUATION_ID,
    PREDICTED_AXIS_TAG,
    SCHEMA_VERSION,
    UNIWARD_INVARIANT_ENUMERATIONS_LEDGER_PATH,
    VALID_AXIS_LABELS,
    VALID_INVARIANT_VERDICTS,
    VALID_SURFACE_KINDS,
    RankedUniwardSurfaces,
    UniwardApplicabilityVerdict,
    UniwardCanonicalApplicationSurface,
    UniwardInvariantEnumeration,
    UniwardInvariantEnumerationCorruptError,
    append_enumeration_locked,
    enumerate_uniward_canonical_application_surfaces,
    load_enumerations_strict,
    rank_uniward_applicable_surfaces_by_predicted_delta_s,
    verify_uniward_applicability,
)

__all__ = [
    "CANONICAL_EQUATION_ID",
    "PREDICTED_AXIS_TAG",
    "SCHEMA_VERSION",
    "UNIWARD_INVARIANT_ENUMERATIONS_LEDGER_PATH",
    "VALID_AXIS_LABELS",
    "VALID_INVARIANT_VERDICTS",
    "VALID_SURFACE_KINDS",
    "RankedUniwardSurfaces",
    "UniwardApplicabilityVerdict",
    "UniwardCanonicalApplicationSurface",
    "UniwardInvariantEnumeration",
    "UniwardInvariantEnumerationCorruptError",
    "append_enumeration_locked",
    "enumerate_uniward_canonical_application_surfaces",
    "load_enumerations_strict",
    "rank_uniward_applicable_surfaces_by_predicted_delta_s",
    "verify_uniward_applicability",
]
