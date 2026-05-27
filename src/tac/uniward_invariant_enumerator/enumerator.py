# SPDX-License-Identifier: MIT
"""META-LIFT-4 UNIWARD canonical-application-surface invariant enumerator.

Per the 11th standing directive ORDER-MATTERS discipline: this module is the
ONE canonical UNIWARD-applicability enumerator across all substrate surfaces;
per-substrate consumption happens SECOND through the sister cathedral consumer
that auto-discovers this enumerator's output.

Mathematical contract (per Holub-Fridrich-Denemark 2014 + Sallee 2003 +
Fridrich 2007 canonical references + canonical equation #344 family):

  Per-surface UNIWARD applicability INVARIANT (the 4-condition canonical
  Fridrich application domain test):

    A surface S is UNIWARD-applicable iff ALL of:

      1. ENTROPY_CODED:
            S's bytes participate in entropy coding (arithmetic / range /
            Huffman / ANS / brotli) with per-symbol routability — surfaces
            where per-byte mutation produces per-symbol distortion at
            decode time.

      2. QUANTIZED:
            S's bytes are quantized INTEGER symbols (uint8 / int8 / int16);
            UNIWARD weights operate on per-symbol distortion which is
            ill-defined for raw-float surfaces (this is why the 5th + 6th
            order RGB tensor application FAILED per Carmack-dissent).

      3. PER_SYMBOL_ROUTABLE:
            Per-symbol bit allocation has direct control over per-symbol
            distortion (the canonical Fridrich STC routing condition; per
            Filler-Judas-Fridrich 2011 STC arithmetic coder canonical).

      4. CANONICAL_FORMULA_GROUNDED:
            Per-symbol distortion formula cites one of the canonical
            references (Holub-Fridrich-Denemark 2014 universal distortion;
            Sallee 2003 weighted-median; Fridrich 2007 inverse-Fisher;
            Filler-Fridrich 2011 STC).

  Per-surface ranking (canonical equation
  ``per_pair_master_gradient_score_impact_taylor_v1``):

    leverage_S_axis = ||∇S_axis||_2 / sqrt(N_symbols_S)
    upper_bound_S_axis = ||∇S_axis||_2 · ||Δθ_max||_2

  Cross-surface ranking ordered by ``upper_bound_S_axis`` DESC per axis
  per Catalog #356 per-axis decomposition discipline.

  Aggregate cross-surface Cauchy-Schwarz bound (canonical equation
  ``cross_substrate_master_gradient_aggregate_ranking_taylor_savings_v1``):

    |Σ_S ΔS_S| ≤ Σ_S ||∇S_S||_2 · ||Δθ_S||_2

  Sister of META-LIFT-1 (different surface granularity: META-LIFT-1
  ranks substrates; META-LIFT-4 ranks canonical-application-surfaces
  WITHIN each substrate).

All outputs are OBSERVABILITY-ONLY per Catalog #341 + CLAUDE.md
"Apples-to-apples evidence discipline":

  - ``axis_tag = "[predicted]"``
  - ``score_claim = False``
  - ``promotable = False``
  - ``evidence_grade = "[predicted; uniward-canonical-application-surface-enumeration]"``

Promotion to a contest score signal REQUIRES paired-CUDA empirical anchor
per CLAUDE.md "Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-
COMPLIANT HARDWARE" non-negotiable.

Architecture (Catalog #230 sister-disjoint):

  - Inputs:
      * Per-substrate canonical-application-surface descriptors (static
        metadata compiled from substrate architecture / archive grammar /
        inflate runtime introspection at module-import time)
      * Optional per-substrate master-gradient anchors via
        :func:`tac.master_gradient_consumers.load_aggregate_gradient_from_anchor`
        when available for per-surface Cauchy-Schwarz ranking
  - Outputs: ``UniwardInvariantEnumeration`` frozen dataclass persisted to
    fcntl-locked JSONL at
    ``.omx/state/uniward_invariant_enumerations.jsonl``
  - Discipline: Catalog #131 / #138 / #245 (fcntl-locked + strict-load +
    canonical 4-layer ledger) + #287 / #323 (placeholder rejection +
    canonical Provenance) + #341 (routing markers) + #356 (per-axis)
"""
from __future__ import annotations

import datetime
import fcntl
import json
import os
import socket
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# Module-level constants (canonical paths + schema versions)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[3]

UNIWARD_INVARIANT_ENUMERATIONS_LEDGER_PATH = (
    REPO_ROOT / ".omx" / "state" / "uniward_invariant_enumerations.jsonl"
)
"""Canonical fcntl-locked JSONL append-only ledger.

Sister of:
  - ``.omx/state/master_gradient_anchors.jsonl`` (Catalog #245 + #327)
  - ``.omx/state/cross_substrate_master_gradient_analyses.jsonl`` (META-LIFT-1)
  - ``.omx/state/pareto_polytope_solutions.jsonl`` (META-LIFT-2)

at the UNIWARD canonical-application-surface invariant sub-surface.
"""

_LEDGER_LOCK_PATH = (
    REPO_ROOT / ".omx" / "state" / ".uniward_invariant_enumerations.lock"
)

SCHEMA_VERSION = "uniward_invariant_enumeration_v1"

# Per Catalog #356 per-axis decomposition: the canonical axis labels.
VALID_AXIS_LABELS: frozenset[str] = frozenset({"seg", "pose", "rate"})

# Per Catalog #341 routing markers (Tier A observability-only).
PREDICTED_AXIS_TAG = "[predicted]"

# Per Catalog #287 placeholder rejection.
_PLACEHOLDER_RATIONALES: frozenset[str] = frozenset(
    {"<rationale>", "<reason>", "<rationale_here>", "<reason_here>", ""}
)
_MIN_RATIONALE_LEN = 4

# Canonical surface-kind taxonomy per Holub-Fridrich-Denemark 2014 +
# Catalog #303 cargo-cult audit (every surface-kind cites a canonical
# reference; placeholder rejected).
VALID_SURFACE_KINDS: frozenset[str] = frozenset(
    {
        "chroma_lut_quantized_codebook",     # NSCS06 v8 chroma LUT (7th-order ANCHOR)
        "grayscale_lut_quantized_codebook",  # NSCS06 grayscale_lut (T3 #2)
        "vq_vae_indices_blob",               # VQ-VAE indices (T3 #3)
        "fec_selector_indices",              # FEC family selector indices
        "wyner_ziv_codec_layer",             # Wyner-Ziv side-info layer
        "scorer_class_softmax_indices",      # SegNet class softmax indices
        "dct_quantized_coefficient_blob",    # JPEG-DCT analog (canonical Fridrich domain)
        "arithmetic_coded_symbol_stream",    # ATW V1/V2 + Catalog #229 STC-Dasher
        "ans_coded_symbol_stream",           # ANS family (constriction)
        "huffman_coded_symbol_stream",       # Huffman canonical
    }
)
"""Canonical per-surface-kind taxonomy per Fridrich 14+ years steganalysis
literature. Each kind has a documented canonical reference (Sallee 2003,
Fridrich 2007, Holub-Fridrich-Denemark 2014, Filler-Judas-Fridrich 2011,
Yousfi 2017 detector-informed, etc.).
"""

VALID_INVARIANT_VERDICTS: frozenset[str] = frozenset(
    {
        "APPLICABLE_CANONICAL_FRIDRICH_NATURAL",
        "APPLICABLE_VARIANT_REQUIRES_FORMULA_ADAPTER",
        "INAPPLICABLE_RAW_FLOAT_DOMAIN",
        "INAPPLICABLE_NO_ENTROPY_CODING",
        "INAPPLICABLE_NO_PER_SYMBOL_ROUTABILITY",
        "INAPPLICABLE_NO_CANONICAL_FORMULA_GROUNDING",
        "UNKNOWN_PENDING_INVESTIGATION",
    }
)
"""Canonical 7-verdict taxonomy per the 4-condition UNIWARD applicability
test; one verdict per condition failure mode + 2 APPLICABLE classes."""

# Per Catalog #344 + CLAUDE.md "Canonical equations + models registry":
# the canonical equation id this enumerator enables (FORMALIZATION_PENDING
# until paired-CUDA empirical anchor lands).
CANONICAL_EQUATION_ID = (
    "uniward_canonical_application_surface_invariant_enumeration_v1"
)

# Per CLAUDE.md "SegNet vs PoseNet importance — operating-point dependent":
# the contest scorer canonical coefficients used for per-axis Taylor projection.
CANONICAL_SEG_COEFFICIENT = 100.0
CANONICAL_RATE_DENOM_BYTES = 37_545_489
CANONICAL_RATE_NUMERATOR = 25.0
CANONICAL_POSE_SQRT_INNER = 10.0


# ---------------------------------------------------------------------------
# Custom exception (canonical strict-load fail-closed sister per Catalog #138)
# ---------------------------------------------------------------------------


class UniwardInvariantEnumerationCorruptError(RuntimeError):
    """Strict-load corruption marker per Catalog #138 fail-closed discipline.

    Sister of:
      - :class:`tac.master_gradient.MasterGradientAnchorsCorruptError`
      - :class:`tac.deploy.modal.call_id_ledger.CallIdLedgerCorruptError`
      - :class:`tac.cross_substrate_master_gradient_analyzer.CrossSubstrateMasterGradientAnalysisCorruptError`

    The canonical strict-load helper raises this so any future consumer of
    the enumeration ledger inherits fail-closed-on-corruption semantics — a
    parse failure does NOT silently coerce missing rows to ``[]`` (the bug
    class Catalog #138 extincts).
    """


# ---------------------------------------------------------------------------
# Frozen dataclasses (canonical contract per Catalog #335 + #323)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UniwardCanonicalApplicationSurface:
    """Per-surface canonical-application descriptor.

    The descriptor captures the 4 canonical-application invariant conditions
    per Holub-Fridrich-Denemark 2014 + sister canonical references. The
    enumerator iterates ALL known surfaces in the codebase and instantiates
    one descriptor per surface.

    Per Catalog #287 placeholder rejection: every text field MUST be ≥4 chars
    substantive (placeholders rejected at ``__post_init__``).

    Per Catalog #323 canonical Provenance: the descriptor IS the canonical
    Provenance source-of-truth at the per-surface granularity; downstream
    consumers cite this descriptor by ``surface_id``.
    """

    surface_id: str
    """Canonical surface id (e.g. ``"nscs06_v8_chroma_lut"``)."""

    surface_kind: str
    """One of ``VALID_SURFACE_KINDS``."""

    substrate_id: str
    """Substrate id this surface belongs to (e.g. ``"nscs06_v8_chroma_lut"``).

    May be the same as ``surface_id`` for substrates with a single canonical
    application surface; differs when one substrate has multiple surfaces
    (e.g. dual-stream codec with quantizer + residual layer).
    """

    entropy_coded_axis: str
    """Per Holub-Fridrich-Denemark 2014 condition #1.

    One of: ``"arithmetic"`` / ``"range"`` / ``"huffman"`` / ``"ans"`` /
    ``"brotli"`` / ``"lzma"`` / ``"none"`` / ``"variable_length_integer"``.

    ``"none"`` means the surface is NOT entropy-coded and FAILS the canonical
    Fridrich application domain test on condition #1.
    """

    quantization_axis: str
    """Per Holub-Fridrich-Denemark 2014 condition #2.

    One of: ``"uint8"`` / ``"int8"`` / ``"int16"`` / ``"int32"`` /
    ``"none"`` (= raw float; fails condition #2 — the 5th + 6th order
    RGB tensor failure mode).
    """

    per_symbol_routable_axis: str
    """Per Holub-Fridrich-Denemark 2014 condition #3.

    One of: ``"direct_per_symbol"`` (canonical Fridrich) /
    ``"per_block_routable"`` / ``"per_pair_routable"`` /
    ``"none"`` (per-byte mutation does NOT produce predictable per-symbol
    distortion at decode time; fails condition #3).
    """

    canonical_formula_reference: str
    """Per Holub-Fridrich-Denemark 2014 condition #4.

    Canonical reference citation (≥4 chars substantive; placeholder
    rejected). Canonical refs:
      - ``"Holub-Fridrich-Denemark 2014 universal distortion"``
      - ``"Sallee 2003 weighted-median CDF"``
      - ``"Fridrich 2007 inverse-Fisher steganography"``
      - ``"Filler-Judas-Fridrich 2011 syndrome-trellis-code arithmetic"``
      - ``"Yousfi 2017 detector-informed JPEG steganography"``
      - ``"PENDING_INVESTIGATION:<surface_id>:<canonical_paper>"``
    """

    n_symbols_estimated: int
    """Estimated number of per-symbol routable units in the surface.

    Used for per-byte leverage normalization. For NSCS06 v8 chroma LUT:
    16 grayscale_levels × 5 segnet_classes × 3 RGB = 240. For VQ-VAE
    indices: number of indices in the blob. For FEC selector: 600 frames.
    """

    architecture_layer: str
    """Where in the substrate architecture this surface lives.

    One of: ``"compress_time_only"`` (compress-side derivation) /
    ``"inflate_time_only"`` (inflate-side lookup) / ``"both"`` /
    ``"sidecar"`` (separate archive member; canonical Fridrich entropy-coded
    sidecar surface).
    """

    notes: str
    """Substantive non-placeholder notes per Catalog #287 (≥4 chars)."""

    def __post_init__(self) -> None:
        if not self.surface_id or self.surface_id.strip() == "":
            raise ValueError("surface_id must be non-empty")
        if self.surface_kind not in VALID_SURFACE_KINDS:
            raise ValueError(
                f"surface_kind must be in {sorted(VALID_SURFACE_KINDS)}; "
                f"got {self.surface_kind!r}"
            )
        if not self.substrate_id or self.substrate_id.strip() == "":
            raise ValueError("substrate_id must be non-empty")
        if not self.entropy_coded_axis or self.entropy_coded_axis.strip() == "":
            raise ValueError("entropy_coded_axis must be non-empty")
        if not self.quantization_axis or self.quantization_axis.strip() == "":
            raise ValueError("quantization_axis must be non-empty")
        if not self.per_symbol_routable_axis or self.per_symbol_routable_axis.strip() == "":
            raise ValueError("per_symbol_routable_axis must be non-empty")
        if (
            not self.canonical_formula_reference
            or self.canonical_formula_reference.strip() == ""
            or self.canonical_formula_reference.strip() in _PLACEHOLDER_RATIONALES
            or len(self.canonical_formula_reference.strip()) < _MIN_RATIONALE_LEN
        ):
            raise ValueError(
                f"canonical_formula_reference must be substantive non-placeholder "
                f"(≥{_MIN_RATIONALE_LEN} chars; placeholder literals rejected per "
                f"Catalog #287); got {self.canonical_formula_reference!r}"
            )
        if self.n_symbols_estimated <= 0:
            raise ValueError("n_symbols_estimated must be positive")
        if not self.architecture_layer or self.architecture_layer.strip() == "":
            raise ValueError("architecture_layer must be non-empty")
        if (
            not self.notes
            or self.notes.strip() == ""
            or self.notes.strip() in _PLACEHOLDER_RATIONALES
            or len(self.notes.strip()) < _MIN_RATIONALE_LEN
        ):
            raise ValueError(
                f"notes must be substantive non-placeholder "
                f"(≥{_MIN_RATIONALE_LEN} chars; placeholder rejected per Catalog #287); "
                f"got {self.notes!r}"
            )

    def as_dict(self) -> dict:
        return {
            "surface_id": self.surface_id,
            "surface_kind": self.surface_kind,
            "substrate_id": self.substrate_id,
            "entropy_coded_axis": self.entropy_coded_axis,
            "quantization_axis": self.quantization_axis,
            "per_symbol_routable_axis": self.per_symbol_routable_axis,
            "canonical_formula_reference": self.canonical_formula_reference,
            "n_symbols_estimated": int(self.n_symbols_estimated),
            "architecture_layer": self.architecture_layer,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class UniwardApplicabilityVerdict:
    """Per-surface UNIWARD applicability verdict (4-condition test result).

    Each verdict is per-surface (not per-substrate); a substrate with
    multiple canonical-application surfaces gets multiple verdicts (one
    per surface).

    Per Catalog #341 routing markers: this verdict is OBSERVABILITY-ONLY
    by construction (carries non-promotable markers in
    :class:`UniwardInvariantEnumeration`).
    """

    surface_id: str
    """Canonical surface id (matches :attr:`UniwardCanonicalApplicationSurface.surface_id`)."""

    verdict: str
    """One of :data:`VALID_INVARIANT_VERDICTS` 7-verdict taxonomy."""

    condition_1_entropy_coded: bool
    """Per Holub-Fridrich-Denemark 2014 condition #1 result."""

    condition_2_quantized: bool
    """Per Holub-Fridrich-Denemark 2014 condition #2 result."""

    condition_3_per_symbol_routable: bool
    """Per Holub-Fridrich-Denemark 2014 condition #3 result."""

    condition_4_canonical_formula_grounded: bool
    """Per Holub-Fridrich-Denemark 2014 condition #4 result."""

    rationale: str
    """Substantive non-placeholder verdict rationale per Catalog #287
    (≥4 chars; placeholders rejected at ``__post_init__``)."""

    canonical_reference_cited: str
    """Canonical reference citation (matches
    :attr:`UniwardCanonicalApplicationSurface.canonical_formula_reference`
    for APPLICABLE verdicts; cites the failing-condition canonical for
    INAPPLICABLE verdicts).
    """

    def __post_init__(self) -> None:
        if not self.surface_id or self.surface_id.strip() == "":
            raise ValueError("surface_id must be non-empty")
        if self.verdict not in VALID_INVARIANT_VERDICTS:
            raise ValueError(
                f"verdict must be in {sorted(VALID_INVARIANT_VERDICTS)}; "
                f"got {self.verdict!r}"
            )
        if (
            not self.rationale
            or self.rationale.strip() == ""
            or self.rationale.strip() in _PLACEHOLDER_RATIONALES
            or len(self.rationale.strip()) < _MIN_RATIONALE_LEN
        ):
            raise ValueError(
                f"rationale must be substantive non-placeholder "
                f"(≥{_MIN_RATIONALE_LEN} chars; placeholder rejected per Catalog #287); "
                f"got {self.rationale!r}"
            )
        if not self.canonical_reference_cited or self.canonical_reference_cited.strip() == "":
            raise ValueError("canonical_reference_cited must be non-empty")
        # Cross-validate verdict against per-condition booleans.
        all_conditions_pass = (
            self.condition_1_entropy_coded
            and self.condition_2_quantized
            and self.condition_3_per_symbol_routable
            and self.condition_4_canonical_formula_grounded
        )
        verdict_is_applicable = self.verdict.startswith("APPLICABLE_")
        if all_conditions_pass and not verdict_is_applicable:
            raise ValueError(
                f"verdict={self.verdict!r} inconsistent with all 4 conditions PASS; "
                "must use APPLICABLE_ verdict prefix"
            )
        if not all_conditions_pass and verdict_is_applicable:
            raise ValueError(
                f"verdict={self.verdict!r} inconsistent with at least 1 condition FAIL; "
                "must use INAPPLICABLE_ or UNKNOWN_ verdict prefix"
            )

    def all_conditions_pass(self) -> bool:
        return (
            self.condition_1_entropy_coded
            and self.condition_2_quantized
            and self.condition_3_per_symbol_routable
            and self.condition_4_canonical_formula_grounded
        )

    def as_dict(self) -> dict:
        return {
            "surface_id": self.surface_id,
            "verdict": self.verdict,
            "condition_1_entropy_coded": bool(self.condition_1_entropy_coded),
            "condition_2_quantized": bool(self.condition_2_quantized),
            "condition_3_per_symbol_routable": bool(self.condition_3_per_symbol_routable),
            "condition_4_canonical_formula_grounded": bool(
                self.condition_4_canonical_formula_grounded
            ),
            "rationale": self.rationale,
            "canonical_reference_cited": self.canonical_reference_cited,
        }


@dataclass(frozen=True)
class RankedUniwardSurfaces:
    """Cross-surface ranking ordered by predicted ΔS per Cauchy-Schwarz bound.

    Per Catalog #356 per-axis decomposition: ranking is per-axis (separate
    rankings for ``seg`` / ``pose`` / ``rate``) because per-axis leverage
    can flip across operating points per CLAUDE.md "SegNet vs PoseNet
    importance — operating-point dependent".

    Per Catalog #341: every ranked entry is OBSERVABILITY-ONLY.
    """

    axis: str
    """One of ``"seg"`` / ``"pose"`` / ``"rate"``."""

    ranked_surface_ids: tuple[str, ...]
    """Surface ids ordered DESC by predicted ΔS per Cauchy-Schwarz bound."""

    per_surface_predicted_delta_s_upper_bound: tuple[float, ...]
    """Per-surface predicted ΔS upper bound (same order as
    ``ranked_surface_ids``). All non-negative; Cauchy-Schwarz upper bound.
    """

    per_surface_per_byte_leverage: tuple[float, ...]
    """Per-surface per-byte leverage = ||∇S||_2 / sqrt(N) (same order)."""

    canonical_equation_reference: str
    """Canonical equation citation: ``per_pair_master_gradient_score_impact_taylor_v1``."""

    def __post_init__(self) -> None:
        if self.axis not in VALID_AXIS_LABELS:
            raise ValueError(
                f"axis must be in {sorted(VALID_AXIS_LABELS)}; got {self.axis!r}"
            )
        n = len(self.ranked_surface_ids)
        if n != len(self.per_surface_predicted_delta_s_upper_bound):
            raise ValueError(
                "ranked_surface_ids length must match "
                "per_surface_predicted_delta_s_upper_bound length"
            )
        if n != len(self.per_surface_per_byte_leverage):
            raise ValueError(
                "ranked_surface_ids length must match per_surface_per_byte_leverage length"
            )
        for v in self.per_surface_predicted_delta_s_upper_bound:
            if not np.isfinite(v) or v < 0:
                raise ValueError(
                    "per_surface_predicted_delta_s_upper_bound entries must be non-negative finite"
                )
        for v in self.per_surface_per_byte_leverage:
            if not np.isfinite(v) or v < 0:
                raise ValueError(
                    "per_surface_per_byte_leverage entries must be non-negative finite"
                )
        # Verify DESC sort invariant.
        for i in range(1, n):
            prev = self.per_surface_predicted_delta_s_upper_bound[i - 1]
            curr = self.per_surface_predicted_delta_s_upper_bound[i]
            if prev < curr:
                raise ValueError(
                    f"ranked_surface_ids must be DESC by predicted_delta_s_upper_bound; "
                    f"position {i - 1}={prev} < position {i}={curr}"
                )
        if not self.canonical_equation_reference or self.canonical_equation_reference.strip() == "":
            raise ValueError("canonical_equation_reference must be non-empty")

    def as_dict(self) -> dict:
        return {
            "axis": self.axis,
            "ranked_surface_ids": list(self.ranked_surface_ids),
            "per_surface_predicted_delta_s_upper_bound": list(
                self.per_surface_predicted_delta_s_upper_bound
            ),
            "per_surface_per_byte_leverage": list(self.per_surface_per_byte_leverage),
            "canonical_equation_reference": self.canonical_equation_reference,
        }


@dataclass(frozen=True)
class UniwardInvariantEnumeration:
    """Canonical META-LIFT-4 UNIWARD invariant enumeration output.

    Sister of:
      - :class:`tac.cross_substrate_master_gradient_analyzer.CrossSubstrateMasterGradientAnalysis`
        (META-LIFT-1 at per-substrate aggregate granularity)
      - :class:`tac.pareto_polytope_unified_solver.ParetoPolytopeSolution`
        (META-LIFT-2 at Pareto polytope feasibility granularity)

    at the per-canonical-application-surface invariant sub-surface.

    Persisted to the canonical fcntl-locked JSONL ledger at
    ``.omx/state/uniward_invariant_enumerations.jsonl``.
    """

    schema_version: str

    enumeration_id: str
    """Deterministic id ``uniward_invariant_<utc_compact>_<n_surfaces>``."""

    measurement_utc: str

    surfaces: tuple[UniwardCanonicalApplicationSurface, ...]
    """Per-surface canonical-application descriptors (one per known surface)."""

    verdicts: tuple[UniwardApplicabilityVerdict, ...]
    """Per-surface verdicts (one per surface; same order as ``surfaces``)."""

    rankings_per_axis: tuple[RankedUniwardSurfaces, ...]
    """Per-axis ranked surfaces (3-tuple: seg, pose, rate)."""

    n_applicable_surfaces: int
    """Count of surfaces where verdict ∈ APPLICABLE_*."""

    n_inapplicable_surfaces: int
    """Count of surfaces where verdict ∈ INAPPLICABLE_*."""

    n_unknown_surfaces: int
    """Count of surfaces where verdict = UNKNOWN_PENDING_INVESTIGATION."""

    axis_tag: str
    """Always ``"[predicted]"`` per Catalog #341."""

    score_claim: bool
    """Always ``False`` per CLAUDE.md "Apples-to-apples evidence discipline"."""

    promotable: bool
    """Always ``False`` per Catalog #341 + #192."""

    evidence_grade: str

    canonical_helper_invocation: str
    """``"tac.uniward_invariant_enumerator.enumerate_uniward_canonical_application_surfaces"``."""

    canonical_equation_id: str

    canonical_equation_status: str
    """``"FORMALIZATION_PENDING"`` until paired-CUDA empirical anchor."""

    written_at_utc: str = ""
    written_pid: int = 0
    written_host: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must equal {SCHEMA_VERSION!r}; got {self.schema_version!r}"
            )
        if not self.enumeration_id:
            raise ValueError("enumeration_id must be non-empty")
        if not self.measurement_utc:
            raise ValueError("measurement_utc must be non-empty")
        if not self.surfaces:
            raise ValueError("surfaces must be non-empty")
        if len(self.surfaces) != len(self.verdicts):
            raise ValueError(
                f"surfaces length ({len(self.surfaces)}) must equal verdicts length ({len(self.verdicts)})"
            )
        # Cross-validate surface_ids match between surfaces and verdicts.
        for surf, verd in zip(self.surfaces, self.verdicts):
            if surf.surface_id != verd.surface_id:
                raise ValueError(
                    f"surfaces[i].surface_id={surf.surface_id!r} must equal "
                    f"verdicts[i].surface_id={verd.surface_id!r}"
                )
        if len(self.rankings_per_axis) != 3:
            raise ValueError(
                f"rankings_per_axis must have exactly 3 entries (seg, pose, rate); "
                f"got {len(self.rankings_per_axis)}"
            )
        seen_axes = {r.axis for r in self.rankings_per_axis}
        if seen_axes != VALID_AXIS_LABELS:
            raise ValueError(
                f"rankings_per_axis must cover all axes {sorted(VALID_AXIS_LABELS)}; "
                f"got {sorted(seen_axes)}"
            )
        if self.axis_tag != PREDICTED_AXIS_TAG:
            raise ValueError(f"axis_tag must equal {PREDICTED_AXIS_TAG!r}; got {self.axis_tag!r}")
        if self.score_claim is not False:
            raise ValueError("score_claim must be False per Catalog #341")
        if self.promotable is not False:
            raise ValueError("promotable must be False per Catalog #341")
        if not self.evidence_grade.startswith("[predicted;"):
            raise ValueError("evidence_grade must start with '[predicted;' per Catalog #287 / #323")
        if self.canonical_equation_id != CANONICAL_EQUATION_ID:
            raise ValueError(
                f"canonical_equation_id must equal {CANONICAL_EQUATION_ID!r}; "
                f"got {self.canonical_equation_id!r}"
            )
        if self.canonical_equation_status not in {"FORMALIZATION_PENDING", "REGISTERED"}:
            raise ValueError(
                "canonical_equation_status must be 'FORMALIZATION_PENDING' or 'REGISTERED' "
                "per Catalog #344"
            )
        # Cross-validate count fields.
        expected_applicable = sum(1 for v in self.verdicts if v.verdict.startswith("APPLICABLE_"))
        expected_inapplicable = sum(
            1 for v in self.verdicts if v.verdict.startswith("INAPPLICABLE_")
        )
        expected_unknown = sum(
            1 for v in self.verdicts if v.verdict == "UNKNOWN_PENDING_INVESTIGATION"
        )
        if self.n_applicable_surfaces != expected_applicable:
            raise ValueError(
                f"n_applicable_surfaces={self.n_applicable_surfaces} must equal "
                f"actual count={expected_applicable}"
            )
        if self.n_inapplicable_surfaces != expected_inapplicable:
            raise ValueError(
                f"n_inapplicable_surfaces={self.n_inapplicable_surfaces} must equal "
                f"actual count={expected_inapplicable}"
            )
        if self.n_unknown_surfaces != expected_unknown:
            raise ValueError(
                f"n_unknown_surfaces={self.n_unknown_surfaces} must equal "
                f"actual count={expected_unknown}"
            )

    def as_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "enumeration_id": self.enumeration_id,
            "measurement_utc": self.measurement_utc,
            "surfaces": [s.as_dict() for s in self.surfaces],
            "verdicts": [v.as_dict() for v in self.verdicts],
            "rankings_per_axis": [r.as_dict() for r in self.rankings_per_axis],
            "n_applicable_surfaces": int(self.n_applicable_surfaces),
            "n_inapplicable_surfaces": int(self.n_inapplicable_surfaces),
            "n_unknown_surfaces": int(self.n_unknown_surfaces),
            "axis_tag": self.axis_tag,
            "score_claim": self.score_claim,
            "promotable": self.promotable,
            "evidence_grade": self.evidence_grade,
            "canonical_helper_invocation": self.canonical_helper_invocation,
            "canonical_equation_id": self.canonical_equation_id,
            "canonical_equation_status": self.canonical_equation_status,
            "written_at_utc": self.written_at_utc,
            "written_pid": int(self.written_pid),
            "written_host": self.written_host,
        }


# ---------------------------------------------------------------------------
# Static canonical-application-surface registry
# ---------------------------------------------------------------------------


def _canonical_surface_registry() -> tuple[UniwardCanonicalApplicationSurface, ...]:
    """Static canonical-application-surface registry.

    Per CLAUDE.md UNIQUE-AND-COMPLETE-PER-METHOD: each surface entry is
    canonical-distinct (not a generic placeholder). Compiled at import time
    from substrate architecture introspection + canonical Fridrich literature
    references.

    Adding a new canonical-application surface = adding one entry here +
    re-running ``enumerate_uniward_canonical_application_surfaces``.
    """
    return (
        # 7th-order ANCHOR (PARADIGM-VALIDATED-AT-ENTROPY-CODED-SIDECAR
        # commit 87bd1c355 2026-05-26): NSCS06 v8 chroma LUT
        UniwardCanonicalApplicationSurface(
            surface_id="nscs06_v8_chroma_lut",
            surface_kind="chroma_lut_quantized_codebook",
            substrate_id="nscs06_v8_chroma_lut",
            entropy_coded_axis="brotli",
            quantization_axis="uint8",
            per_symbol_routable_axis="direct_per_symbol",
            canonical_formula_reference=(
                "Holub-Fridrich-Denemark 2014 universal distortion + "
                "Sallee 2003 weighted-median CDF (per-LUT-index UNIWARD)"
            ),
            n_symbols_estimated=16 * 5 * 3,  # levels x classes x RGB
            architecture_layer="both",
            notes=(
                "7th-order PARADIGM-VALIDATED anchor commit 87bd1c355; "
                "43/72 nonempty bins routed; 576x dynamic range; "
                "canonical v8 CH08 archive grammar byte-stream compatible"
            ),
        ),
        # T3 council #2 stacking candidate: NSCS06 grayscale LUT
        UniwardCanonicalApplicationSurface(
            surface_id="nscs06_grayscale_lut",
            surface_kind="grayscale_lut_quantized_codebook",
            substrate_id="nscs06_grayscale_lut",
            entropy_coded_axis="brotli",
            quantization_axis="uint8",
            per_symbol_routable_axis="direct_per_symbol",
            canonical_formula_reference=(
                "Holub-Fridrich-Denemark 2014 universal distortion + "
                "Sallee 2003 weighted-median CDF (sister of v8 chroma LUT)"
            ),
            n_symbols_estimated=16 * 5,  # levels x classes (grayscale)
            architecture_layer="both",
            notes=(
                "T3 council PR110 stacking #2 candidate; sister-test pending "
                "per 7th-order memo operator-routable alternative C scope"
            ),
        ),
        # T3 council #3 stacking candidate: VQ-VAE indices_blob
        UniwardCanonicalApplicationSurface(
            surface_id="vq_vae_indices_blob",
            surface_kind="vq_vae_indices_blob",
            substrate_id="vq_vae",
            entropy_coded_axis="huffman",
            quantization_axis="uint8",
            per_symbol_routable_axis="direct_per_symbol",
            canonical_formula_reference=(
                "Holub-Fridrich-Denemark 2014 universal distortion + "
                "Sallee 2003 weighted-median CDF (vq_vae sister application)"
            ),
            n_symbols_estimated=600 * 32,  # nominal: 600 pairs x 32 indices
            architecture_layer="both",
            notes=(
                "T3 council PR110 stacking #3 candidate; sister-test pending "
                "per 7th-order memo operator-routable alternative C scope"
            ),
        ),
        # FEC family selector indices (FEC6 / FEC8 / FEC9 / FEC10 cascade)
        UniwardCanonicalApplicationSurface(
            surface_id="fec_selector_indices_per_frame",
            surface_kind="fec_selector_indices",
            substrate_id="fec_cascade_family",
            entropy_coded_axis="brotli",
            quantization_axis="uint8",
            per_symbol_routable_axis="direct_per_symbol",
            canonical_formula_reference=(
                "Holub-Fridrich-Denemark 2014 universal distortion adapter + "
                "Filler-Judas-Fridrich 2011 STC routing (per-frame selector)"
            ),
            n_symbols_estimated=600,  # 600 frames
            architecture_layer="sidecar",
            notes=(
                "FEC family rate attack canonical sidecar (FEC6 249B / "
                "FEC8 245B 1st-order / FEC8 166B 2nd-order Markov / FEC10 in-flight)"
            ),
        ),
        # Wyner-Ziv codec layer (per Catalog #319 deliverability_proof)
        UniwardCanonicalApplicationSurface(
            surface_id="wyner_ziv_codec_layer",
            surface_kind="wyner_ziv_codec_layer",
            substrate_id="wyner_ziv_cooperative_receiver",
            entropy_coded_axis="ans",
            quantization_axis="int16",
            per_symbol_routable_axis="per_pair_routable",
            canonical_formula_reference=(
                "Holub-Fridrich-Denemark 2014 + Wyner-Ziv 1976 side-information + "
                "Atick-Redlich 1990 cooperative-receiver"
            ),
            n_symbols_estimated=600 * 8,  # per-pair coded symbols
            architecture_layer="sidecar",
            notes=(
                "Wyner-Ziv codec layer per Catalog #319 deliverability_proof; "
                "per-pair routable (not direct per-symbol) — UNIWARD adapter required"
            ),
        ),
        # ATW arithmetic-coded symbol stream (ATW V1 + V2 dead-section)
        UniwardCanonicalApplicationSurface(
            surface_id="atw_arithmetic_coded_symbol_stream",
            surface_kind="arithmetic_coded_symbol_stream",
            substrate_id="atw_codec",
            entropy_coded_axis="arithmetic",
            quantization_axis="int16",
            per_symbol_routable_axis="direct_per_symbol",
            canonical_formula_reference=(
                "Holub-Fridrich-Denemark 2014 + Filler-Judas-Fridrich 2011 STC + "
                "Yousfi 2017 detector-informed (canonical Fridrich-natural domain)"
            ),
            n_symbols_estimated=2560,
            architecture_layer="sidecar",
            notes=(
                "ATW V1 active arithmetic-coded symbol stream; "
                "V2 dead-section EMPIRICALLY-FALSIFIED via byte-mutation smoke 2026-05-21"
            ),
        ),
        # DCT analog quantized coefficient blob (canonical JPEG Fridrich domain)
        UniwardCanonicalApplicationSurface(
            surface_id="dct_analog_quantized_coefficient_blob",
            surface_kind="dct_quantized_coefficient_blob",
            substrate_id="dct_analog_codec",
            entropy_coded_axis="huffman",
            quantization_axis="int16",
            per_symbol_routable_axis="direct_per_symbol",
            canonical_formula_reference=(
                "Holub-Fridrich-Denemark 2014 ORIGINAL canonical UNIWARD "
                "application on JPEG DCT (the canonical training-data domain)"
            ),
            n_symbols_estimated=8 * 8 * 600,  # 8x8 DCT block x 600 frames nominal
            architecture_layer="sidecar",
            notes=(
                "Canonical Fridrich JPEG DCT analog — the ORIGINAL UNIWARD "
                "application surface; sister substrate scaffold pending"
            ),
        ),
        # SegNet class softmax indices (per-pair per-pixel)
        UniwardCanonicalApplicationSurface(
            surface_id="segnet_class_softmax_indices",
            surface_kind="scorer_class_softmax_indices",
            substrate_id="segnet_argmax_residual",
            entropy_coded_axis="ans",
            quantization_axis="uint8",
            per_symbol_routable_axis="direct_per_symbol",
            canonical_formula_reference=(
                "Holub-Fridrich-Denemark 2014 + Fridrich 2007 inverse-Fisher + "
                "Sallee 2003 weighted-median CDF (per-class routing)"
            ),
            n_symbols_estimated=5 * 384 * 512,  # 5 classes x H x W
            architecture_layer="compress_time_only",
            notes=(
                "SegNet 5-class argmax indices at compress time; canonical "
                "per-class routable surface per Fridrich inverse-Fisher"
            ),
        ),
        # ANS coded symbol stream (constriction sister)
        UniwardCanonicalApplicationSurface(
            surface_id="ans_coded_symbol_stream_constriction",
            surface_kind="ans_coded_symbol_stream",
            substrate_id="constriction_ans_codec",
            entropy_coded_axis="ans",
            quantization_axis="int16",
            per_symbol_routable_axis="direct_per_symbol",
            canonical_formula_reference=(
                "Holub-Fridrich-Denemark 2014 + Duda 2013 ANS canonical + "
                "Filler-Judas-Fridrich 2011 STC sister"
            ),
            n_symbols_estimated=600 * 16,
            architecture_layer="sidecar",
            notes=(
                "constriction ANS canonical (rANS / tANS); per-symbol "
                "routable via canonical Duda 2013 + Fridrich STC adapter"
            ),
        ),
        # Master-gradient per-byte (raw byte authority FORBIDDEN per Catalog #318)
        UniwardCanonicalApplicationSurface(
            surface_id="master_gradient_per_byte_raw_authority",
            surface_kind="dct_quantized_coefficient_blob",  # closest taxonomy
            substrate_id="master_gradient_canonical",
            entropy_coded_axis="none",  # raw bytes — fails condition #1
            quantization_axis="none",   # raw float — fails condition #2
            per_symbol_routable_axis="none",  # raw byte — fails condition #3
            canonical_formula_reference=(
                "Catalog #318 FORBIDDEN raw byte authority — fails ALL 4 "
                "Fridrich application conditions; INAPPLICABLE by construction"
            ),
            n_symbols_estimated=1,  # nominal; not applicable
            architecture_layer="compress_time_only",
            notes=(
                "Per Catalog #318 master-gradient raw-byte-authority is "
                "FORBIDDEN; included here as anti-example for invariant test"
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Canonical helpers (4 public entry points)
# ---------------------------------------------------------------------------


def verify_uniward_applicability(
    surface: UniwardCanonicalApplicationSurface,
) -> UniwardApplicabilityVerdict:
    """Per-surface UNIWARD applicability verdict (4-condition canonical test).

    Per Holub-Fridrich-Denemark 2014 + Sallee 2003 + Fridrich 2007 canonical
    references: a surface is UNIWARD-applicable iff ALL 4 conditions PASS.

    Returns
    -------
    UniwardApplicabilityVerdict
        Per-condition booleans + 7-verdict taxonomy classification.

    Notes
    -----
    The verdict is OBSERVABILITY-ONLY per Catalog #341 (carries the canonical
    non-promotable markers via :class:`UniwardInvariantEnumeration` wrapper).
    """
    # Condition #1: ENTROPY_CODED.
    entropy_coded_pass = surface.entropy_coded_axis.strip().lower() not in {"none", ""}

    # Condition #2: QUANTIZED.
    quantized_pass = surface.quantization_axis.strip().lower() not in {"none", ""}

    # Condition #3: PER_SYMBOL_ROUTABLE.
    routable_pass = surface.per_symbol_routable_axis.strip().lower() in {
        "direct_per_symbol",
        "per_block_routable",
        "per_pair_routable",
    }

    # Condition #4: CANONICAL_FORMULA_GROUNDED.
    canonical_refs_lower = surface.canonical_formula_reference.lower()
    canonical_grounded_pass = any(
        ref in canonical_refs_lower
        for ref in (
            "holub-fridrich-denemark 2014",
            "sallee 2003",
            "fridrich 2007",
            "filler-judas-fridrich 2011",
            "yousfi 2017",
            "wyner-ziv 1976",
            "atick-redlich 1990",
            "duda 2013",
        )
    )

    all_pass = (
        entropy_coded_pass
        and quantized_pass
        and routable_pass
        and canonical_grounded_pass
    )

    # Classify verdict per the 4-condition pattern.
    if all_pass:
        # Canonical Fridrich-natural = direct per-symbol routable; variant
        # = per_block or per_pair routable.
        if surface.per_symbol_routable_axis == "direct_per_symbol":
            verdict = "APPLICABLE_CANONICAL_FRIDRICH_NATURAL"
            rationale = (
                f"Surface {surface.surface_id!r} passes ALL 4 canonical "
                "Fridrich application conditions with direct-per-symbol "
                "routability (the canonical natural domain)."
            )
        else:
            verdict = "APPLICABLE_VARIANT_REQUIRES_FORMULA_ADAPTER"
            rationale = (
                f"Surface {surface.surface_id!r} passes ALL 4 conditions but "
                f"with {surface.per_symbol_routable_axis!r} routability; "
                "requires canonical formula adapter (e.g., per-block STC)."
            )
    elif not entropy_coded_pass:
        verdict = "INAPPLICABLE_NO_ENTROPY_CODING"
        rationale = (
            f"Surface {surface.surface_id!r} FAILS condition #1 "
            f"(entropy_coded_axis={surface.entropy_coded_axis!r})."
        )
    elif not quantized_pass:
        verdict = "INAPPLICABLE_RAW_FLOAT_DOMAIN"
        rationale = (
            f"Surface {surface.surface_id!r} FAILS condition #2 "
            f"(quantization_axis={surface.quantization_axis!r}); raw-float "
            "domain — see 5th + 6th order PARADIGM-NULL precedent."
        )
    elif not routable_pass:
        verdict = "INAPPLICABLE_NO_PER_SYMBOL_ROUTABILITY"
        rationale = (
            f"Surface {surface.surface_id!r} FAILS condition #3 "
            f"(per_symbol_routable_axis={surface.per_symbol_routable_axis!r})."
        )
    elif not canonical_grounded_pass:
        verdict = "INAPPLICABLE_NO_CANONICAL_FORMULA_GROUNDING"
        rationale = (
            f"Surface {surface.surface_id!r} FAILS condition #4; "
            "canonical_formula_reference does not cite any of the canonical "
            "Fridrich-family references (Holub-Fridrich-Denemark 2014, "
            "Sallee 2003, Fridrich 2007, Filler-Judas-Fridrich 2011, "
            "Yousfi 2017, Wyner-Ziv 1976, Atick-Redlich 1990, Duda 2013)."
        )
    else:
        # Defensive — shouldn't reach here.
        verdict = "UNKNOWN_PENDING_INVESTIGATION"
        rationale = (
            f"Surface {surface.surface_id!r}: unexpected condition combination "
            "— operator investigation required."
        )

    return UniwardApplicabilityVerdict(
        surface_id=surface.surface_id,
        verdict=verdict,
        condition_1_entropy_coded=entropy_coded_pass,
        condition_2_quantized=quantized_pass,
        condition_3_per_symbol_routable=routable_pass,
        condition_4_canonical_formula_grounded=canonical_grounded_pass,
        rationale=rationale,
        canonical_reference_cited=surface.canonical_formula_reference,
    )


def rank_uniward_applicable_surfaces_by_predicted_delta_s(
    surfaces: Sequence[UniwardCanonicalApplicationSurface],
    verdicts: Sequence[UniwardApplicabilityVerdict],
    *,
    axis: str,
    per_surface_gradient_l2_norms: Mapping[str, float] | None = None,
) -> RankedUniwardSurfaces:
    """Per-axis ranking by predicted ΔS upper bound (Cauchy-Schwarz).

    For each APPLICABLE surface, compute:
        leverage = ||∇S_axis||_2 / sqrt(N_symbols)
        upper_bound = ||∇S_axis||_2 · ||Δθ_unit||_2

    where ``||Δθ_unit||_2 = sqrt(N_symbols)`` for unit-norm perturbation.

    Surfaces without empirical gradient data are ranked by a structural prior:
        leverage_structural = 1.0 / sqrt(N_symbols)
        upper_bound_structural = sqrt(N_symbols) (canonical structural prior)

    INAPPLICABLE surfaces are ranked LAST (upper_bound = 0.0).

    Parameters
    ----------
    surfaces : sequence
        Canonical-application-surface descriptors.
    verdicts : sequence
        Per-surface verdicts (same order; same length).
    axis : str
        One of ``"seg"`` / ``"pose"`` / ``"rate"``.
    per_surface_gradient_l2_norms : mapping[str, float] | None
        Optional map ``{surface_id: ||∇S_axis||_2}`` from cached
        per-substrate master-gradient anchors. Surfaces missing from the
        mapping use the canonical structural prior.

    Returns
    -------
    RankedUniwardSurfaces
        Surfaces ordered DESC by predicted ΔS upper bound.
    """
    if axis not in VALID_AXIS_LABELS:
        raise ValueError(f"axis must be in {sorted(VALID_AXIS_LABELS)}; got {axis!r}")
    if len(surfaces) != len(verdicts):
        raise ValueError(
            f"surfaces length ({len(surfaces)}) must equal verdicts length ({len(verdicts)})"
        )

    norms = per_surface_gradient_l2_norms or {}

    entries: list[tuple[str, float, float]] = []  # (surface_id, upper_bound, leverage)
    for surface, verdict in zip(surfaces, verdicts):
        if not verdict.all_conditions_pass():
            # INAPPLICABLE — rank last with zero upper bound.
            entries.append((surface.surface_id, 0.0, 0.0))
            continue
        n = max(1, surface.n_symbols_estimated)
        sqrt_n = float(np.sqrt(n))
        gradient_l2 = float(norms.get(surface.surface_id, 1.0))  # structural prior=1.0
        leverage = gradient_l2 / sqrt_n
        upper_bound = gradient_l2 * sqrt_n
        entries.append((surface.surface_id, upper_bound, leverage))

    # Sort DESC by upper_bound (stable; secondary by surface_id for determinism).
    entries.sort(key=lambda e: (-e[1], e[0]))

    ranked_ids = tuple(e[0] for e in entries)
    ranked_bounds = tuple(float(e[1]) for e in entries)
    ranked_leverages = tuple(float(e[2]) for e in entries)

    return RankedUniwardSurfaces(
        axis=axis,
        ranked_surface_ids=ranked_ids,
        per_surface_predicted_delta_s_upper_bound=ranked_bounds,
        per_surface_per_byte_leverage=ranked_leverages,
        canonical_equation_reference=(
            "per_pair_master_gradient_score_impact_taylor_v1 (canonical "
            "equation #344) + Cauchy-Schwarz upper bound"
        ),
    )


def enumerate_uniward_canonical_application_surfaces(
    *,
    per_surface_gradient_l2_norms_per_axis: Mapping[str, Mapping[str, float]] | None = None,
    custom_surfaces: Sequence[UniwardCanonicalApplicationSurface] | None = None,
) -> UniwardInvariantEnumeration:
    """Iterate ALL canonical-application surfaces + emit ranked enumeration.

    The canonical META-LIFT-4 entry point. Loads the static canonical
    application surface registry (or accepts ``custom_surfaces`` for testing),
    computes per-surface UNIWARD applicability verdicts via
    :func:`verify_uniward_applicability`, and ranks per-axis via
    :func:`rank_uniward_applicable_surfaces_by_predicted_delta_s`.

    Parameters
    ----------
    per_surface_gradient_l2_norms_per_axis : mapping[axis, mapping[surface_id, float]] | None
        Optional per-axis per-surface gradient L2 norms (loaded from cached
        master-gradient anchors). When provided, ranking incorporates
        empirical gradient signal; otherwise uses structural prior.
    custom_surfaces : sequence | None
        Optional custom surface registry (for testing).

    Returns
    -------
    UniwardInvariantEnumeration
        Frozen canonical enumeration (observability-only per Catalog #341).
    """
    surfaces = tuple(custom_surfaces) if custom_surfaces else _canonical_surface_registry()
    verdicts = tuple(verify_uniward_applicability(s) for s in surfaces)

    norms_per_axis = per_surface_gradient_l2_norms_per_axis or {}
    rankings_list = []
    for axis in ("seg", "pose", "rate"):  # canonical axis order
        axis_norms = norms_per_axis.get(axis)
        rankings_list.append(
            rank_uniward_applicable_surfaces_by_predicted_delta_s(
                surfaces, verdicts, axis=axis, per_surface_gradient_l2_norms=axis_norms
            )
        )
    rankings = tuple(rankings_list)

    n_applicable = sum(1 for v in verdicts if v.verdict.startswith("APPLICABLE_"))
    n_inapplicable = sum(1 for v in verdicts if v.verdict.startswith("INAPPLICABLE_"))
    n_unknown = sum(1 for v in verdicts if v.verdict == "UNKNOWN_PENDING_INVESTIGATION")

    now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat()
    utc_compact = now_utc.replace(":", "").replace("-", "").replace(".", "").split("+")[0]
    enumeration_id = f"uniward_invariant_{utc_compact}_{len(surfaces)}"

    return UniwardInvariantEnumeration(
        schema_version=SCHEMA_VERSION,
        enumeration_id=enumeration_id,
        measurement_utc=now_utc,
        surfaces=surfaces,
        verdicts=verdicts,
        rankings_per_axis=rankings,
        n_applicable_surfaces=n_applicable,
        n_inapplicable_surfaces=n_inapplicable,
        n_unknown_surfaces=n_unknown,
        axis_tag=PREDICTED_AXIS_TAG,
        score_claim=False,
        promotable=False,
        evidence_grade="[predicted; uniward-canonical-application-surface-enumeration]",
        canonical_helper_invocation=(
            "tac.uniward_invariant_enumerator.enumerate_uniward_canonical_application_surfaces"
        ),
        canonical_equation_id=CANONICAL_EQUATION_ID,
        canonical_equation_status="FORMALIZATION_PENDING",
    )


# ---------------------------------------------------------------------------
# Canonical fcntl-locked JSONL ledger I/O (Catalog #131 / #138 / #245)
# ---------------------------------------------------------------------------


@contextmanager
def _ledger_lock():
    """fcntl LOCK_EX on the canonical lock path."""
    _LEDGER_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(_LEDGER_LOCK_PATH), os.O_CREAT | os.O_RDWR, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def append_enumeration_locked(
    enumeration: UniwardInvariantEnumeration,
    ledger_path: Path | None = None,
) -> None:
    """Append a UNIWARD invariant enumeration to the canonical ledger.

    Per Catalog #131 / #138 / #245 fcntl-locked sister discipline. Writes
    atomically via ``.tmp.<uuid>`` + ``os.replace`` inside ``LOCK_EX``.
    """
    target = ledger_path or UNIWARD_INVARIANT_ENUMERATIONS_LEDGER_PATH
    target.parent.mkdir(parents=True, exist_ok=True)

    # Stamp written_* fields.
    stamped = UniwardInvariantEnumeration(
        schema_version=enumeration.schema_version,
        enumeration_id=enumeration.enumeration_id,
        measurement_utc=enumeration.measurement_utc,
        surfaces=enumeration.surfaces,
        verdicts=enumeration.verdicts,
        rankings_per_axis=enumeration.rankings_per_axis,
        n_applicable_surfaces=enumeration.n_applicable_surfaces,
        n_inapplicable_surfaces=enumeration.n_inapplicable_surfaces,
        n_unknown_surfaces=enumeration.n_unknown_surfaces,
        axis_tag=enumeration.axis_tag,
        score_claim=enumeration.score_claim,
        promotable=enumeration.promotable,
        evidence_grade=enumeration.evidence_grade,
        canonical_helper_invocation=enumeration.canonical_helper_invocation,
        canonical_equation_id=enumeration.canonical_equation_id,
        canonical_equation_status=enumeration.canonical_equation_status,
        written_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        written_pid=os.getpid(),
        written_host=socket.gethostname(),
    )

    payload_line = json.dumps(stamped.as_dict(), sort_keys=True) + "\n"

    with _ledger_lock():
        import uuid as _uuid

        tmp_path = target.with_suffix(target.suffix + f".tmp.{_uuid.uuid4().hex[:12]}")
        existing_content = target.read_bytes() if target.exists() else b""
        tmp_path.write_bytes(existing_content + payload_line.encode("utf-8"))
        os.replace(str(tmp_path), str(target))


def load_enumerations_strict(
    ledger_path: Path | None = None,
) -> list[Mapping]:
    """Strict-load enumerations per Catalog #138 fail-closed.

    Raises :class:`UniwardInvariantEnumerationCorruptError` on JSON parse
    failure (no silent coercion to ``[]``).
    """
    target = ledger_path or UNIWARD_INVARIANT_ENUMERATIONS_LEDGER_PATH
    if not target.exists():
        return []
    rows: list[Mapping] = []
    for line_no, raw in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise UniwardInvariantEnumerationCorruptError(
                f"line {line_no}: JSON decode error: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise UniwardInvariantEnumerationCorruptError(
                f"line {line_no}: row is not a dict; got {type(row).__name__}"
            )
        rows.append(row)
    return rows
