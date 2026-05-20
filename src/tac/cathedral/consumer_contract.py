# SPDX-License-Identifier: MIT
"""Canonical contract for cathedral autopilot auto-ingested consumers.

Per CLAUDE.md "Subagent coherence-by-default" non-negotiable + Catalog #125
6-hook wire-in + operator directive 2026-05-19 *"What if we change the
paradigm by making cathedral autopilot ingest by default if within a certain
directory and exposing/respecting a certain contract or schema. Fix
permanently and self protect against"*.

Every package in ``src/tac/cathedral_consumers/`` MUST expose this contract
OR carry ``# CATHEDRAL_CONSUMER_DEFERRED_OK:<rationale>`` waiver in
``__init__.py`` first 30 lines per Catalog #335 STRICT preflight gate.

Sister of:
- Catalog #265 ``check_symposium_impls_canonical_contract`` (same canonical
  contract pattern at the symposium-impl surface)
- Catalog #125 6-hook wire-in non-negotiable (this contract IS the
  structural extinction of hook #4 cathedral autopilot dispatch)
- ``tac.atom.linguistic_extensions`` + ``tac.council_continual_learning``
  (both expose ``update_from_anchor`` as the canonical contract token)

The contract is intentionally minimal:
1. Module-level metadata (CONSUMER_NAME / CONSUMER_VERSION / CONSUMER_HOOK_NUMBERS)
2. ``update_from_anchor(anchor)`` — Catalog #125 hook #5 (continual-learning posterior)
3. ``consume_candidate(candidate) -> dict`` — Catalog #125 hook #4 (cathedral dispatch)

Validation is structural (importable, satisfies Protocol, fields well-typed).
Runtime correctness is the consumer's responsibility.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable


WAIVER_TOKEN = "CATHEDRAL_CONSUMER_DEFERRED_OK"
"""Canonical same-line waiver token for non-compliant consumer packages.

Format: ``# CATHEDRAL_CONSUMER_DEFERRED_OK:<rationale>`` in __init__.py first
30 lines. Placeholder rationales (``<rationale>`` / ``<reason>`` literals,
empty, or <4 chars) rejected per Catalog #287 sister discipline.
"""

# Placeholder waiver rationale literals refused per Catalog #287 sister.
_PLACEHOLDER_RATIONALES = ("<rationale>", "<reason>", "rationale", "reason")
_MIN_RATIONALE_LEN = 4
_WAIVER_LOOKBACK_LINES = 30


class HookNumber(IntEnum):
    """Catalog #125 6-hook wire-in surfaces.

    Per CLAUDE.md "Subagent coherence-by-default" non-negotiable, every
    landing must declare which of the 6 canonical hooks it consumes (or
    explicitly mark N/A with rationale).
    """

    SENSITIVITY_MAP = 1
    PARETO_CONSTRAINT = 2
    BIT_ALLOCATOR = 3
    CATHEDRAL_AUTOPILOT_DISPATCH = 4
    CONTINUAL_LEARNING_POSTERIOR = 5
    PROBE_DISAMBIGUATOR = 6


class ConsumerTier(IntEnum):
    """Dual-tier cathedral consumer architecture per CATHEDRAL-SMARTER-DESIGN-MEMO
    Dimension 6 Step 6.1 (2026-05-20).

    Resolves the operator-mental-model gap surfaced by WIRE-IN-RIGOR audit:
    44+ cathedral consumers fire per loop iteration but per Catalog #341 ALL
    return ``predicted_delta_adjustment=0.0`` (observability-only, by design).
    The operator-mental-model gap was *"are these consumers FAKE?"* — the
    answer is NO: Tier A is safe-by-construction non-promotable per CLAUDE.md
    "Forbidden score claims" + "MPS auth eval is NOISE" non-negotiables.

    **TIER_A_OBSERVABILITY_ONLY** (default; backward-compatible):

    - Catalog #341 canonical contract.
    - ``predicted_delta_adjustment`` MUST be ``0.0``.
    - Routing / annotation / diagnostic only.
    - Safe-by-construction: cannot leak into score signal or promotion.
    - Existing 44+ production consumers stay in Tier A.

    **TIER_B_SCORE_CONTRIBUTING** (future extension; per Dim 6 Step 6.5):

    - ``predicted_delta_adjustment`` MAY be non-zero (signed; bounded; finite).
    - REQUIRES canonical Provenance per Catalog #323 in the return dict
      (``provenance`` field with axis + hardware + grade triple).
    - REQUIRES ``axis_tag != "[predicted]"`` (must be canonical contest or
      diagnostic axis honoring CLAUDE.md "Apples-to-apples").
    - REQUIRES ``promotable=False`` STILL — Tier B contributes to RANKING
      but NEVER to PROMOTION (promotion requires paired contest-CPU +
      contest-CUDA empirical anchors per CLAUDE.md "Submission auth eval
      — BOTH CPU AND CUDA").
    - REQUIRES ``predicted_axis_decomposition`` per :class:`AxisDecomposition`
      with non-empty ``canonical_provenance`` (Dim 3 Step 3.1 prerequisite).
    - Enforced structurally by Catalog #357 STRICT preflight gate.

    The Tier B / Tier A distinction: ranking influence is design-intentful
    (a well-calibrated Bayesian posterior IS a legitimate ranking signal);
    promotion influence is gated by 1:1 contest-compliant empirical
    hardware per CLAUDE.md non-negotiables. The dual-tier architecture
    makes this distinction structural rather than tribal-knowledge.

    Cite-chain:

    - CATHEDRAL-SMARTER-DESIGN-MEMO §Dim 6 (cathedral_autopilot_smarter_design_blueprint_20260520T130325Z.md)
    - Catalog #341 (Tier A canonical-routing-markers gate)
    - Catalog #357 (Tier B canonical-contract STRICT gate)
    - Catalog #356 (Tier B per-axis decomposition + Provenance prerequisite)
    - Catalog #323 (canonical Provenance umbrella)
    - CLAUDE.md "Apples-to-apples evidence discipline"
    """

    TIER_A_OBSERVABILITY_ONLY = 1
    TIER_B_SCORE_CONTRIBUTING = 2


# Backward-compatible default for any consumer that omits CONSUMER_TIER.
# Per Dim 6 Step 6.2: existing consumers (lacking the new field) default to
# TIER_A_OBSERVABILITY_ONLY which is the canonical safe choice — Catalog
# #341 semantics are preserved without any migration burden.
DEFAULT_CONSUMER_TIER: ConsumerTier = ConsumerTier.TIER_A_OBSERVABILITY_ONLY


# Canonical axis tokens that a Tier B consumer MUST NOT carry (Tier B
# requires a contest or diagnostic axis; ``[predicted]`` and pure-advisory
# tags would re-collapse Tier B into Tier A semantics).
# Per Catalog #323 canonical Provenance umbrella: ``[predicted]`` is the
# canonical observability-only axis; Tier B by definition contributes to
# ranking with empirically-grounded provenance, NOT prediction.
_TIER_B_FORBIDDEN_AXIS_TOKENS: frozenset[str] = frozenset(
    {
        "[predicted]",
        "[advisory only]",
        "[macos-cpu advisory]",
        "[mps-proxy]",
        "[mps-research-signal]",
    }
)


@dataclass(frozen=True)
class AxisDecomposition:
    """Per-axis prediction decomposition for cathedral consumer contributions.

    Per CATHEDRAL-SMARTER-DESIGN-MEMO Dimension 3 Step 3.1 (operator
    blanket approval 2026-05-20) + CLAUDE.md "SegNet vs PoseNet importance
    — operating-point dependent" (UPDATED 2026-05-04): the contest scorer
    is

        S = 100 * d_seg + sqrt(10 * d_pose) + 25 * archive_bytes / 37545489

    so per-axis sensitivities differ in magnitude AND can flip direction
    across operating points (PR97 anti-pattern empirical anchor: PR97 lost
    0.042 score by trading pose for seg unwittingly; per-axis would have
    surfaced the tradeoff).

    Consumers that have per-axis signal MAY emit an ``AxisDecomposition``
    in addition to the scalar ``predicted_delta_adjustment``; the
    cathedral autopilot ranker auto-composes the per-axis deltas via
    :func:`tac.score_composition.compose_score_from_axes` and surfaces
    per-axis breakdown in observability output.

    The decomposition is OBSERVABILITY-ONLY per Catalog #287/#323/#341:

    - ``axis_tag`` MUST be ``"[predicted]"`` (or an advisory grade) per
      Catalog #341 canonical routing markers (predicted_delta_adjustment
      stays canonical non-promotable).
    - ``canonical_provenance`` MUST be a :class:`tac.provenance.Provenance`
      dict-form per Catalog #323 canonical Provenance umbrella; consumers
      MUST construct via
      :func:`tac.provenance.builders.build_provenance_for_predicted`
      (or sister builder) and pass the result through
      :func:`tac.provenance.validator.provenance_to_dict`.

    Backward compat: consumers that don't have per-axis signal emit
    ``None`` (or omit the ``predicted_axis_decomposition`` field
    entirely); the ranker handles None correctly per Step 3.3.

    Catalog #356 STRICT preflight gate refuses any consumer that emits
    ``predicted_axis_decomposition`` without ``canonical_provenance``
    per Catalog #287 + #323.

    Args:
        predicted_d_seg_delta: SegNet distortion delta (signed; negative
            = improvement; canonical units: identical to upstream
            ``score_seg`` per ``experiments/contest_auth_eval.py``).
        predicted_d_pose_delta: PoseNet distortion delta (signed; negative
            = improvement; canonical units: identical to upstream
            ``score_pose`` per ``experiments/contest_auth_eval.py``).
        predicted_archive_bytes_delta: archive size delta in BYTES
            (signed; negative = smaller); the canonical rate term is
            ``25 * archive_bytes / 37545489`` so a -200 B delta yields
            score contribution ``25 * (-200) / 37545489 ≈ -1.33e-4``.
        axis_tag: canonical axis token per Catalog #287/#341. Defaults to
            ``"[predicted]"``. Promotable axes (``[contest-CPU]`` /
            ``[contest-CUDA]``) require paired Linux x86_64 + NVIDIA
            evidence per CLAUDE.md "Submission auth eval — BOTH CPU AND
            CUDA" non-negotiable; cathedral consumer contributions are
            never promotable by construction.
        canonical_provenance: dict-form Provenance payload per Catalog
            #323; produced by ``provenance_to_dict(prov)`` where ``prov``
            comes from ``build_provenance_for_predicted(...)`` or sister
            builder. Empty dict OR missing field rejected by Catalog #356.
    """

    predicted_d_seg_delta: float
    predicted_d_pose_delta: float
    predicted_archive_bytes_delta: int  # signed
    axis_tag: str = "[predicted]"
    canonical_provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Type coercion + invariants. Frozen dataclass requires
        # object.__setattr__ for in-place mutation.
        for fname in (
            "predicted_d_seg_delta",
            "predicted_d_pose_delta",
        ):
            value = getattr(self, fname)
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"AxisDecomposition.{fname} must be numeric, got "
                    f"{type(value).__name__}"
                )
            if math.isnan(value):
                raise ValueError(
                    f"AxisDecomposition.{fname} is NaN; consumers must "
                    "emit finite per-axis deltas (use 0.0 for no-signal)"
                )
            if math.isinf(value):
                raise ValueError(
                    f"AxisDecomposition.{fname} is infinite; consumers "
                    "must emit finite per-axis deltas"
                )
            object.__setattr__(self, fname, float(value))
        bytes_value = self.predicted_archive_bytes_delta
        if not isinstance(bytes_value, int) or isinstance(bytes_value, bool):
            raise ValueError(
                "AxisDecomposition.predicted_archive_bytes_delta must be "
                f"int (signed), got {type(bytes_value).__name__}"
            )
        if not self.axis_tag or not isinstance(self.axis_tag, str):
            raise ValueError(
                "AxisDecomposition.axis_tag must be a non-empty string "
                "(canonical: '[predicted]' per Catalog #287/#341)"
            )
        # canonical_provenance shape validation deferred to Catalog #356
        # STRICT preflight gate; here we enforce only that the field is
        # a Mapping (None / list / other types rejected at construction).
        if not isinstance(self.canonical_provenance, Mapping):
            raise ValueError(
                "AxisDecomposition.canonical_provenance must be a Mapping "
                f"(Catalog #323 dict-form Provenance), got "
                f"{type(self.canonical_provenance).__name__}"
            )

    def as_dict(self) -> dict[str, Any]:
        """JSON-safe serialization for consumer ``consume_candidate`` output."""
        return {
            "predicted_d_seg_delta": float(self.predicted_d_seg_delta),
            "predicted_d_pose_delta": float(self.predicted_d_pose_delta),
            "predicted_archive_bytes_delta": int(
                self.predicted_archive_bytes_delta
            ),
            "axis_tag": str(self.axis_tag),
            "canonical_provenance": dict(self.canonical_provenance),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "AxisDecomposition":
        """Deserialize from a consumer's emitted dict-form decomposition.

        Caller is responsible for validating presence of the field on the
        consumer's output; this constructor enforces shape + invariants
        per ``__post_init__``.
        """
        return cls(
            predicted_d_seg_delta=float(data["predicted_d_seg_delta"]),
            predicted_d_pose_delta=float(data["predicted_d_pose_delta"]),
            predicted_archive_bytes_delta=int(
                data["predicted_archive_bytes_delta"]
            ),
            axis_tag=str(data.get("axis_tag", "[predicted]")),
            canonical_provenance=dict(data.get("canonical_provenance") or {}),
        )


@dataclass(frozen=True)
class ConsumerRegistration:
    """Canonical registration record for an auto-ingested consumer.

    Emitted by :func:`validate_consumer_module` after successful contract
    verification. Consumed by
    ``tools/cathedral_autopilot_autonomous_loop.discover_and_register_consumers``
    to populate the ranker cascade.
    """

    consumer_name: str
    consumer_version: str  # semver-like; not strictly enforced
    consumer_hook_numbers: tuple[HookNumber, ...]
    consumer_module_path: str
    contract_compliant: bool
    waiver_rationale: str | None = None
    waiver_active: bool = False
    validation_errors: tuple[str, ...] = field(default_factory=tuple)
    # Per Dim 6 Step 6.2: defaults to TIER_A_OBSERVABILITY_ONLY for backward
    # compatibility with consumers that omit CONSUMER_TIER. Catalog #341
    # canonical-routing-markers semantics preserved by default.
    consumer_tier: ConsumerTier = ConsumerTier.TIER_A_OBSERVABILITY_ONLY

    def __post_init__(self) -> None:
        # Frozen-dataclass invariants.
        if not self.consumer_name or not isinstance(self.consumer_name, str):
            raise ValueError("consumer_name must be a non-empty string")
        if not isinstance(self.consumer_hook_numbers, tuple):
            raise ValueError("consumer_hook_numbers must be a tuple")
        for hook in self.consumer_hook_numbers:
            if not isinstance(hook, HookNumber):
                raise ValueError(
                    f"consumer_hook_numbers entries must be HookNumber, got {type(hook).__name__}"
                )
        if self.waiver_active and not self.waiver_rationale:
            raise ValueError(
                "waiver_active=True requires non-empty waiver_rationale"
            )
        if self.waiver_active and self.contract_compliant:
            raise ValueError(
                "waiver_active=True is incompatible with contract_compliant=True "
                "(waiver is the explicit non-compliance escape)"
            )
        if not isinstance(self.consumer_tier, ConsumerTier):
            raise ValueError(
                f"consumer_tier must be ConsumerTier, got "
                f"{type(self.consumer_tier).__name__}"
            )


@runtime_checkable
class CathedralConsumerContract(Protocol):
    """Canonical Protocol every auto-ingested cathedral consumer must satisfy.

    Per Catalog #265 canonical contract pattern (sister of
    ``tac.atom.linguistic_extensions`` + ``tac.council_continual_learning``).

    The 3 module-level fields plus 2 callable surfaces are the minimum the
    auto-discovery loop needs to register + invoke the consumer.

    Consumers MAY expose additional attributes; the contract is structurally
    minimal so future hooks can extend without breaking the canonical surface.
    """

    CONSUMER_NAME: str
    CONSUMER_VERSION: str
    CONSUMER_HOOK_NUMBERS: tuple[HookNumber, ...]
    # OPTIONAL ``CONSUMER_TIER`` per Dim 6 Step 6.2 — INTENTIONALLY NOT a
    # Protocol-required attribute so that ``runtime_checkable``
    # ``isinstance`` checks remain backward-compatible for the existing
    # 44+ production consumers that predate the dual-tier landing.
    # Consumers that omit ``CONSUMER_TIER`` default to
    # :data:`DEFAULT_CONSUMER_TIER` (``ConsumerTier.TIER_A_OBSERVABILITY_ONLY``)
    # via :func:`validate_consumer_module`. Tier B consumers MUST declare
    # ``CONSUMER_TIER = ConsumerTier.TIER_B_SCORE_CONTRIBUTING`` at module
    # level + satisfy the Catalog #357 STRICT preflight contract.

    # OPTIONAL ``CONSUMES_MASTER_GRADIENT_ANCHORS`` per WAVE-3-AUTO-TRIGGER-
    # RUNTIME-WIRE-IN (2026-05-20) — INTENTIONALLY NOT a Protocol-required
    # attribute so ``runtime_checkable`` ``isinstance`` checks stay
    # backward-compatible. Consumers that opt-in by setting the module-level
    # ``CONSUMES_MASTER_GRADIENT_ANCHORS = True`` receive every new
    # master-gradient anchor via their ``update_from_anchor(anchor_row)``
    # hook (fired by :func:`tac.master_gradient.append_anchor_locked` after
    # the fcntl-locked append succeeds). Default-False per
    # ``getattr(mod, "CONSUMES_MASTER_GRADIENT_ANCHORS", False)`` lookup;
    # non-opt-in consumers are skipped. Sister of Catalog #343 pattern at
    # :func:`tac.canonical_frontier_pointer.auto_refresh_canonical_frontier_after_dispatch_outcome`.
    # Canonical opt-in fixture:
    # :mod:`tac.cathedral_consumers.auto_trigger_similarity_after_master_gradient_anchor_consumer`.

    @staticmethod
    def update_from_anchor(anchor: Any) -> None:
        """Continual-learning posterior hook (Catalog #125 hook #5).

        Called when a new empirical anchor (contest-CUDA / contest-CPU /
        diagnostic) is appended to the canonical posterior store. Consumer
        updates its internal state (e.g. refits a SLIM ranker, updates a
        Rashomon ensemble member, recomputes a sensitivity prior).

        Per CLAUDE.md "Apples-to-apples evidence discipline": the anchor's
        evidence_grade / axis_tag / hardware_substrate must be honored;
        consumers may not silently promote diagnostic to contest-grade.
        """
        ...

    @staticmethod
    def consume_candidate(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
        """Cathedral autopilot dispatch hook (Catalog #125 hook #4).

        Called per-candidate by the autopilot ranker cascade. Returns the
        consumer's contribution to ranking as a dict with at minimum:

        - ``predicted_delta_adjustment``: float (additive to candidate's
          predicted score delta; bounded; non-NaN)
        - ``rationale``: str (≥4 chars; human-readable why)
        - ``axis_tag``: str (one of ``[contest-CPU]`` / ``[contest-CUDA]`` /
          ``[diagnostic-CPU]`` / ``[diagnostic-CUDA]`` / ``[predicted]`` /
          ``[advisory only]`` per CLAUDE.md "Apples-to-apples")

        Optional fields the autopilot loop honors:
        - ``promotable``: bool (default False; True only with paired-axis evidence)
        - ``provenance``: dict (Catalog #323 canonical Provenance payload)
        - ``confidence``: float in [0, 1]
        - ``predicted_axis_decomposition``: dict-form of
          :class:`AxisDecomposition` (Catalog #356 + CATHEDRAL-SMARTER-
          DESIGN-MEMO Dimension 3 Step 3.1; per-axis (seg, pose, archive
          bytes) prediction; auto-composed via
          :func:`tac.score_composition.compose_score_from_axes`); MUST
          carry ``canonical_provenance`` per Catalog #323 when present.
        """
        ...


class CathedralConsumerContractError(Exception):
    """Raised when a package fails to satisfy the canonical contract.

    Includes explicit field-by-field rationale so the operator can
    distinguish missing CONSUMER_NAME from wrong-type CONSUMER_HOOK_NUMBERS
    from missing update_from_anchor without re-reading the source.
    """


# Same-line waiver detection regex. Mirrors Catalog #287 / #303 / #305 pattern:
# the gate's docstring example token (``CATHEDRAL_CONSUMER_DEFERRED_OK``) cannot
# self-waive because placeholder rationales are rejected.
_WAIVER_PATTERN = re.compile(
    r"#\s*" + re.escape(WAIVER_TOKEN) + r":\s*(?P<rationale>[^\n]+)"
)


def discover_waiver_in_init(init_path: Path) -> tuple[str | None, bool]:
    """Read first 30 lines of ``__init__.py`` looking for canonical waiver.

    Returns ``(rationale, active)`` where:
    - ``rationale`` is the raw rationale string (or None if no waiver line)
    - ``active`` is True only if rationale is well-formed (non-placeholder,
      ≥4 chars, non-empty after strip)

    Per Catalog #287 sister discipline: placeholder rationales
    (``<rationale>`` / ``<reason>`` / empty / <4 chars) reject the waiver so
    the gate's docstring example cannot self-waive.
    """
    if not init_path.exists() or not init_path.is_file():
        return (None, False)
    try:
        with init_path.open("r", encoding="utf-8", errors="replace") as fh:
            lines = []
            for i, line in enumerate(fh):
                if i >= _WAIVER_LOOKBACK_LINES:
                    break
                lines.append(line)
    except OSError:
        return (None, False)
    text = "".join(lines)
    match = _WAIVER_PATTERN.search(text)
    if match is None:
        return (None, False)
    raw_rationale = match.group("rationale").strip()
    # Strip trailing comment characters / quotes / etc.
    rationale = raw_rationale.rstrip("'\"`*")
    if not rationale:
        return (raw_rationale, False)
    if rationale.lower() in _PLACEHOLDER_RATIONALES:
        return (raw_rationale, False)
    if len(rationale) < _MIN_RATIONALE_LEN:
        return (raw_rationale, False)
    return (rationale, True)


def _validate_field(
    module: Any,
    field_name: str,
    expected_type: type | tuple[type, ...],
    errors: list[str],
) -> bool:
    """Helper: check a module-level field exists + is correctly typed."""
    if not hasattr(module, field_name):
        errors.append(f"missing module-level field: {field_name}")
        return False
    value = getattr(module, field_name)
    if not isinstance(value, expected_type):
        type_names = (
            expected_type.__name__
            if isinstance(expected_type, type)
            else "/".join(t.__name__ for t in expected_type)
        )
        errors.append(
            f"{field_name} must be {type_names}, got {type(value).__name__}"
        )
        return False
    return True


def validate_consumer_module(
    module: Any, *, module_path: str | None = None
) -> ConsumerRegistration:
    """Verify a module implements :class:`CathedralConsumerContract`.

    Returns :class:`ConsumerRegistration` with ``contract_compliant=True``
    on success.

    Returns :class:`ConsumerRegistration` with ``contract_compliant=False``
    and populated ``validation_errors`` on failure (does NOT raise; the
    caller decides whether to refuse or apply a waiver per the auto-discovery
    loop's strict-mode semantics).

    The single hard-raise case is when ``module`` is None or not a Python
    module object (programming error, not contract failure).
    """
    if module is None:
        raise CathedralConsumerContractError("module is None")

    errors: list[str] = []
    resolved_path = module_path or getattr(module, "__name__", "<unknown>")

    # Field validation.
    name_ok = _validate_field(module, "CONSUMER_NAME", str, errors)
    version_ok = _validate_field(module, "CONSUMER_VERSION", str, errors)
    hooks_ok = _validate_field(
        module, "CONSUMER_HOOK_NUMBERS", tuple, errors
    )

    # Hook number element validation.
    hook_numbers: tuple[HookNumber, ...] = ()
    if hooks_ok:
        raw_hooks = getattr(module, "CONSUMER_HOOK_NUMBERS")
        validated_hooks: list[HookNumber] = []
        for i, hook in enumerate(raw_hooks):
            if not isinstance(hook, HookNumber):
                errors.append(
                    f"CONSUMER_HOOK_NUMBERS[{i}] must be HookNumber, "
                    f"got {type(hook).__name__}"
                )
            else:
                validated_hooks.append(hook)
        if not validated_hooks and not errors:
            errors.append("CONSUMER_HOOK_NUMBERS must not be empty")
        hook_numbers = tuple(validated_hooks)

    # Callable surface validation.
    for callable_name in ("update_from_anchor", "consume_candidate"):
        if not hasattr(module, callable_name):
            errors.append(f"missing callable: {callable_name}")
            continue
        attr = getattr(module, callable_name)
        if not callable(attr):
            errors.append(
                f"{callable_name} must be callable, got {type(attr).__name__}"
            )

    consumer_name = (
        getattr(module, "CONSUMER_NAME", "") if name_ok else ""
    )
    consumer_version = (
        getattr(module, "CONSUMER_VERSION", "") if version_ok else "unknown"
    )

    # Per Dim 6 Step 6.2: CONSUMER_TIER is OPTIONAL. Missing → backward-compat
    # default (TIER_A_OBSERVABILITY_ONLY). Present-but-wrong-type → validation
    # error (caller decides whether to refuse).
    consumer_tier: ConsumerTier = DEFAULT_CONSUMER_TIER
    if hasattr(module, "CONSUMER_TIER"):
        raw_tier = getattr(module, "CONSUMER_TIER")
        if isinstance(raw_tier, ConsumerTier):
            consumer_tier = raw_tier
        else:
            errors.append(
                f"CONSUMER_TIER must be ConsumerTier, got "
                f"{type(raw_tier).__name__}"
            )

    return ConsumerRegistration(
        consumer_name=consumer_name or "<invalid>",
        consumer_version=consumer_version or "unknown",
        consumer_hook_numbers=hook_numbers,
        consumer_module_path=resolved_path,
        contract_compliant=(not errors),
        waiver_rationale=None,
        waiver_active=False,
        validation_errors=tuple(errors),
        consumer_tier=consumer_tier,
    )


# ---------------------------------------------------------------------------
# Tier B canonical contract validators (Dim 6 Step 6.3; Catalog #357)
# ---------------------------------------------------------------------------


def is_tier_b_axis_tag_valid(axis_tag: Any) -> bool:
    """Return True if ``axis_tag`` is acceptable for a Tier B contribution.

    Tier B contributions REQUIRE empirically-grounded axis tokens (canonical
    contest or diagnostic axis). The forbidden set (``[predicted]``,
    ``[advisory only]``, ``[mps-*]``, ``[macos-cpu advisory]``) re-collapses
    Tier B into Tier A semantics — if the prediction is purely speculative,
    the consumer should remain Tier A.

    Empty / non-string / placeholder axis_tag rejected.
    """
    if not isinstance(axis_tag, str):
        return False
    stripped = axis_tag.strip()
    if not stripped:
        return False
    if stripped.lower() in _TIER_B_FORBIDDEN_AXIS_TOKENS:
        return False
    return True


def validate_tier_b_contribution(
    contribution: Mapping[str, Any],
) -> tuple[bool, tuple[str, ...]]:
    """Validate that a Tier B consumer's ``consume_candidate`` return dict
    satisfies the Catalog #357 canonical contract.

    Returns ``(is_valid, error_messages)`` — ``is_valid`` is True iff zero
    errors. Caller (auto-discovery loop, ranker, gate) decides refusal
    semantics.

    Tier B canonical requirements (per Dim 6 Step 6.3):

    1. ``predicted_delta_adjustment`` must be a finite float (NaN / inf
       rejected; bool rejected because Python `bool` is subclass of int but
       semantically not a score-delta).
    2. ``axis_tag`` must be a non-forbidden empirically-grounded axis token
       (NOT ``[predicted]``; the score signal must be grounded).
    3. ``promotable=False`` must be present and False (Tier B contributes
       to RANKING but NEVER to PROMOTION per CLAUDE.md "Submission auth
       eval — BOTH CPU AND CUDA").
    4. ``provenance`` must be present and be a non-empty Mapping (Catalog
       #323 canonical Provenance umbrella; payload shape validated by
       :func:`tac.provenance.validate_provenance` downstream).
    5. ``rationale`` must be a non-placeholder string ≥4 chars.

    These invariants are CHECKED HERE but ENFORCED structurally at landing
    time by Catalog #357 STRICT preflight gate (scans consumer source code
    for hand-written Tier B contributions that omit any required field).

    Per CLAUDE.md "Bugs must be permanently fixed AND self-protected
    against" non-negotiable: the runtime validator (here) + the static
    preflight gate (Catalog #357) close the Tier B canonical-contract bug
    class at BOTH the per-call and per-source surfaces.
    """
    if not isinstance(contribution, Mapping):
        return False, (
            f"contribution must be a Mapping, got {type(contribution).__name__}",
        )

    errors: list[str] = []

    # (1) predicted_delta_adjustment: finite float
    if "predicted_delta_adjustment" not in contribution:
        errors.append("missing required field: predicted_delta_adjustment")
    else:
        delta = contribution["predicted_delta_adjustment"]
        if isinstance(delta, bool) or not isinstance(delta, (int, float)):
            errors.append(
                "predicted_delta_adjustment must be a finite float, got "
                f"{type(delta).__name__}"
            )
        else:
            d = float(delta)
            if math.isnan(d):
                errors.append("predicted_delta_adjustment is NaN")
            elif math.isinf(d):
                errors.append("predicted_delta_adjustment is infinite")

    # (2) axis_tag: empirically-grounded (not [predicted] / [advisory])
    axis_tag = contribution.get("axis_tag")
    if not is_tier_b_axis_tag_valid(axis_tag):
        errors.append(
            "Tier B axis_tag must be empirically-grounded (NOT '[predicted]' / "
            "'[advisory only]' / '[mps-*]'); got "
            f"{axis_tag!r}. Tier B contributions ground per-row score "
            "signals in canonical Provenance per Catalog #323; if the "
            "signal is speculative, the consumer should remain Tier A."
        )

    # (3) promotable=False (preserved; promotion requires paired-axis evidence)
    if "promotable" not in contribution:
        errors.append(
            "missing required field: promotable (Tier B contributions MUST "
            "carry promotable=False; promotion requires paired contest-CPU "
            "+ contest-CUDA evidence per CLAUDE.md 'Submission auth eval')"
        )
    elif contribution["promotable"] is not False:
        errors.append(
            f"promotable must be False (got {contribution['promotable']!r}); "
            "Tier B contributes to RANKING but NEVER to PROMOTION per "
            "CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA'"
        )

    # (4) provenance: non-empty Mapping per Catalog #323
    if "provenance" not in contribution:
        errors.append(
            "missing required field: provenance (Tier B contributions MUST "
            "carry canonical Provenance per Catalog #323; produce via "
            "tac.provenance.builders.build_provenance_for_predicted then "
            "tac.provenance.validator.provenance_to_dict)"
        )
    else:
        prov = contribution["provenance"]
        if not isinstance(prov, Mapping):
            errors.append(
                f"provenance must be a Mapping, got {type(prov).__name__}"
            )
        elif not prov:
            errors.append(
                "provenance must be a non-empty Mapping (Catalog #323 "
                "canonical Provenance payload required)"
            )

    # (5) rationale: non-placeholder string ≥4 chars
    rationale = contribution.get("rationale")
    if not isinstance(rationale, str):
        errors.append(
            f"rationale must be a string, got {type(rationale).__name__}"
        )
    else:
        stripped_rationale = rationale.strip()
        if not stripped_rationale:
            errors.append("rationale must be non-empty (≥4 chars)")
        elif len(stripped_rationale) < _MIN_RATIONALE_LEN:
            errors.append(
                f"rationale too short (got {len(stripped_rationale)} chars; "
                f"min {_MIN_RATIONALE_LEN})"
            )
        elif stripped_rationale.lower() in _PLACEHOLDER_RATIONALES:
            errors.append(
                f"rationale is placeholder literal ({stripped_rationale!r}); "
                "non-placeholder rationale required per Catalog #287 sister "
                "discipline"
            )

    return (not errors), tuple(errors)
