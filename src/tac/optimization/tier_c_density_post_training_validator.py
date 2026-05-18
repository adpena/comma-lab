# SPDX-License-Identifier: MIT
"""tac.optimization.tier_c_density_post_training_validator — Catalog #324 canonical helper.

Per operator NON-NEGOTIABLE 2026-05-17 (META-FIX Catalog #324): every
``predicted_band`` emission must be paired with an attestation that the
Tier-C density evidence it was derived from was measured on a
POST-TRAINING architecture artifact, not random-init weights.

Empirical bug class anchor — the C6 IBPS 22× miss (2026-05-17)
=================================================================

C6 IBPS 50ep Modal A10G smoke (call_id ``fc-01KRW353MJJ9A6QW8H99QWZEMH``)
returned ``final_score = 3.04`` against a recipe-declared ``predicted_band:
[0.113, 0.163]`` derived from a Tier-C ACROSS_CLASS density of ``2.67e-5``.
The actual smoke landed ``18-22×`` outside the predicted upper bound
because the Tier-C density was measured on RANDOM-INIT weights via
``tools/mdl_scorer_conditional_ablation.py`` BEFORE any training had run,
NOT on the post-training architecture artifact the contest scorer actually
sees at inflate time.

Sister #835's recipe-fix sextet Assumption-Adversary had verbatim WARNED:

  *"$0.76 paired CPU+CUDA dispatch is sufficient to confirm band
   empirically" → CARGO-CULTED; "provides only UPPER bound on
   disconfirmation; operator must accept asymmetry"*

  *"Tier-C ACROSS_CLASS density 2.67e-5 predicts post-training class
   shift" → CARGO-CULTED; "Tier-C computed on RANDOM INIT archive
   (pre-training). Post-training Tier-C may differ."*

Sister #836 EMPIRICALLY FALSIFIED the prediction (the actual smoke
returned 3.04 vs target 0.138 — a 22× miss). The R1 council
Assumption-Adversary verdict was HARD-EARNED-EMPIRICALLY-VERIFIED.

This module operationalizes the protection structurally: every
``predicted_band`` must carry a ``TierCDensityWithProvenance`` attestation
that classifies the density's source as ``POST_TRAINING_50EP_SMOKE`` /
``POST_TRAINING_200EP_FULL`` / ``RANDOM_INIT_PRE_TRAINING`` /
``UNKNOWN_PROVENANCE`` / ``OPERATOR_WAIVED``. The ``RANDOM_INIT_*``
classification flips the band's ``validation_status`` to
``phantom_random_init`` and the consuming Catalog #324 STRICT preflight
gate refuses the recipe unless it carries ``research_only: true`` OR
explicit operator waiver with rationale.

Sister of:

  * Catalog #321 ``check_no_phantom_wyner_ziv_savings_from_research_sidecar``
    (same META-class at the WZ-deliverability surface; this module is the
    predicted-band surface).
  * Catalog #322 ``check_no_autopilot_adjustment_derived_from_phantom_provenance_composition_alpha``
    (same META-class at the autopilot composition_alpha surface).
  * Catalog #323 ``check_no_score_claim_without_canonical_provenance``
    (the META-meta umbrella; this module extends the Provenance contract
    to the predicted_band sub-surface).
  * ``tac.provenance.Provenance`` (canonical attestation dataclass; this
    module's ``TierCDensityWithProvenance`` composes-with rather than
    replaces).
  * ``tac.wyner_ziv_deliverability.proof_builder.DeliverabilityProof``
    (canonical pattern this module mirrors at the Tier-C-density surface).

Per CLAUDE.md "Forbidden premature KILL without research exhaustion": a
``RANDOM_INIT_PRE_TRAINING`` classification is NOT a kill — it is a
"reactivation requires post-training Tier-C measurement" defer. The
canonical reactivation path is documented per recipe.

Per CLAUDE.md "Apples-to-apples evidence discipline": every Tier-C
density anchor must declare ``measured_at_utc`` + ``archive_sha256`` +
``canonical_helper_invocation`` so a future audit can verify the
density was actually measured on the named archive.

Per CLAUDE.md "Beauty, simplicity, and developer experience": narrow
public API — 2 frozen dataclasses + 3 builder helpers + 1 validator
function. The Catalog #324 STRICT gate body lives in
``src/tac/preflight.py``; this module is the structural primitive.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Tuple

# Schema version pinned in module-level constant.
TIER_C_POST_TRAINING_SCHEMA_VERSION: str = "tier_c_post_training_v1_20260517"


class TierCDensitySource(Enum):
    """Canonical Tier-C density measurement-source taxonomy.

    The taxonomy distinguishes the 4 empirically distinct cases that
    produce different epistemic confidence in a derived predicted_band:

    - ``RANDOM_INIT_PRE_TRAINING``: density measured before any training
      step. Empirically falsified at 22× by C6 IBPS smoke 2026-05-17.
      Triggers ``phantom_random_init`` validation status.
    - ``POST_TRAINING_50EP_SMOKE``: density measured on a 50ep smoke
      archive. Validated for trend-direction (within ~5x band).
    - ``POST_TRAINING_200EP_FULL``: density measured on a converged
      200ep full-run archive. Validated for absolute-band claims.
    - ``UNKNOWN_PROVENANCE``: density source not attested. REFUSED.
    - ``OPERATOR_WAIVED``: explicit operator override with rationale.
    """

    RANDOM_INIT_PRE_TRAINING = "random_init_pre_training"
    POST_TRAINING_50EP_SMOKE = "post_training_50ep_smoke"
    POST_TRAINING_200EP_FULL = "post_training_200ep_full"
    UNKNOWN_PROVENANCE = "unknown_provenance"
    OPERATOR_WAIVED = "operator_waived"


# Source -> required-post-training-revalidation map. Used by
# PredictedBandWithValidation.__post_init__ to derive the validation_status.
_SOURCE_REQUIRES_REVALIDATION: dict[TierCDensitySource, bool] = {
    TierCDensitySource.RANDOM_INIT_PRE_TRAINING: True,
    TierCDensitySource.POST_TRAINING_50EP_SMOKE: False,
    TierCDensitySource.POST_TRAINING_200EP_FULL: False,
    TierCDensitySource.UNKNOWN_PROVENANCE: True,  # refused at __post_init__
    TierCDensitySource.OPERATOR_WAIVED: False,  # operator attested
}


# Canonical validation_status taxonomy for the predicted_band field. These
# are the strings recipe YAML readers see in the ``predicted_band_validation_status``
# field. Mirror the canonical set declared in CLAUDE.md "Catalog #324" row.
VALIDATION_STATUS_VALIDATED_POST_TRAINING: str = "validated_post_training"
VALIDATION_STATUS_PENDING_POST_TRAINING: str = "pending_post_training"
VALIDATION_STATUS_PHANTOM_RANDOM_INIT: str = "phantom_random_init"
VALIDATION_STATUS_OPERATOR_WAIVED: str = "operator_waived"

# Canonical waiver token recognized by the recipe-audit + STRICT gate.
PREDICTED_BAND_RANDOM_INIT_WAIVER_TOKEN: str = "PREDICTED_BAND_RANDOM_INIT_OK"

# Forbidden placeholder rationales — sister of Catalog #229 / #287 / #321 /
# #323 patterns. The gate's own docstring example cannot self-waive.
_PLACEHOLDER_RATIONALES: frozenset[str] = frozenset({
    "<rationale>",
    "<reason>",
    "rationale>",
    "reason>",
    "TBD",
    "tbd",
    "placeholder",
    "PLACEHOLDER",
})


class PredictedBandValidationError(ValueError):
    """Raised when a PredictedBandWithValidation construction violates invariants.

    The error message includes ``blockers`` (list of canonical refusal
    reasons) so callers can surface actionable diagnostics in council
    deliberations + audit reports.
    """

    def __init__(self, msg: str, blockers: Optional[list[str]] = None) -> None:
        super().__init__(msg)
        self.blockers: list[str] = list(blockers or [])


@dataclass(frozen=True)
class TierCDensityWithProvenance:
    """Canonical attestation for a Tier-C density measurement.

    Composes with ``tac.provenance.Provenance`` at the validator surface
    via ``to_provenance_dict()``. Frozen per CLAUDE.md HISTORICAL_PROVENANCE
    Catalog #110/#113 — once an attestation is constructed, it cannot be
    silently mutated by a downstream consumer.

    Invariants enforced at ``__post_init__``:

    - ``density_value`` must be a finite non-negative float
    - ``source`` must be a valid ``TierCDensitySource`` enum member
    - ``measured_at_utc`` must be a non-empty ISO-8601 UTC timestamp
    - ``archive_sha256`` must be a 64-char lowercase hex string or
      the sentinel ``"random_init_no_archive"`` for RANDOM_INIT_PRE_TRAINING
    - ``source == OPERATOR_WAIVED`` requires ``rationale`` (non-placeholder)
    - ``source == UNKNOWN_PROVENANCE`` is REFUSED (raises immediately)
    - ``epochs_trained`` must match the source classification
      (RANDOM_INIT requires 0; POST_TRAINING_50EP requires >= 1;
      POST_TRAINING_200EP requires >= 50)
    """

    density_value: float
    source: TierCDensitySource
    measured_at_utc: str
    archive_sha256: str
    rationale: str = ""
    epochs_trained: int = 0
    canonical_helper_invocation: str = ""

    # Pinned schema version so external readers can verify compatibility.
    schema_version: str = TIER_C_POST_TRAINING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        blockers: list[str] = []

        # Density must be finite, non-negative.
        try:
            dv = float(self.density_value)
        except (TypeError, ValueError):
            blockers.append(f"density_value must be a finite float; got {self.density_value!r}")
            dv = float("nan")
        else:
            import math
            if math.isnan(dv) or math.isinf(dv):
                blockers.append(f"density_value must be finite; got {dv}")
            elif dv < 0:
                blockers.append(f"density_value must be non-negative; got {dv}")

        # Source must be enum member.
        if not isinstance(self.source, TierCDensitySource):
            blockers.append(f"source must be TierCDensitySource enum member; got {self.source!r}")

        # measured_at_utc must be non-empty ISO timestamp.
        if not isinstance(self.measured_at_utc, str) or not self.measured_at_utc.strip():
            blockers.append("measured_at_utc must be a non-empty ISO-8601 UTC timestamp")
        elif not re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", self.measured_at_utc):
            blockers.append(
                f"measured_at_utc must be ISO-8601 format YYYY-MM-DDTHH:MM:SS...; got {self.measured_at_utc!r}"
            )

        # archive_sha256 must be 64-char hex OR the sentinel for random-init.
        if not isinstance(self.archive_sha256, str) or not self.archive_sha256.strip():
            blockers.append("archive_sha256 must be non-empty")
        elif self.archive_sha256 == "random_init_no_archive":
            if self.source != TierCDensitySource.RANDOM_INIT_PRE_TRAINING:
                blockers.append(
                    "archive_sha256='random_init_no_archive' sentinel only valid for "
                    f"source=RANDOM_INIT_PRE_TRAINING; got source={self.source!r}"
                )
        elif not re.match(r"^[0-9a-f]{64}$", self.archive_sha256):
            blockers.append(
                f"archive_sha256 must be 64-char lowercase hex OR 'random_init_no_archive' sentinel; "
                f"got {self.archive_sha256!r}"
            )

        # UNKNOWN_PROVENANCE source REFUSED.
        if self.source == TierCDensitySource.UNKNOWN_PROVENANCE:
            blockers.append(
                "TierCDensityWithProvenance with source=UNKNOWN_PROVENANCE REFUSED "
                "per Catalog #324; predicted_band derived from unknown-provenance Tier-C "
                "density is structurally indistinguishable from random-init (the C6 IBPS "
                "22× miss bug class anchor 2026-05-17)"
            )

        # OPERATOR_WAIVED requires non-placeholder rationale.
        if self.source == TierCDensitySource.OPERATOR_WAIVED:
            if not self.rationale or not self.rationale.strip():
                blockers.append("source=OPERATOR_WAIVED requires non-empty rationale")
            elif self.rationale.strip() in _PLACEHOLDER_RATIONALES:
                blockers.append(
                    f"source=OPERATOR_WAIVED rationale must be non-placeholder; "
                    f"got literal {self.rationale!r} (Catalog #229 placeholder rejection)"
                )
            elif len(self.rationale.strip()) < 8:
                blockers.append(
                    f"source=OPERATOR_WAIVED rationale must be substantive (>=8 chars); "
                    f"got {self.rationale!r}"
                )

        # epochs_trained consistency with source classification.
        if isinstance(self.source, TierCDensitySource):
            if self.source == TierCDensitySource.RANDOM_INIT_PRE_TRAINING:
                if self.epochs_trained != 0:
                    blockers.append(
                        f"source=RANDOM_INIT_PRE_TRAINING requires epochs_trained=0; "
                        f"got {self.epochs_trained}"
                    )
            elif self.source == TierCDensitySource.POST_TRAINING_50EP_SMOKE:
                if self.epochs_trained < 1:
                    blockers.append(
                        f"source=POST_TRAINING_50EP_SMOKE requires epochs_trained>=1; "
                        f"got {self.epochs_trained}"
                    )
            elif self.source == TierCDensitySource.POST_TRAINING_200EP_FULL:
                if self.epochs_trained < 50:
                    blockers.append(
                        f"source=POST_TRAINING_200EP_FULL requires epochs_trained>=50; "
                        f"got {self.epochs_trained}"
                    )

        if blockers:
            raise PredictedBandValidationError(
                "TierCDensityWithProvenance construction failed: " + "; ".join(blockers),
                blockers=blockers,
            )

    def requires_post_training_revalidation(self) -> bool:
        """Returns True iff the derived predicted_band needs post-training validation."""
        return _SOURCE_REQUIRES_REVALIDATION.get(self.source, True)

    def to_provenance_dict(self) -> dict[str, Any]:
        """Render as a tac.provenance.Provenance-compatible dict.

        Bridges to the canonical Catalog #323 contract so downstream
        consumers (audit tools, autopilot ranker) can read Tier-C
        attestations via the same shape as other Provenance instances.
        """
        return {
            "tier_c_density_value": self.density_value,
            "tier_c_density_source": self.source.value,
            "tier_c_density_measured_at_utc": self.measured_at_utc,
            "tier_c_density_archive_sha256": self.archive_sha256,
            "tier_c_density_epochs_trained": self.epochs_trained,
            "tier_c_density_canonical_helper_invocation": self.canonical_helper_invocation,
            "tier_c_density_rationale": self.rationale,
            "tier_c_density_schema_version": self.schema_version,
        }


@dataclass(frozen=True)
class PredictedBandWithValidation:
    """Canonical predicted_band with mandatory Tier-C-source attestation.

    Used by recipes + design memos + autopilot consumer surfaces. The
    validation_status is AUTO-DERIVED from the underlying Tier-C density
    source so callers cannot independently claim ``validated_post_training``
    when the density is in fact ``RANDOM_INIT_PRE_TRAINING``.

    Invariants enforced at ``__post_init__``:

    - ``band_low <= target <= band_high``
    - All three (band_low, target, band_high) must be finite floats
    - ``derived_from_tier_c_density.source == RANDOM_INIT_PRE_TRAINING``
      forces ``validation_status == phantom_random_init``
    - ``derived_from_tier_c_density.source == UNKNOWN_PROVENANCE`` is
      REFUSED at the density layer (raised in TierCDensityWithProvenance)
    - ``requires_post_training_revalidation`` is auto-derived from source
    """

    band_low: float
    band_high: float
    target: float
    derived_from_tier_c_density: TierCDensityWithProvenance
    schema_version: str = TIER_C_POST_TRAINING_SCHEMA_VERSION

    def __post_init__(self) -> None:
        blockers: list[str] = []

        # Finite floats.
        import math
        for name, value in (("band_low", self.band_low), ("target", self.target), ("band_high", self.band_high)):
            try:
                fv = float(value)
            except (TypeError, ValueError):
                blockers.append(f"{name} must be a finite float; got {value!r}")
                continue
            if math.isnan(fv) or math.isinf(fv):
                blockers.append(f"{name} must be finite; got {fv}")

        # Ordering invariant.
        if not blockers:
            if not (self.band_low <= self.target <= self.band_high):
                blockers.append(
                    f"band_low ({self.band_low}) <= target ({self.target}) <= "
                    f"band_high ({self.band_high}) invariant violated"
                )

        # derived_from must be TierCDensityWithProvenance instance.
        if not isinstance(self.derived_from_tier_c_density, TierCDensityWithProvenance):
            blockers.append(
                f"derived_from_tier_c_density must be TierCDensityWithProvenance instance; "
                f"got {type(self.derived_from_tier_c_density).__name__}"
            )

        if blockers:
            raise PredictedBandValidationError(
                "PredictedBandWithValidation construction failed: " + "; ".join(blockers),
                blockers=blockers,
            )

    @property
    def validation_status(self) -> str:
        """Auto-derived from the underlying Tier-C density source."""
        src = self.derived_from_tier_c_density.source
        if src == TierCDensitySource.RANDOM_INIT_PRE_TRAINING:
            return VALIDATION_STATUS_PHANTOM_RANDOM_INIT
        if src in (TierCDensitySource.POST_TRAINING_50EP_SMOKE, TierCDensitySource.POST_TRAINING_200EP_FULL):
            return VALIDATION_STATUS_VALIDATED_POST_TRAINING
        if src == TierCDensitySource.OPERATOR_WAIVED:
            return VALIDATION_STATUS_OPERATOR_WAIVED
        return VALIDATION_STATUS_PENDING_POST_TRAINING

    @property
    def requires_post_training_revalidation(self) -> bool:
        return self.derived_from_tier_c_density.requires_post_training_revalidation()

    def to_recipe_dict(self) -> dict[str, Any]:
        """Render as a recipe YAML-compatible dict for backfill."""
        return {
            "predicted_band": [self.band_low, self.band_high],
            "predicted_score_target": self.target,
            "predicted_band_validation_status": self.validation_status,
            "predicted_band_requires_post_training_revalidation": (
                self.requires_post_training_revalidation
            ),
            "predicted_band_tier_c_density_provenance": (
                self.derived_from_tier_c_density.to_provenance_dict()
            ),
            "predicted_band_schema_version": self.schema_version,
        }


# Canonical builder for predicted bands. ENFORCES validation status auto-derivation.
def build_predicted_band_from_tier_c_density(
    *,
    tier_c_density: TierCDensityWithProvenance,
    proposed_band_low: float,
    proposed_band_high: float,
    proposed_target: float,
) -> PredictedBandWithValidation:
    """Canonical builder for predicted bands.

    The validation_status is AUTO-DERIVED from the tier_c_density.source —
    callers cannot override. If the density source is
    ``RANDOM_INIT_PRE_TRAINING``, the returned band carries
    ``validation_status == phantom_random_init`` AND
    ``requires_post_training_revalidation == True``.

    Usage::

        density = TierCDensityWithProvenance(
            density_value=2.67e-5,
            source=TierCDensitySource.RANDOM_INIT_PRE_TRAINING,
            measured_at_utc="2026-05-17T22:00:00Z",
            archive_sha256="random_init_no_archive",
            epochs_trained=0,
            canonical_helper_invocation="tools/mdl_scorer_conditional_ablation.py --tier c (random-init)",
        )
        band = build_predicted_band_from_tier_c_density(
            tier_c_density=density,
            proposed_band_low=0.113,
            proposed_band_high=0.163,
            proposed_target=0.138,
        )
        assert band.validation_status == "phantom_random_init"
        assert band.requires_post_training_revalidation
    """
    return PredictedBandWithValidation(
        band_low=proposed_band_low,
        band_high=proposed_band_high,
        target=proposed_target,
        derived_from_tier_c_density=tier_c_density,
    )


def build_tier_c_density_post_training(
    *,
    density_value: float,
    epochs_trained: int,
    archive_sha256: str,
    measured_at_utc: str,
    canonical_helper_invocation: str = "tools/mdl_scorer_conditional_ablation.py --tier c",
) -> TierCDensityWithProvenance:
    """Single-line builder for the canonical post-training case (>= 1ep)."""
    if epochs_trained < 50:
        source = TierCDensitySource.POST_TRAINING_50EP_SMOKE
    else:
        source = TierCDensitySource.POST_TRAINING_200EP_FULL
    return TierCDensityWithProvenance(
        density_value=density_value,
        source=source,
        measured_at_utc=measured_at_utc,
        archive_sha256=archive_sha256,
        epochs_trained=epochs_trained,
        canonical_helper_invocation=canonical_helper_invocation,
    )


def build_tier_c_density_random_init(
    *,
    density_value: float,
    measured_at_utc: str,
    canonical_helper_invocation: str = "tools/mdl_scorer_conditional_ablation.py --tier c (random-init)",
) -> TierCDensityWithProvenance:
    """Single-line builder for the random-init case (forces phantom verdict)."""
    return TierCDensityWithProvenance(
        density_value=density_value,
        source=TierCDensitySource.RANDOM_INIT_PRE_TRAINING,
        measured_at_utc=measured_at_utc,
        archive_sha256="random_init_no_archive",
        epochs_trained=0,
        canonical_helper_invocation=canonical_helper_invocation,
    )


def build_tier_c_density_operator_waived(
    *,
    density_value: float,
    measured_at_utc: str,
    rationale: str,
    archive_sha256: str,
) -> TierCDensityWithProvenance:
    """Single-line builder for the operator-waived case (requires substantive rationale + real sha).

    The ``archive_sha256`` must be a 64-char hex string; the sentinel
    ``"random_init_no_archive"`` is NOT accepted for OPERATOR_WAIVED
    (the sentinel is reserved for RANDOM_INIT_PRE_TRAINING). Callers
    that want to waive without an archive should use a placeholder
    64-char-hex value documenting the waiver context (e.g. zero-sha).
    """
    if not rationale or rationale.strip() in _PLACEHOLDER_RATIONALES or len(rationale.strip()) < 8:
        raise PredictedBandValidationError(
            f"build_tier_c_density_operator_waived rationale must be non-placeholder and "
            f">=8 chars; got {rationale!r}"
        )
    return TierCDensityWithProvenance(
        density_value=density_value,
        source=TierCDensitySource.OPERATOR_WAIVED,
        measured_at_utc=measured_at_utc,
        archive_sha256=archive_sha256,
        rationale=rationale,
        epochs_trained=0,
        canonical_helper_invocation="operator-attested",
    )


# Recipe-level validator helper used by the audit tool + STRICT preflight gate.

@dataclass(frozen=True)
class RecipeAuditVerdict:
    """Verdict for a single recipe's predicted_band provenance audit."""

    recipe_path: str
    has_predicted_band: bool
    validation_status: str  # "validated_post_training" / "pending_post_training" / "phantom_random_init" / "research_only" / "operator_waived" / "absent"
    is_valid: bool
    blockers: tuple[str, ...] = ()
    is_research_only: bool = False
    waiver_rationale: Optional[str] = None
    detected_predicted_band: Optional[Tuple[float, float]] = None
    detected_target: Optional[float] = None


# Recipe research-only token detection (mirrors sister Catalog #240 pattern).
_RESEARCH_ONLY_TOKENS: tuple[str, ...] = (
    "research_only: true",
    "research_only=true",
    "research-only: true",
    "research-only=true",
    "smoke_only: true",
    "dispatch_enabled: false",
)


def _detect_research_only(recipe_text: str) -> bool:
    """Detect research_only opt-out in recipe YAML body (mirrors Catalog #240)."""
    lower = recipe_text.lower()
    return any(tok.lower() in lower for tok in _RESEARCH_ONLY_TOKENS)


def _detect_predicted_band_waiver(recipe_text: str) -> Optional[str]:
    """Detect Catalog #324 waiver token in recipe YAML.

    Returns the rationale string if a valid waiver is present, None otherwise.
    Placeholder rationales are NOT accepted (Catalog #229 pattern).
    """
    # Match same-line waivers on any line. Pattern: # PREDICTED_BAND_RANDOM_INIT_OK:<rationale>
    pattern = re.compile(
        rf"#\s*{re.escape(PREDICTED_BAND_RANDOM_INIT_WAIVER_TOKEN)}:([^\n]+)"
    )
    match = pattern.search(recipe_text)
    if not match:
        return None
    rationale = match.group(1).strip()
    if not rationale or rationale in _PLACEHOLDER_RATIONALES:
        return None
    if len(rationale) < 8:
        return None
    return rationale


def _detect_predicted_band_validation_status(recipe_text: str) -> Optional[str]:
    """Extract recipe's declared predicted_band_validation_status field, if any."""
    pattern = re.compile(
        r"predicted_band_validation_status:\s*['\"]?([a-z_]+)['\"]?",
        re.MULTILINE,
    )
    match = pattern.search(recipe_text)
    if not match:
        return None
    return match.group(1).strip()


def _detect_predicted_band_range(recipe_text: str) -> Optional[Tuple[float, float]]:
    """Extract recipe's declared predicted_band: [lo, hi] field, if any."""
    pattern = re.compile(
        r"^predicted_band:\s*\[\s*([+-]?\d+\.?\d*)\s*,\s*([+-]?\d+\.?\d*)\s*\]",
        re.MULTILINE,
    )
    match = pattern.search(recipe_text)
    if not match:
        return None
    try:
        return (float(match.group(1)), float(match.group(2)))
    except (TypeError, ValueError):
        return None


def _detect_predicted_score_target(recipe_text: str) -> Optional[float]:
    """Extract recipe's declared predicted_score_target field, if any."""
    pattern = re.compile(
        r"^predicted_score_target:\s*([+-]?\d+\.?\d*)\s*(?:#.*)?$",
        re.MULTILINE,
    )
    match = pattern.search(recipe_text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except (TypeError, ValueError):
        return None


def validate_recipe_predicted_band(recipe_path: str | Path) -> RecipeAuditVerdict:
    """Audit a recipe YAML for predicted_band provenance.

    Returns ``RecipeAuditVerdict`` with ``is_valid`` True iff the recipe
    either: (a) declares no predicted_band (out-of-scope); (b) declares
    predicted_band AND has validation_status in
    {validated_post_training, pending_post_training, operator_waived};
    (c) is research_only=true OR dispatch_enabled=false; (d) carries a
    same-line waiver with substantive rationale.

    The verdict's ``blockers`` field lists actionable refusal reasons
    when ``is_valid`` is False so the audit tool + STRICT gate can
    surface canonical fix paths.
    """
    p = Path(recipe_path)
    if not p.exists():
        return RecipeAuditVerdict(
            recipe_path=str(p),
            has_predicted_band=False,
            validation_status="absent",
            is_valid=True,
            blockers=("recipe file not found",),
        )

    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return RecipeAuditVerdict(
            recipe_path=str(p),
            has_predicted_band=False,
            validation_status="absent",
            is_valid=True,
            blockers=(f"recipe read error: {exc}",),
        )

    band_range = _detect_predicted_band_range(text)
    target = _detect_predicted_score_target(text)
    has_band = band_range is not None
    declared_status = _detect_predicted_band_validation_status(text)
    research_only = _detect_research_only(text)
    waiver = _detect_predicted_band_waiver(text)

    # Out-of-scope: no predicted_band declared at all.
    if not has_band:
        return RecipeAuditVerdict(
            recipe_path=str(p),
            has_predicted_band=False,
            validation_status="absent",
            is_valid=True,
        )

    # Research-only opt-out cascade.
    if research_only:
        return RecipeAuditVerdict(
            recipe_path=str(p),
            has_predicted_band=True,
            validation_status="research_only",
            is_valid=True,
            is_research_only=True,
            detected_predicted_band=band_range,
            detected_target=target,
        )

    # Same-line waiver cascade.
    if waiver:
        return RecipeAuditVerdict(
            recipe_path=str(p),
            has_predicted_band=True,
            validation_status=VALIDATION_STATUS_OPERATOR_WAIVED,
            is_valid=True,
            waiver_rationale=waiver,
            detected_predicted_band=band_range,
            detected_target=target,
        )

    # Declared validation status cascade.
    valid_statuses = {
        VALIDATION_STATUS_VALIDATED_POST_TRAINING,
        VALIDATION_STATUS_PENDING_POST_TRAINING,
        VALIDATION_STATUS_OPERATOR_WAIVED,
    }
    if declared_status in valid_statuses:
        return RecipeAuditVerdict(
            recipe_path=str(p),
            has_predicted_band=True,
            validation_status=declared_status,
            is_valid=True,
            detected_predicted_band=band_range,
            detected_target=target,
        )

    if declared_status == VALIDATION_STATUS_PHANTOM_RANDOM_INIT:
        return RecipeAuditVerdict(
            recipe_path=str(p),
            has_predicted_band=True,
            validation_status=VALIDATION_STATUS_PHANTOM_RANDOM_INIT,
            is_valid=False,
            blockers=(
                "predicted_band declared as 'phantom_random_init' but recipe is dispatch-enabled; "
                "must satisfy one of: (1) replace with post-training Tier-C density measurement; "
                "(2) declare research_only=true OR dispatch_enabled=false; "
                "(3) add same-line waiver `# PREDICTED_BAND_RANDOM_INIT_OK:<rationale>`",
            ),
            detected_predicted_band=band_range,
            detected_target=target,
        )

    # No validation status declared and no opt-out: REFUSED.
    return RecipeAuditVerdict(
        recipe_path=str(p),
        has_predicted_band=True,
        validation_status="missing_validation_status",
        is_valid=False,
        blockers=(
            "predicted_band declared but no predicted_band_validation_status field; "
            "per Catalog #324 (C6 IBPS 22× miss anchor 2026-05-17) every predicted_band emission "
            "MUST attest to whether the Tier-C density was measured pre-training (random-init) "
            "or post-training. Add one of: "
            f"predicted_band_validation_status: {VALIDATION_STATUS_VALIDATED_POST_TRAINING}, "
            f"predicted_band_validation_status: {VALIDATION_STATUS_PENDING_POST_TRAINING}, "
            "research_only: true, dispatch_enabled: false, "
            f"OR same-line waiver `# {PREDICTED_BAND_RANDOM_INIT_WAIVER_TOKEN}:<rationale>`",
        ),
        detected_predicted_band=band_range,
        detected_target=target,
    )


__all__ = [
    # Schema
    "TIER_C_POST_TRAINING_SCHEMA_VERSION",
    "VALIDATION_STATUS_VALIDATED_POST_TRAINING",
    "VALIDATION_STATUS_PENDING_POST_TRAINING",
    "VALIDATION_STATUS_PHANTOM_RANDOM_INIT",
    "VALIDATION_STATUS_OPERATOR_WAIVED",
    "PREDICTED_BAND_RANDOM_INIT_WAIVER_TOKEN",
    # Enum
    "TierCDensitySource",
    # Errors
    "PredictedBandValidationError",
    # Dataclasses
    "TierCDensityWithProvenance",
    "PredictedBandWithValidation",
    "RecipeAuditVerdict",
    # Builders
    "build_predicted_band_from_tier_c_density",
    "build_tier_c_density_post_training",
    "build_tier_c_density_random_init",
    "build_tier_c_density_operator_waived",
    # Validator
    "validate_recipe_predicted_band",
]
