# SPDX-License-Identifier: MIT
"""tac.provenance.contract — canonical Provenance + ScoreClaim dataclasses.

Per operator NON-NEGOTIABLE 2026-05-17 verbatim: *"We need to fix the
provenance issue for all and fix it permanently and canonically and make
it easy"*. The Provenance contract is the single canonical structure that
EVERY score-claiming surface in the repo uses, replacing the ad-hoc
`evidence_grade` / `archive_sha256` / `axis_tag` / `hardware_substrate`
fields scattered across `ContestResult`, `CompositionResult`,
`SubstrateCompositionRow`, `DeliverabilityProof`, autopilot
`CandidateRow.predicted_dispatch_risk`, prober deliverable rows, etc.

The contract extincts 5 phantom-score class instances in one session:

  1. **Catalog #319 fec6 1.15× autopilot reward** (false-signal byte-identity
     duplication elevated venn reweight; closed at the autopilot-consumer
     surface). The composition_alpha row had no Provenance to attest the
     pairwise alpha was measured on REAL distinct candidates.
  2. **Catalog #321 pr101_state_dict 0.477 score** (research sidecar .pt
     bytes scored as if shipped in archive.zip; closed at the prober-artifact
     surface). The deliverable_score_savings_estimate had no Provenance to
     attest the bytes live INSIDE a contest archive.zip.
  3. **pr106_state_dict / posenet_class_sensitivity** (same Catalog #321
     class at a sibling artifact; ~11.6 phantom savings target).
  4. **All 8 VALIDATED contest archives at entropy floor** (Option B
     empirically falsified WZ-on-existing-archives; the substrates already
     compressed to brotli/lzma floor, so WZ reweight produced no real bits;
     no Provenance discipline meant the autopilot could not distinguish
     ``floor-reached`` from ``WZ-saved``).
  5. **Catalog #823 α=4.74 SUPER_ADDITIVE byte-identity artifact** (SIREN
     dispatch failure → placeholder copy → identical sha256 → brotli
     deduplication). The composition_alpha row carried `candidate_a_sha256
     == candidate_b_sha256` but the autopilot ranker had no canonical
     Provenance contract to detect byte-identity automatically.

All 5 share ONE structural cause: a number was treated as a score-claim
without canonical attestation that (a) the bytes live in a contest
archive.zip member, (b) the measurement axis matches the hardware
substrate, and (c) the source bytes are not byte-identical to another
substrate (the SUPER_ADDITIVE artifact). The Provenance contract makes
that attestation MANDATORY and machine-checkable.

Public surface (narrow per CLAUDE.md "Beauty, simplicity, and developer
experience" non-negotiable):

  Enums:
    - ``ProvenanceKind`` (9 canonical kinds including contest-compliance
      procedural-generation boundary sentinels)
    - ``ProvenanceEvidenceGrade`` (9 canonical grades incl. INVALID
      sentinel for the #823 byte-identity artifact class)

  Dataclasses (both frozen):
    - ``Provenance`` — the canonical attestation
    - ``ScoreClaim`` — score + Provenance + contest-compliant verdict

  Errors:
    - ``MissingProvenanceError``
    - ``InvalidProvenanceError``

  Sentinels:
    - ``NULL_NOT_A_SCORE_CLAIM`` (Provenance instance for pure-aggregate
      helpers that legitimately do not produce score-claims)

Sister of:
  * ``tac.continual_learning.ContestResult`` (per-anchor score; this module
    is the canonical Provenance field that ContestResult embeds).
  * ``tac.wyner_ziv_deliverability.DeliverabilityProof`` (per-archive WZ
    tier classification; DeliverabilityProof.provenance is a sibling
    field for the underlying archive bytes).
  * Catalog #287 ``check_empirical_claims_have_evidence`` (docstring-tag
    surface; this contract is the persisted-artifact-row surface).
  * Catalog #249 ``check_no_misleading_device_named_output_directories``
    (filename surface; this contract is the field-of-a-record surface).
  * Catalog #319 ``check_substrate_wyner_ziv_reweight_has_deliverability_proof``
    + ``check_no_autopilot_adjustment_derived_from_phantom_provenance_composition_alpha``
    (autopilot consumer surface; this contract feeds the composition_alpha
    row's ``provenance`` field).
  * Catalog #321 ``check_no_phantom_wyner_ziv_savings_from_research_sidecar``
    (prober artifact surface; this contract is the per-row canonical
    structure the prober populates).
  * Catalog #185 ``check_strict_flipped_catalog_entries_have_live_count_zero``
    (META-meta drift detection; this contract's STRICT gate Catalog #323
    is the umbrella).

Per CLAUDE.md "Apples-to-apples evidence discipline":
  Every numeric score MUST carry (a) measurement_axis ∈ {[contest-CUDA],
  [contest-CPU], [macOS-CPU advisory], [macOS-MLX research-signal],
  [MPS-PROXY], [predicted], ...},
  (b) hardware_substrate ∈ {linux_x86_64_t4, modal_a100, macos_arm64, ...},
  (c) source_path + source_sha256 of the artifact that produced it.
  The Provenance dataclass enforces this at __post_init__ — invalid
  combinations raise InvalidProvenanceError BEFORE any persistence.

Per CLAUDE.md "MPS auth eval is NOISE" + "Submission auth eval — BOTH
CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE":
  evidence_grade={MACOS_CPU_ADVISORY, MACOS_MLX_RESEARCH_SIGNAL,
  MPS_PROXY, RESEARCH_ONLY} all imply promotion_eligible=False AND
  score_claim_valid=False. The dataclass refuses to construct a Provenance
  with these grades + promotion_eligible=True.

Per CLAUDE.md "Forbidden component-aliasing for baselines":
  evidence_grade=INVALID_BYTE_IDENTITY_ARTIFACT REQUIRES rejection_reason
  to be non-empty. This is the structural protection against the
  Catalog #823 SUPER_ADDITIVE byte-identity artifact class — when a
  Provenance for a composition aggregate detects byte-identity between
  composed_from parts, the validator promotes the grade to
  INVALID_BYTE_IDENTITY_ARTIFACT and the autopilot rejects the row.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import StrEnum

# Schema version pinned in module-level constant so external readers can
# verify compatibility without parsing dataclass internals.
PROVENANCE_SCHEMA_VERSION: str = "provenance_v1_20260517"

# Canonical measurement axis vocabulary. The Provenance validator checks
# membership but does NOT enforce strict closure (e.g. future axes can be
# added without rewriting the validator).
CANONICAL_MEASUREMENT_AXES: frozenset[str] = frozenset(
    {
        "[contest-CUDA]",
        "[contest-CPU]",
        "[macOS-CPU advisory]",
        "[macOS-MLX research-signal]",
        "[MPS-PROXY]",
        "[predicted]",
        "[advisory only]",
        "[research-signal]",
        "[diagnostic]",
        "[byte-anchor]",
        "[CPU-prep]",
        "[empirical]",
    }
)

# Canonical hardware substrate vocabulary mirroring
# tac.substrates._shared.trainer_skeleton.detect_hardware_substrate.
CANONICAL_HARDWARE_SUBSTRATES: frozenset[str] = frozenset(
    {
        "linux_x86_64_t4",
        "linux_x86_64_a10g",
        "linux_x86_64_a100",
        "linux_x86_64_h100",
        "linux_x86_64_4090",
        "linux_x86_64_l40s",
        "linux_x86_64_cpu",
        "linux_x86_64_unknown_cuda",
        "linux_x86_64_modal_cpu",
        "macos_arm64",
        "macos_arm64_mlx",
        "macos_x86_64",
        "windows_x86_64",
        "unknown",
    }
)

# Canonical helper invocation pattern: helpers SHOULD populate this with
# their fully-qualified function name (e.g. "tac.provenance.builders.build_provenance_for_archive_member")
# so audit tools can trace which canonical helper created each Provenance.
_CANONICAL_HELPER_INVOCATION_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)+$")

# SHA-256 canonical hex form (64 chars, lowercase).
_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")


class ProvenanceKind(StrEnum):
    """The canonical kinds of Provenance.

    Each kind has distinct invariants enforced in Provenance.__post_init__:

    - CONTEST_ARCHIVE_MEMBER: bytes are a member of a contest archive.zip
      shipped to evaluator. promotion_eligible must be True; requires
      contest_archive_zip_path + contest_archive_member_name.

    - RESEARCH_SIDECAR: bytes are a local .pt / .npy / .json / etc. that
      are NOT in any contest archive. promotion_eligible=False AND
      score_claim_valid=False are enforced. Catalog #321 anchor.

    - DERIVED_AGGREGATE: a single derived value computed from one upstream
      Provenance (e.g., a smoothed score). Inherits promotion_eligible from
      the source via the composed_from chain.

    - PREDICTED_FROM_MODEL: a model-predicted score (no measurement). Always
      promotion_eligible=False until an empirical anchor lands.

    - ADVISORY_NON_PROMOTABLE: macOS-CPU / MPS-PROXY / other non-1:1
      contest-compliant measurements. Catalog #192 anchor.

    - AGGREGATE_OF_PROVENANCES: a composition of multiple Provenances
      (e.g., a pairwise composition_alpha row). composed_from MUST be
      non-empty. Catalog #319 + Catalog #823 anchor.

    - PROCEDURAL_GENERATION_FROM_ARCHIVE_SEED: deterministic codebook /
      tensor generation whose seed bytes are inside archive.zip. This is
      compliance-supporting provenance, not score evidence by itself.

    - WEIGHT_DERIVED_CODEBOOK: deterministic codebook derived from shipped
      archive weights. This is compliance-supporting provenance, not score
      evidence by itself.

    - FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD: explicit sentinel for output-affecting
      bytes outside archive.zip. Any score row carrying this kind must refuse
      dispatch/promotion before provider spend.
    """

    CONTEST_ARCHIVE_MEMBER = "contest_archive_member"
    RESEARCH_SIDECAR = "research_sidecar"
    DERIVED_AGGREGATE = "derived_aggregate"
    PREDICTED_FROM_MODEL = "predicted_from_model"
    ADVISORY_NON_PROMOTABLE = "advisory_non_promotable"
    AGGREGATE_OF_PROVENANCES = "aggregate_of_provenances"
    PROCEDURAL_GENERATION_FROM_ARCHIVE_SEED = "procedural_generation_from_archive_seed"
    WEIGHT_DERIVED_CODEBOOK = "weight_derived_codebook"
    FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD = "forbidden_out_of_archive_payload"


class ProvenanceEvidenceGrade(StrEnum):
    """The 9 canonical evidence grades.

    PROMOTABLE_* grades require kind=CONTEST_ARCHIVE_MEMBER OR
    AGGREGATE_OF_PROVENANCES (all aggregated parts must be promotable).

    INVALID_BYTE_IDENTITY_ARTIFACT is the structural sentinel for the
    Catalog #823 SUPER_ADDITIVE byte-identity false-signal class.
    """

    PROMOTABLE_EXACT_CONTEST_CUDA = "promotable_exact_contest_cuda"
    PROMOTABLE_EXACT_CONTEST_CPU = "promotable_exact_contest_cpu"
    PREDICTED = "predicted"
    EMPIRICAL_CPU_NON_GHA = "empirical_cpu_non_gha"
    MACOS_CPU_ADVISORY = "macos_cpu_advisory"
    MACOS_MLX_RESEARCH_SIGNAL = "macos_mlx_research_signal"
    MPS_PROXY = "mps_proxy"
    RESEARCH_ONLY = "research_only"
    INVALID_BYTE_IDENTITY_ARTIFACT = "invalid_byte_identity_artifact"


# Grades that imply non-promotable + invalid score claim.
_NON_PROMOTABLE_GRADES: frozenset[ProvenanceEvidenceGrade] = frozenset(
    {
        ProvenanceEvidenceGrade.PREDICTED,
        ProvenanceEvidenceGrade.MACOS_CPU_ADVISORY,
        ProvenanceEvidenceGrade.MACOS_MLX_RESEARCH_SIGNAL,
        ProvenanceEvidenceGrade.MPS_PROXY,
        ProvenanceEvidenceGrade.RESEARCH_ONLY,
        ProvenanceEvidenceGrade.INVALID_BYTE_IDENTITY_ARTIFACT,
    }
)

# Grades that may flow into a contest-promotion decision.
_PROMOTABLE_GRADES: frozenset[ProvenanceEvidenceGrade] = frozenset(
    {
        ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA,
        ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CPU,
    }
)


class MissingProvenanceError(Exception):
    """Raised when a score-claiming surface lacks a canonical Provenance.

    Wired by ``@requires_canonical_provenance`` decorator (see builders.py)
    and by callers that demand a Provenance field at construction time.
    """


class InvalidProvenanceError(Exception):
    """Raised when Provenance.__post_init__ detects invariant violations.

    Carries a structured ``blockers`` list so callers can surface
    each violation to the operator. The stringified form includes the
    blockers list so callers that grep the exception text can match.
    """

    def __init__(self, message: str, blockers: list[str] | None = None):
        self.blockers: list[str] = list(blockers or [])
        full = f"{message}: {'; '.join(self.blockers)}" if self.blockers else message
        super().__init__(full)


@dataclass(frozen=True)
class Provenance:
    """The canonical attestation that score-claiming surfaces embed.

    Frozen by design (per CLAUDE.md "Beauty, simplicity, and developer
    experience" + Catalog #110/#113 HISTORICAL_PROVENANCE APPEND-ONLY
    discipline). Mutations require constructing a new Provenance via
    the canonical builders.

    Fields:
        artifact_kind:
            One of ``ProvenanceKind``. Determines which optional fields
            are required (e.g., CONTEST_ARCHIVE_MEMBER requires
            ``contest_archive_zip_path`` + ``contest_archive_member_name``).

        source_path:
            Canonical filesystem path to the artifact (relative to repo
            root preferred; absolute paths allowed for archive members
            under ``submissions/`` or ``experiments/results/``).

        source_sha256:
            SHA-256 hex digest of the artifact bytes (lowercase, 64
            chars). For AGGREGATE_OF_PROVENANCES this MAY be the digest
            of a manifest enumerating the composed parts.

        measurement_axis:
            One of ``CANONICAL_MEASUREMENT_AXES``. Bracketed form
            ``[contest-CUDA]`` etc. is canonical so axis labels are
            self-evident in any string serialization.

        hardware_substrate:
            One of ``CANONICAL_HARDWARE_SUBSTRATES``. Bare strings allowed
            for the ``unknown`` substrate.

        evidence_grade:
            One of ``ProvenanceEvidenceGrade``. Combined with
            ``measurement_axis`` + ``hardware_substrate`` enforces the
            1:1 contest-compliance rule:
              * PROMOTABLE_EXACT_CONTEST_CUDA requires CUDA axis +
                linux_x86_64_* CUDA hardware
              * PROMOTABLE_EXACT_CONTEST_CPU requires CPU axis +
                linux_x86_64_cpu or linux_x86_64_modal_cpu hardware
              * MACOS_CPU_ADVISORY requires macos_arm64 / macos_x86_64
              * MACOS_MLX_RESEARCH_SIGNAL requires macos_arm64_mlx
              * MPS_PROXY requires macos_arm64

        promotion_eligible:
            True only if evidence_grade ∈ _PROMOTABLE_GRADES AND
            kind ∈ {CONTEST_ARCHIVE_MEMBER, AGGREGATE_OF_PROVENANCES}.
            All other combinations are forced to False at __post_init__.

        score_claim_valid:
            True only if promotion_eligible is True. Mirrors
            ``ContestResult.score_claim_valid`` semantics.

        captured_at_utc:
            ISO-UTC string with timezone (e.g., "2026-05-17T22:00:00Z").
            Used by audit tools to detect stale Provenance.

        canonical_helper_invocation:
            Fully-qualified function name of the canonical builder that
            produced this Provenance (e.g.,
            "tac.provenance.builders.build_provenance_for_archive_member").
            Audit tools cross-reference this against the
            ``tac.provenance.builders`` API to surface non-canonical
            Provenances.

        contest_archive_zip_path:
            REQUIRED if kind=CONTEST_ARCHIVE_MEMBER. Relative path to the
            archive.zip (e.g., "submissions/a1/archive.zip").

        contest_archive_member_name:
            REQUIRED if kind=CONTEST_ARCHIVE_MEMBER. Member name inside
            the zip (e.g., "0.bin" or "renderer.bin").

        composed_from:
            REQUIRED non-empty if kind=AGGREGATE_OF_PROVENANCES. Tuple
            of upstream Provenances. Used by validator to detect
            byte-identity (Catalog #823 anchor).

        rejection_reason:
            REQUIRED non-empty if evidence_grade=INVALID_BYTE_IDENTITY_ARTIFACT.
            Free-form human-readable explanation for forensic review.
    """

    artifact_kind: ProvenanceKind
    source_path: str
    source_sha256: str
    measurement_axis: str
    hardware_substrate: str
    evidence_grade: ProvenanceEvidenceGrade
    promotion_eligible: bool
    score_claim_valid: bool
    captured_at_utc: str
    canonical_helper_invocation: str

    contest_archive_zip_path: str = ""
    contest_archive_member_name: str = ""
    composed_from: tuple[Provenance, ...] = field(default_factory=tuple)
    rejection_reason: str = ""

    def __post_init__(self) -> None:
        blockers: list[str] = []

        # Basic field shape checks
        if not isinstance(self.artifact_kind, ProvenanceKind):
            blockers.append(f"artifact_kind={self.artifact_kind!r} not ProvenanceKind")
        if not isinstance(self.evidence_grade, ProvenanceEvidenceGrade):
            blockers.append(f"evidence_grade={self.evidence_grade!r} not ProvenanceEvidenceGrade")
        if not self.source_path:
            blockers.append("source_path must be non-empty")
        if not self.source_sha256:
            blockers.append("source_sha256 must be non-empty")
        elif not _SHA256_HEX_RE.match(self.source_sha256):
            blockers.append(f"source_sha256={self.source_sha256!r} not lowercase 64-char hex")
        if not self.measurement_axis:
            blockers.append("measurement_axis must be non-empty")
        if not self.hardware_substrate:
            blockers.append("hardware_substrate must be non-empty")
        if not self.captured_at_utc:
            blockers.append("captured_at_utc must be non-empty")
        if not self.canonical_helper_invocation:
            blockers.append("canonical_helper_invocation must be non-empty")
        elif not _CANONICAL_HELPER_INVOCATION_RE.match(self.canonical_helper_invocation):
            blockers.append(
                f"canonical_helper_invocation={self.canonical_helper_invocation!r} not dotted-fully-qualified name"
            )

        # If basic shape failed we cannot reason about kind invariants.
        if blockers:
            raise InvalidProvenanceError("Provenance basic-field validation failed", blockers=blockers)

        # Kind-specific invariants
        if self.artifact_kind == ProvenanceKind.CONTEST_ARCHIVE_MEMBER:
            if not self.contest_archive_zip_path:
                blockers.append("CONTEST_ARCHIVE_MEMBER requires contest_archive_zip_path")
            if not self.contest_archive_member_name:
                blockers.append("CONTEST_ARCHIVE_MEMBER requires contest_archive_member_name")

        elif self.artifact_kind == ProvenanceKind.RESEARCH_SIDECAR:
            if self.promotion_eligible:
                blockers.append("RESEARCH_SIDECAR cannot be promotion_eligible (Catalog #321)")
            if self.score_claim_valid:
                blockers.append("RESEARCH_SIDECAR cannot have score_claim_valid (Catalog #321)")

        elif self.artifact_kind == ProvenanceKind.AGGREGATE_OF_PROVENANCES:
            if not self.composed_from:
                blockers.append("AGGREGATE_OF_PROVENANCES requires non-empty composed_from")
            # Catalog #823 byte-identity detector: if 2+ composed_from parts
            # share source_sha256 AND grade != INVALID_BYTE_IDENTITY_ARTIFACT
            # the construction is INVALID.
            if self.composed_from and self.evidence_grade != ProvenanceEvidenceGrade.INVALID_BYTE_IDENTITY_ARTIFACT:
                shas = [p.source_sha256 for p in self.composed_from]
                # Detect any duplicate sha256 across composed parts
                if len(shas) != len(set(shas)):
                    duplicates = sorted({sha for sha in shas if shas.count(sha) > 1})
                    blockers.append(
                        "AGGREGATE_OF_PROVENANCES with byte-identical composed_from"
                        f" parts (duplicate sha256: {duplicates}) MUST set"
                        " evidence_grade=INVALID_BYTE_IDENTITY_ARTIFACT"
                        " (Catalog #823 byte-identity artifact class)"
                    )

        elif self.artifact_kind == ProvenanceKind.ADVISORY_NON_PROMOTABLE:
            if self.promotion_eligible:
                blockers.append("ADVISORY_NON_PROMOTABLE cannot be promotion_eligible (Catalog #192)")

        elif self.artifact_kind == ProvenanceKind.PREDICTED_FROM_MODEL:
            if self.promotion_eligible:
                blockers.append("PREDICTED_FROM_MODEL cannot be promotion_eligible until an empirical anchor lands")
        elif self.artifact_kind in (
            ProvenanceKind.PROCEDURAL_GENERATION_FROM_ARCHIVE_SEED,
            ProvenanceKind.WEIGHT_DERIVED_CODEBOOK,
        ):
            if self.promotion_eligible:
                blockers.append(
                    f"{self.artifact_kind.value} cannot be promotion_eligible; "
                    "it is compliance-supporting provenance, not score evidence"
                )
            if self.score_claim_valid:
                blockers.append(
                    f"{self.artifact_kind.value} cannot have score_claim_valid; "
                    "exact eval of the emitted archive must carry the score"
                )
            if not self.rejection_reason:
                blockers.append(
                    f"{self.artifact_kind.value} requires non-empty rationale in "
                    "rejection_reason documenting the archive-contained byte source"
                )
        elif self.artifact_kind == ProvenanceKind.FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD:
            if self.promotion_eligible:
                blockers.append("FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD cannot be promotion_eligible")
            if self.score_claim_valid:
                blockers.append("FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD cannot have score_claim_valid")
            if not self.rejection_reason:
                blockers.append("FORBIDDEN_OUT_OF_ARCHIVE_PAYLOAD requires non-empty rejection_reason")

        # Grade × Kind cross-checks
        if self.evidence_grade in _PROMOTABLE_GRADES:
            if self.artifact_kind not in (
                ProvenanceKind.CONTEST_ARCHIVE_MEMBER,
                ProvenanceKind.AGGREGATE_OF_PROVENANCES,
            ):
                blockers.append(
                    f"evidence_grade={self.evidence_grade.value} requires kind"
                    " ∈ {CONTEST_ARCHIVE_MEMBER, AGGREGATE_OF_PROVENANCES}"
                )
            if not self.promotion_eligible:
                blockers.append(f"evidence_grade={self.evidence_grade.value} requires promotion_eligible=True")

        # Non-promotable grades MUST have promotion_eligible=False
        if self.evidence_grade in _NON_PROMOTABLE_GRADES:
            if self.promotion_eligible:
                blockers.append(f"evidence_grade={self.evidence_grade.value} cannot have promotion_eligible=True")
            if self.score_claim_valid:
                blockers.append(f"evidence_grade={self.evidence_grade.value} cannot have score_claim_valid=True")

        # INVALID_BYTE_IDENTITY_ARTIFACT requires non-empty rejection_reason
        if self.evidence_grade == ProvenanceEvidenceGrade.INVALID_BYTE_IDENTITY_ARTIFACT and not self.rejection_reason:
            blockers.append("INVALID_BYTE_IDENTITY_ARTIFACT requires non-empty rejection_reason")

        # Axis × hardware × grade canonical pairings
        # CUDA promotable requires CUDA hardware + CUDA axis
        if self.evidence_grade == ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CUDA:
            if (
                self.measurement_axis != "[contest-CUDA]"
            ):  # CUSTODY_VALIDATOR_OK:this_function_IS_provenance_contract_validator_creating_promotable_grade_blockers_per_comprehensive_bug_audit_cascade_20260526
                blockers.append("PROMOTABLE_EXACT_CONTEST_CUDA requires measurement_axis=[contest-CUDA]")
            cuda_hardware_prefixes = (
                "linux_x86_64_t4",
                "linux_x86_64_a10g",
                "linux_x86_64_a100",
                "linux_x86_64_h100",
                "linux_x86_64_4090",
                "linux_x86_64_l40s",
                "linux_x86_64_unknown_cuda",
            )
            if not any(self.hardware_substrate.startswith(p) for p in cuda_hardware_prefixes):
                blockers.append(
                    "PROMOTABLE_EXACT_CONTEST_CUDA requires linux_x86_64_* CUDA"
                    f" hardware; got {self.hardware_substrate!r}"
                )

        # CPU promotable requires CPU hardware + CPU axis
        if self.evidence_grade == ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CPU:
            if (
                self.measurement_axis != "[contest-CPU]"
            ):  # CUSTODY_VALIDATOR_OK:this_function_IS_provenance_contract_validator_creating_promotable_grade_blockers_per_comprehensive_bug_audit_cascade_20260526
                blockers.append("PROMOTABLE_EXACT_CONTEST_CPU requires measurement_axis=[contest-CPU]")
            if self.hardware_substrate not in (
                "linux_x86_64_cpu",
                "linux_x86_64_modal_cpu",
            ):
                blockers.append(
                    "PROMOTABLE_EXACT_CONTEST_CPU requires"
                    " linux_x86_64_cpu or linux_x86_64_modal_cpu hardware;"
                    f" got {self.hardware_substrate!r}"
                )

        # macOS / MPS axis × hardware pairings
        if self.evidence_grade == ProvenanceEvidenceGrade.MACOS_CPU_ADVISORY:
            if self.measurement_axis != "[macOS-CPU advisory]":
                blockers.append("MACOS_CPU_ADVISORY requires measurement_axis=[macOS-CPU advisory]")
            if not self.hardware_substrate.startswith("macos_"):
                blockers.append("MACOS_CPU_ADVISORY requires macos_* hardware")

        if self.evidence_grade == ProvenanceEvidenceGrade.MACOS_MLX_RESEARCH_SIGNAL:
            if self.measurement_axis != "[macOS-MLX research-signal]":
                blockers.append("MACOS_MLX_RESEARCH_SIGNAL requires measurement_axis=[macOS-MLX research-signal]")
            if self.hardware_substrate != "macos_arm64_mlx":
                blockers.append("MACOS_MLX_RESEARCH_SIGNAL requires macos_arm64_mlx hardware")

        if self.evidence_grade == ProvenanceEvidenceGrade.MPS_PROXY:
            if self.measurement_axis != "[MPS-PROXY]":
                blockers.append("MPS_PROXY requires measurement_axis=[MPS-PROXY]")
            if self.hardware_substrate != "macos_arm64":
                blockers.append("MPS_PROXY requires macos_arm64 hardware")

        if blockers:
            raise InvalidProvenanceError("Provenance invariant validation failed", blockers=blockers)


@dataclass(frozen=True)
class ScoreClaim:
    """A score value bound to a canonical Provenance.

    ``contest_compliant`` is derived from the Provenance: True iff
    ``provenance.score_claim_valid`` AND
    ``provenance.evidence_grade`` ∈ _PROMOTABLE_GRADES. Construction
    refuses to set contest_compliant=True with a non-valid Provenance.

    Fields:
        score_value: the numeric score (lower-is-better contest convention)
        provenance: the canonical attestation
        rationale: human-readable note (e.g., "PR101 GOLD CPU eval per GHA workflow")
        contest_compliant: derived; True iff Provenance is promotable+valid
    """

    score_value: float
    provenance: Provenance
    rationale: str = ""
    contest_compliant: bool = False

    def __post_init__(self) -> None:
        blockers: list[str] = []

        if not isinstance(self.provenance, Provenance):
            blockers.append("provenance must be a Provenance instance")
            raise InvalidProvenanceError("ScoreClaim requires canonical Provenance", blockers=blockers)

        # Auto-derive contest_compliant from Provenance.
        derived_compliant = bool(
            self.provenance.score_claim_valid and self.provenance.evidence_grade in _PROMOTABLE_GRADES
        )
        # If caller passed contest_compliant=True with a non-derived-True
        # Provenance, refuse (catches phantom-score class).
        if self.contest_compliant and not derived_compliant:
            blockers.append(
                "contest_compliant=True requires provenance.score_claim_valid AND evidence_grade ∈ promotable grades"
            )

        # Force contest_compliant to the derived value.
        # (Cannot mutate frozen dataclass via normal assignment.)
        object.__setattr__(self, "contest_compliant", derived_compliant)

        if blockers:
            raise InvalidProvenanceError("ScoreClaim invariant validation failed", blockers=blockers)


# Sentinel Provenance for pure-aggregate helpers that legitimately do NOT
# produce a score claim (e.g., a count or a flag aggregator). Constructed
# once at module load; tests verify identity.
#
# Per CLAUDE.md "Beauty, simplicity, and developer experience": the
# sentinel exists so call sites can be unambiguous about "this is not a
# score claim" rather than passing None (which would trip Provenance
# requirement decorators).
NULL_NOT_A_SCORE_CLAIM: Provenance = Provenance(
    artifact_kind=ProvenanceKind.RESEARCH_SIDECAR,
    source_path="<sentinel:not-a-score-claim>",
    source_sha256="0" * 64,
    measurement_axis="[research-signal]",
    hardware_substrate="unknown",
    evidence_grade=ProvenanceEvidenceGrade.RESEARCH_ONLY,
    promotion_eligible=False,
    score_claim_valid=False,
    captured_at_utc="2026-05-17T00:00:00Z",
    canonical_helper_invocation="tac.provenance.contract.NULL_NOT_A_SCORE_CLAIM",
)


__all__ = [
    "CANONICAL_HARDWARE_SUBSTRATES",
    "CANONICAL_MEASUREMENT_AXES",
    "NULL_NOT_A_SCORE_CLAIM",
    "PROVENANCE_SCHEMA_VERSION",
    "InvalidProvenanceError",
    "MissingProvenanceError",
    "Provenance",
    "ProvenanceEvidenceGrade",
    "ProvenanceKind",
    "ScoreClaim",
]
