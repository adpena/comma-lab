# SPDX-License-Identifier: MIT
"""Canonical ``Atom`` frozen dataclass — the META-META-META element.

One atom row subsumes seven previously-scattered atom-shaped surfaces
(see ``tac.atom.types.AtomKind``) into ONE composable canonical type that:

  - composes with ``tac.unified_action.Action`` via the bridge in
    ``tac.atom.unified_action_bridge`` (Catalog #125 hook 2 Pareto-constraint
    contribution; each Atom feeds one additive term),
  - emits canonical observability per Catalog #305 6-facet (inspectable
    per layer / decomposable per signal / diff-able across runs /
    queryable post-hoc / cite-able / counterfactual-able),
  - carries provenance per Catalog #323 (every score-claim carries a
    canonical ``tac.provenance.Provenance``),
  - declares wire-in per Catalog #125 6-hook subset in ``wired_hooks``,
  - serves as an element in the meta-Lagrangian search per CLAUDE.md
    "Meta-Lagrangian/Pareto solver" non-negotiable: typed rows the
    planner can rank by expected score delta + expected information gain.

Why frozen dataclass + Enum + Protocol (NOT pydantic):
  Matches existing canonical patterns (``tac.unified_action.Action``,
  ``tac.provenance.Provenance``, ``tac.council_continual_learning.\
  CouncilDeliberationRecord``, ``tac.deploy.modal.call_id_ledger``).
  Zero runtime overhead. Stdlib only. Native ``__post_init__`` validation
  + canonical builders centralize creation so the construction surface
  refuses malformed atoms BEFORE persistence per Catalog #131 sister
  discipline.

Per operator-approved design 2026-05-18 verbatim "yes" on the
synthesis-memo-amendment design pitch.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from .types import AtomKind, AtomValidationError, ResolutionPath

ATOM_SCHEMA_VERSION = "atom_v1_20260518"

# Catalog #125 6-hook canonical surface. ``wired_hooks`` MUST be a non-empty
# subset of this tuple. Atoms with no wired hook are forbidden because they
# silently orphan the score-improvement signal per CLAUDE.md
# "Subagent coherence-by-default" non-negotiable + Catalog #125 sister gate.
WIRED_HOOKS_CANONICAL: tuple[str, ...] = (
    "sensitivity_map",
    "pareto_constraint",
    "bit_allocator",
    "cathedral_autopilot_dispatch",
    "continual_learning_posterior",
    "probe_disambiguator",
)

# Catalog #305 6-facet canonical observability surface. ``observability_surface``
# MUST be a non-empty subset of this tuple. Atoms with no observable facet are
# forbidden because their behavior cannot be evaluated / decomposed / diffed /
# queried / cited / counterfactually probed without re-instrumentation.
OBSERVABILITY_FACETS_CANONICAL: tuple[str, ...] = (
    "inspectable_per_layer",
    "decomposable_per_signal",
    "diff_able_across_runs",
    "queryable_post_hoc",
    "cite_able",
    "counterfactual_able",
)


@dataclass(frozen=True, slots=True)
class Atom:
    """Canonical atom-shaped element.

    Fields are split into REQUIRED (no default) and OPTIONAL (default
    factory). The required-field set is the minimal contract every atom
    kind shares; the optional-field set carries kind-specific extension
    payloads via the ``metadata`` Mapping.
    """

    # --- identity ---
    atom_id: str
    kind: AtomKind
    resolution_path: ResolutionPath

    # --- predicted impact (closed real interval; lower <= upper) ---
    predicted_impact_delta_s_lower: float
    predicted_impact_delta_s_upper: float

    # --- cost (USD; >= 0) ---
    cost_envelope_usd: float

    # --- canonical observability + provenance + wire-in ---
    # ``provenance`` is a Mapping shape matching ``tac.provenance.Provenance``
    # serialized via ``dataclasses.asdict``; we keep it as Mapping rather than
    # importing the concrete dataclass to avoid an import cycle with
    # ``tac.provenance`` (which itself may want to construct atoms in the
    # future). Validation happens in __post_init__.
    provenance: Mapping[str, Any]
    wired_hooks: Sequence[str]
    observability_surface: Sequence[str]

    # --- citation + helper link ---
    literature_citation: str
    canonical_helper_repo_link: str

    # --- kind-specific extension payload ---
    metadata: Mapping[str, Any] = field(default_factory=dict)

    # --- schema versioning for forward-compat ---
    schema: str = ATOM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate per-field + per-kind invariants.

        Per Catalog #131 sister discipline: the construction surface refuses
        malformed atoms so downstream consumers (autopilot ranker / Pareto
        solver / continual-learning posterior) inherit the validation
        guarantee structurally.
        """
        # --- identity invariants ---
        if not isinstance(self.atom_id, str) or not self.atom_id:
            raise AtomValidationError(
                f"atom_id must be a non-empty str (got {self.atom_id!r})"
            )
        if "\n" in self.atom_id or "\r" in self.atom_id:
            raise AtomValidationError(
                f"atom_id must not contain newlines (got {self.atom_id!r}); "
                "the JSONL ledger fails closed on tail-validation per Catalog #245"
            )
        if not isinstance(self.kind, AtomKind):
            raise AtomValidationError(
                f"kind must be AtomKind (got {type(self.kind).__name__}: {self.kind!r})"
            )
        if not isinstance(self.resolution_path, ResolutionPath):
            raise AtomValidationError(
                f"resolution_path must be ResolutionPath (got "
                f"{type(self.resolution_path).__name__}: {self.resolution_path!r})"
            )

        # --- predicted-impact interval invariants ---
        for fname, fval in (
            ("predicted_impact_delta_s_lower", self.predicted_impact_delta_s_lower),
            ("predicted_impact_delta_s_upper", self.predicted_impact_delta_s_upper),
        ):
            if not isinstance(fval, (int, float)):
                raise AtomValidationError(
                    f"{fname} must be float-coercible (got {type(fval).__name__})"
                )
        if self.predicted_impact_delta_s_lower > self.predicted_impact_delta_s_upper:
            raise AtomValidationError(
                f"predicted_impact_delta_s_lower={self.predicted_impact_delta_s_lower} "
                f"must be <= upper={self.predicted_impact_delta_s_upper}"
            )

        # --- cost invariants ---
        if not isinstance(self.cost_envelope_usd, (int, float)):
            raise AtomValidationError(
                f"cost_envelope_usd must be float-coercible (got "
                f"{type(self.cost_envelope_usd).__name__})"
            )
        if self.cost_envelope_usd < 0:
            raise AtomValidationError(
                f"cost_envelope_usd must be >= 0 (got {self.cost_envelope_usd})"
            )

        # --- provenance invariant: canonical 14-field shape per Catalog #323 ---
        if not isinstance(self.provenance, Mapping):
            raise AtomValidationError(
                f"provenance must be a Mapping (got {type(self.provenance).__name__})"
            )
        required_provenance_keys = ("artifact_kind", "evidence_grade", "captured_at_utc")
        missing_pkeys = [k for k in required_provenance_keys if k not in self.provenance]
        if missing_pkeys:
            raise AtomValidationError(
                f"provenance missing required keys {missing_pkeys}; "
                "use tac.provenance.build_provenance_for_* canonical builders "
                "per Catalog #323"
            )

        # --- wired_hooks invariant: non-empty subset of canonical 6 ---
        if not isinstance(self.wired_hooks, Sequence) or isinstance(self.wired_hooks, str):
            raise AtomValidationError(
                f"wired_hooks must be a non-string Sequence (got "
                f"{type(self.wired_hooks).__name__})"
            )
        if not self.wired_hooks:
            raise AtomValidationError(
                "wired_hooks must be non-empty; per CLAUDE.md "
                "'Subagent coherence-by-default' + Catalog #125 6-hook wire-in "
                "non-negotiable, atoms with no wired hook are forbidden "
                "(orphan-signal failure mode)"
            )
        unknown_hooks = [h for h in self.wired_hooks if h not in WIRED_HOOKS_CANONICAL]
        if unknown_hooks:
            raise AtomValidationError(
                f"wired_hooks contains non-canonical entries {unknown_hooks}; "
                f"canonical set: {WIRED_HOOKS_CANONICAL}"
            )

        # --- observability_surface invariant: non-empty subset of canonical 6 ---
        if not isinstance(self.observability_surface, Sequence) or isinstance(
            self.observability_surface, str
        ):
            raise AtomValidationError(
                f"observability_surface must be a non-string Sequence (got "
                f"{type(self.observability_surface).__name__})"
            )
        if not self.observability_surface:
            raise AtomValidationError(
                "observability_surface must be non-empty per Catalog #305 "
                "'Max observability' non-negotiable; atoms with no observable "
                "facet cannot be evaluated / decomposed / diff'ed / queried / "
                "cited / counterfactually probed"
            )
        unknown_facets = [
            f for f in self.observability_surface if f not in OBSERVABILITY_FACETS_CANONICAL
        ]
        if unknown_facets:
            raise AtomValidationError(
                f"observability_surface contains non-canonical entries {unknown_facets}; "
                f"canonical set: {OBSERVABILITY_FACETS_CANONICAL}"
            )

        # --- citation + helper link invariants ---
        # Per CLAUDE.md "Beauty, simplicity, and developer experience" +
        # operator standing directive 2026-05-18: "ensure citations and
        # provenance and links". Empty strings are accepted ONLY for
        # PREMISE_VERIFICATION kind (which is itself the canonical
        # verification primitive and has no external literature anchor).
        if self.kind != AtomKind.PREMISE_VERIFICATION:
            if not isinstance(self.literature_citation, str) or not self.literature_citation:
                raise AtomValidationError(
                    f"literature_citation must be a non-empty str for kind "
                    f"{self.kind.name}; only PREMISE_VERIFICATION allows empty"
                )
        else:
            if not isinstance(self.literature_citation, str):
                raise AtomValidationError(
                    f"literature_citation must be a str (got "
                    f"{type(self.literature_citation).__name__})"
                )
        if not isinstance(self.canonical_helper_repo_link, str):
            raise AtomValidationError(
                f"canonical_helper_repo_link must be a str (got "
                f"{type(self.canonical_helper_repo_link).__name__})"
            )

        # --- metadata invariant ---
        if not isinstance(self.metadata, Mapping):
            raise AtomValidationError(
                f"metadata must be a Mapping (got {type(self.metadata).__name__})"
            )

        # --- per-kind invariants ---
        self._validate_per_kind()

    def _validate_per_kind(self) -> None:
        """Per-kind additional invariants.

        Each AtomKind may require kind-specific metadata fields. Keep these
        narrow so the META layer stays uniform across kinds; richer kind
        validation belongs in the canonical builders.
        """
        if self.kind == AtomKind.PROBE_OUTCOME:
            # Probe outcomes MUST cite a verdict per Catalog #313 taxonomy
            verdict = self.metadata.get("verdict")
            if verdict is None:
                raise AtomValidationError(
                    f"PROBE_OUTCOME atom {self.atom_id} metadata missing 'verdict' key; "
                    "required per Catalog #313 probe-outcomes-ledger contract"
                )
            valid_verdicts = {
                "INDEPENDENT",
                "KILL",
                "DEFER",
                "PROMOTE",
                "PROCEED",
                "PARTIAL",
                "OPERATOR_REVIEW_REQUIRED",
            }
            if verdict not in valid_verdicts:
                raise AtomValidationError(
                    f"PROBE_OUTCOME atom {self.atom_id} verdict={verdict!r} not in "
                    f"canonical Catalog #313 taxonomy {valid_verdicts}"
                )
        elif self.kind == AtomKind.COUNCIL_DELIBERATION:
            # Council deliberations MUST cite a tier per Catalog #300 v2 frontmatter
            tier = self.metadata.get("council_tier")
            if tier is None:
                raise AtomValidationError(
                    f"COUNCIL_DELIBERATION atom {self.atom_id} metadata missing "
                    "'council_tier'; required per Catalog #300 v2 frontmatter contract"
                )
            if tier not in {"T1", "T2", "T3", "T4"}:
                raise AtomValidationError(
                    f"COUNCIL_DELIBERATION atom {self.atom_id} council_tier={tier!r} "
                    "must be one of {T1, T2, T3, T4} per Catalog #300"
                )
        elif self.kind == AtomKind.CARGO_CULT_ASSUMPTION:
            # Cargo-cult atoms MUST carry the HARD-EARNED-vs-CARGO-CULTED classification
            classification = self.metadata.get("classification")
            if classification is None:
                raise AtomValidationError(
                    f"CARGO_CULT_ASSUMPTION atom {self.atom_id} metadata missing "
                    "'classification'; required per the hard-earned-vs-cargo-culted "
                    "addendum + Catalog #303"
                )
            valid_classifications = {"HARD-EARNED", "CARGO-CULTED", "UNDECIDED"}
            if classification not in valid_classifications:
                raise AtomValidationError(
                    f"CARGO_CULT_ASSUMPTION atom {self.atom_id} classification="
                    f"{classification!r} must be one of {valid_classifications}"
                )

    def to_jsonl_row(self) -> dict[str, Any]:
        """Serialize to a JSONL-safe dict suitable for the canonical ledger.

        StrEnum members are serialized as their string value per the canonical
        ``tac.deploy.modal.call_id_ledger`` JSONL byte-stable pattern
        (``sort_keys=True`` at write time).
        """
        row = asdict(self)
        # StrEnum -> str
        row["kind"] = self.kind.value
        row["resolution_path"] = self.resolution_path.value
        # Sequence -> list for JSON stability
        row["wired_hooks"] = list(self.wired_hooks)
        row["observability_surface"] = list(self.observability_surface)
        # Mapping -> dict (in case caller passed a non-dict Mapping)
        row["provenance"] = dict(self.provenance)
        row["metadata"] = dict(self.metadata)
        return row

    def to_meta_lagrangian_atom(self) -> dict[str, Any]:
        """Convert to the existing ``tac.meta_lagrangian_allocator`` atom shape.

        This is the canonical bridge into the existing rate-only allocation
        atom format consumed by ``tac.meta_lagrangian_allocator.\
        build_atom_ledger`` and sister allocator primitives. The shape mirrors
        the existing ``atoms_from_hnerv_decoder_recode_profile`` output so
        downstream allocators do not need to know about the canonical
        ``Atom`` type.

        Note: predicted impact is mapped via midpoint of the predicted-impact
        band; downstream allocators that need the full interval should consume
        the canonical Atom directly rather than this lossy projection.
        """
        midpoint = (
            self.predicted_impact_delta_s_lower + self.predicted_impact_delta_s_upper
        ) / 2.0
        byte_delta = int(self.metadata.get("byte_delta", 0))
        return {
            "atom_id": self.atom_id,
            "family": f"atom_kind:{self.kind.value}",
            "family_group": f"resolution_path:{self.resolution_path.value}",
            "conflicts_with_families": list(self.metadata.get("conflicts_with_families", [])),
            "conflicts_with_atoms": list(self.metadata.get("conflicts_with_atoms", [])),
            "byte_delta": byte_delta,
            # The expected_*_dist_delta fields are populated from metadata if
            # the kind carries them; otherwise default to 0.0 + the midpoint
            # delta_s is exposed as expected_score_delta for the allocator.
            "expected_seg_dist_delta": float(self.metadata.get("expected_seg_dist_delta", 0.0)),
            "expected_pose_dist_delta": float(self.metadata.get("expected_pose_dist_delta", 0.0)),
            "expected_score_delta": float(midpoint),
            "expected_score_delta_lower": float(self.predicted_impact_delta_s_lower),
            "expected_score_delta_upper": float(self.predicted_impact_delta_s_upper),
            "confidence": float(self.metadata.get("confidence", 0.5)),
            "evidence_grade": str(self.provenance.get("evidence_grade", "predicted")),
            "hard_pair_support": list(self.metadata.get("hard_pair_support", [])),
            "pair_support": list(self.metadata.get("pair_support", [])),
            "class_support": list(self.metadata.get("class_support", [])),
            "geometry_priors": list(self.metadata.get("geometry_priors", [])),
            "openpilot_priors": list(self.metadata.get("openpilot_priors", [])),
            "source_archive_sha256": str(
                self.provenance.get("source_sha256", "")
            ),
            "cost_envelope_usd": float(self.cost_envelope_usd),
            "atom_kind": self.kind.value,
            "resolution_path": self.resolution_path.value,
            "literature_citation": self.literature_citation,
            "canonical_helper_repo_link": self.canonical_helper_repo_link,
            "wired_hooks": list(self.wired_hooks),
            "observability_surface": list(self.observability_surface),
        }

    def validate(self) -> None:
        """Re-run __post_init__ invariants explicitly.

        Useful for callers that mutate metadata via copy.replace and want to
        force re-validation; the dataclass is frozen so direct mutation is
        forbidden, but ``dataclasses.replace`` returns a new frozen instance
        whose __post_init__ runs automatically. This method is provided as a
        no-op-style explicit checkpoint for documentation purposes.
        """
        # __post_init__ already ran at construction. Re-run by reconstructing
        # via asdict + Atom(**) so callers have a one-line "is this still
        # valid?" hook.
        type(self)(**{k: v for k, v in asdict(self).items() if k != "schema"})


__all__ = [
    "ATOM_SCHEMA_VERSION",
    "OBSERVABILITY_FACETS_CANONICAL",
    "WIRED_HOOKS_CANONICAL",
    "Atom",
]
