"""V3 composition-carrier registry — the typed decision surface that ranks every
carrier *program family* by closeness-to-exact-score-improvement, so "where does
composition live?" is decided by evidence + exact ΔS, NOT architecture attachment.

Operator directive 2026-06-09 (COMPOSITION CARRIER TRIAGE TRANCHE): composition is
a V3 program-design principle, not a HiNeRV feature. The question is not "add a
codebook to HiNeRV" but "which composed program family lowers exact
``100*d_seg + sqrt(10*d_pose) + 25*bytes/N`` fastest per byte AND per engineering
hour." This registry makes that ranking explicit and forces every carrier to declare
its fastest path to a CandidateActionEvaluation row + its current blocker — a carrier
with no such path is *visibly* stalled (the "don't let SNeRV stall silently" rule).

NO-FAKE discipline: scores are axis-tagged and never fabricated; ``current_score`` is
None unless a real measurement exists, and ``expected_delta_s_per_byte`` is an explicit
``planning_only`` qualitative band until a real CandidateActionEvaluation row lands.
The registry is a PRIOR for the V3 waterfiller, never an authority.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Composition-kind sentinels (the structural axis the carrier exploits).
COMPOSITION_LEAF_LATENTS = "leaf_latents"  # 600 independent per-pair latents (HiNeRV today)
COMPOSITION_CODEBOOK = "codebook"  # K shared codewords + per-pair index/residual
COMPOSITION_SELECTOR_MENU = "selector_menu"  # global mode menu x per-pair selector (PR110)
COMPOSITION_SOURCE_STATE = "source_state"  # SNeRV source-forward carrier
COMPOSITION_RESIDUAL_BOOST = "residual_boost"  # base archive + clamped residual (PR110 boost)
COMPOSITION_SOURCE_RECODE = "source_recode"  # deterministic source fp-recode (the frontier)
COMPOSITION_SPARSE_ATOMS = "sparse_atoms"  # ΔS-gated boundary/pose correction atoms

# Closeness-to-exact-archive bands (the engineering-hours axis, NOT a score claim).
READINESS_HAS_EXACT_ARCHIVE = "has_exact_archive_now"  # can emit archive.zip + inflate today
READINESS_SCAFFOLD_NEEDS_TRAIN = "scaffold_needs_training_or_export"
READINESS_DESIGN_ONLY = "design_only_no_archive_path"
READINESS_PENDING_AUDIT = "pending_audit"


@dataclass(frozen=True)
class CompositionCarrierCandidate:
    """One carrier *program family* as a V3 ranking row. Every field is an honest
    record (axis-tagged scores, explicit blockers); no fabricated numbers."""

    vehicle: str  # hinerv | hinerv_codebook | pact_nerv_vq | pr110pp | snerv | source_recode | atom
    composition_kind: str
    readiness: str
    has_archive_path: bool
    has_score_aware_training: bool
    current_score: float | None
    current_score_axis: str  # [contest-CPU] | [contest-CUDA] | [macOS-CPU advisory] | none
    fastest_path_to_candidate_action_evaluation: str
    known_blocker: str
    expected_delta_s_per_byte_qualitative: str  # planning_only band (e.g. "high/uncertain")
    notes: str = ""

    def __post_init__(self) -> None:
        if self.current_score is not None and self.current_score_axis in ("", "none"):
            raise ValueError(
                f"{self.vehicle}: current_score set but axis is '{self.current_score_axis}' "
                "(every score MUST carry an axis tag — no untagged score claims)"
            )

    def as_row(self) -> dict[str, Any]:
        return {
            "vehicle": self.vehicle,
            "composition_kind": self.composition_kind,
            "readiness": self.readiness,
            "has_archive_path": self.has_archive_path,
            "has_score_aware_training": self.has_score_aware_training,
            "current_score": self.current_score,
            "current_score_axis": self.current_score_axis,
            "fastest_path_to_candidate_action_evaluation": self.fastest_path_to_candidate_action_evaluation,
            "known_blocker": self.known_blocker,
            "expected_delta_s_per_byte_qualitative": self.expected_delta_s_per_byte_qualitative,
            "notes": self.notes,
            # Non-authority markers (this registry is a PRIOR, never a score claim).
            "authority": "[planning_only]",
            "promotion_eligible": False,
            "score_claim": False,
        }


# Readiness rank: lower = closer to an exact archive candidate (fewer engineering hours).
_READINESS_RANK = {
    READINESS_HAS_EXACT_ARCHIVE: 0,
    READINESS_SCAFFOLD_NEEDS_TRAIN: 1,
    READINESS_PENDING_AUDIT: 2,
    READINESS_DESIGN_ONLY: 3,
}


def build_canonical_registry_20260609() -> list[CompositionCarrierCandidate]:
    """The honest 2026-06-09 snapshot, post fp16-authority-trace (quant exonerated;
    HiNeRV carrier broken at high precision) + pre live-MLX-gate / pre pact_nerv_vq-audit.

    Scores are tagged; the HiNeRV 0.50/151 is [macOS-CPU advisory] (B2 bridge), the
    frontier 0.192 is [contest-CPU] (read from the canonical pointer, not hardcoded as
    truth here — this is a record of the measured anchor, not a new claim)."""
    return [
        CompositionCarrierCandidate(
            vehicle="hinerv",
            composition_kind=COMPOSITION_LEAF_LATENTS,
            readiness=READINESS_HAS_EXACT_ARCHIVE,
            has_archive_path=True,
            has_score_aware_training=True,
            current_score=89.57,
            current_score_axis="[macOS-CPU advisory]",
            fastest_path_to_candidate_action_evaluation="already emits archive -> B2 bridge (done: rejected)",
            known_blocker="carrier broken at fp16 too (fp16~int8, d_seg 0.50 / d_pose 151); live-MLX gate pending to split model-bad vs MLX->numpy-parity-bad",
            expected_delta_s_per_byte_qualitative="negative_until_carrier_fixed",
            notes="600 independent per-pair leaf latents; the leaf baseline composition wants to improve on.",
        ),
        CompositionCarrierCandidate(
            vehicle="hinerv_codebook",
            composition_kind=COMPOSITION_CODEBOOK,
            readiness=READINESS_DESIGN_ONLY,
            has_archive_path=False,
            has_score_aware_training=False,
            current_score=None,
            current_score_axis="none",
            fastest_path_to_candidate_action_evaluation="retrofit codebook into HiNeRV latents + export -> B2 (only worthwhile if live-MLX gate says HiNeRV carrier is viable)",
            known_blocker="GATED on live-MLX: do NOT retrofit composition onto a carrier that is broken at fp16",
            expected_delta_s_per_byte_qualitative="high_if_carrier_viable_else_zero",
            notes="K<<600 shared codewords + per-pair index/residual; exploits dashcam cross-pair temporal redundancy.",
        ),
        CompositionCarrierCandidate(
            vehicle="pact_nerv_vq",
            composition_kind=COMPOSITION_CODEBOOK,
            readiness=READINESS_PENDING_AUDIT,
            has_archive_path=False,  # unknown until audit lands
            has_score_aware_training=False,  # unknown until audit lands
            current_score=None,
            current_score_axis="none",
            fastest_path_to_candidate_action_evaluation="PENDING pact_nerv_vq_maturity_audit (sibling lane that may already embody the codebook)",
            known_blocker="maturity unknown — audit running; if mature with archive path, invest HERE before HiNeRV retrofit",
            expected_delta_s_per_byte_qualitative="unknown_pending_audit",
            notes="The composed-latent sibling. If further along than a HiNeRV retrofit, composition investment goes here.",
        ),
        CompositionCarrierCandidate(
            vehicle="pr110pp",
            composition_kind=COMPOSITION_SELECTOR_MENU,
            readiness=READINESS_SCAFFOLD_NEEDS_TRAIN,
            has_archive_path=True,  # PR110 base + BPR1 residual grammar exists (boost_nerv_pr110_residual)
            has_score_aware_training=False,  # the differentiable selector/menu is the new work
            current_score=None,
            current_score_axis="none",
            fastest_path_to_candidate_action_evaluation="reproduce K=16 selector on frozen HNeRV -> Huffman selector stream -> B2 (the discrete form already shipped publicly as PR110)",
            known_blocker="differentiable selector x menu optimizer not built; comp-Muon-INSPIRED (not drop-in) partner-whitening on menu/selector factors is the research step",
            expected_delta_s_per_byte_qualitative="high_frontier_direct_pose_via_frame0",
            notes="frozen backend o selected mode(pair) = a composed action program; exploits frame0->PoseNet / SegNet-free asymmetry. The natural comp-Muon analogue.",
        ),
        CompositionCarrierCandidate(
            vehicle="snerv",
            composition_kind=COMPOSITION_SOURCE_STATE,
            readiness=READINESS_SCAFFOLD_NEEDS_TRAIN,
            has_archive_path=False,
            has_score_aware_training=False,
            current_score=None,
            current_score_axis="none",
            fastest_path_to_candidate_action_evaluation="TUB DROP_OR_REIFY source-forward proof -> MFU/HFR binding -> export -> B2 (must produce source-forward CandidateActionEvaluation rows or be deprioritized)",
            known_blocker="source-forward causal proof + LF/HF byte-pressure binding incomplete; NO exact-eval row yet (silent-stall risk per operator)",
            expected_delta_s_per_byte_qualitative="unknown_needs_source_forward_proof",
            notes="Alternative dense/source carrier; only admissible with causal source-forward proof AND pays-rent.",
        ),
        CompositionCarrierCandidate(
            vehicle="source_recode",
            composition_kind=COMPOSITION_SOURCE_RECODE,
            readiness=READINESS_HAS_EXACT_ARCHIVE,
            has_archive_path=True,
            has_score_aware_training=False,  # deterministic, zero trainable params
            current_score=0.19198533626623068,
            current_score_axis="[contest-CPU]",
            fastest_path_to_candidate_action_evaluation="already the CPU frontier anchor (fp11_source_brotli_recode); compose with sparse atoms as the program baseline",
            known_blocker="orphaned from the neural carriers; not yet composed with sparse evaluator atoms in V3",
            expected_delta_s_per_byte_qualitative="frontier_baseline_compose_with_atoms",
            notes="The MDL-minimal program that ACTUALLY made the frontier. No neural leaves. The benchmark every carrier must beat.",
        ),
        CompositionCarrierCandidate(
            vehicle="atom",
            composition_kind=COMPOSITION_SPARSE_ATOMS,
            readiness=READINESS_SCAFFOLD_NEEDS_TRAIN,
            has_archive_path=True,  # target_region_actions + archive_candidate grammar exists
            has_score_aware_training=False,
            current_score=None,
            current_score_axis="none",
            fastest_path_to_candidate_action_evaluation="mine from inverse-steg cost map (seg margin field) + cooperative-receiver nullspace -> materialize -> B2 ΔS",
            known_blocker="atoms only pay rent against a GOOD base; gated on a viable carrier (or on source_recode as the base)",
            expected_delta_s_per_byte_qualitative="high_on_good_base_zero_on_bad_base",
            notes="Sparse ΔS-gated SegNet-boundary / pose-Y / frame0 correction atoms. Inverse-steg + cooperative-receiver are the MINERS; V3 is the judge.",
        ),
    ]


def rank_candidates(
    candidates: list[CompositionCarrierCandidate],
) -> list[dict[str, Any]]:
    """Rank by readiness (engineering-hours proximity to an exact archive), tie-broken
    by whether a real score anchor exists. PLANNING-ONLY: the true ranking is exact ΔS
    per byte, which only the V3 waterfiller can produce from CandidateActionEvaluation
    rows. This ordering is a triage prior, explicitly NOT a score claim."""
    ordered = sorted(
        candidates,
        key=lambda c: (
            _READINESS_RANK.get(c.readiness, 99),
            0 if c.current_score is not None else 1,
            0 if c.has_archive_path else 1,
        ),
    )
    rows = []
    for rank, c in enumerate(ordered):
        row = c.as_row()
        row["triage_rank"] = rank
        rows.append(row)
    return rows


def build_registry_artifact() -> dict[str, Any]:
    """The emittable ``v3_composition_atom_registry.v1`` artifact (a triage prior)."""
    candidates = build_canonical_registry_20260609()
    return {
        "schema": "v3_composition_atom_registry.v1",
        "principle": "composition is a V3 program-design principle, not a carrier feature; "
        "location decided by exact ΔS per byte, not architecture attachment",
        "authority": "[planning_only]",
        "promotion_eligible": False,
        "score_claim": False,
        "ranked_triage": rank_candidates(candidates),
        "decision_law": "every candidate must produce a CandidateActionEvaluation row "
        "(archive.zip + inflate.sh -> evaluate.py -> d_seg/d_pose/bytes -> ΔS); "
        "delta_score_total < 0 is the only admission criterion",
        "gate_dependency": "live-MLX surface result routes HiNeRV-family readiness; "
        "pact_nerv_vq audit routes the codebook-investment location",
    }
