"""Force x (class|edge) x verb representation ledger — task #809 (ddm_cg1 / ddm_cg1r).

Operator directives (2026-07-31, twice): *"account for which forces positively improve
their representation and also semantics and verbs... and which harms them and need to
be protected against or constrained"* + *"targets/caps/dynamic forces per
class+edge+saddle vs real outcome+telemetry+archive."*

Row shape (charter):
    (class | edge | directed-edge | global) x force x VERB
        -> {IMPROVES | HARMS | NEUTRAL | INERT | UNMEASURED}
        x evidence + verdict_scope (INSTANCE < FORMULATION < FAMILY < PARADIGM)
        x protection {BUILT | BUILT_UNFIRED | DESIGNED | ABSENT | N_A}

Verbs are ddm_gt2's lexicon, adopted verbatim from
/Volumes/VertigoDataTier/pact/ddm_gt2_20260803/gt2_verbs.json:verb_lexicon (its memo
names this module as the consumer). AMPLITUDE and PHASE are out of argmax scope by
construction and are sourced from ddm_hg1 / ddm_pc2 respectively.

THE STRUCTURAL FACT every row is conditioned on (verified at source,
upstream/modules.py:112-113):

    diff = (out1.argmax(dim=1) != out2.argmax(dim=1)).float()
    return diff.mean(...)

SegNet distortion is a uniform per-pixel flip count. No per-class term, no per-edge
term, no verb term; logits are discarded. Every force ever scored on the seg leg was
scored by a scalar constitutionally blind to WHICH class it helped and to WHICH VERB
it applied.

THE GOVERNING LAW (top of the ledger, 7 independent instruments): every measured
asymmetry is real; every attempt to cash one through an AGGREGATE failed —
hg1 Simpson reversal · sx2 negative marginal value · pu2 r^2=0.000 · #900 · mf1
tau_end pooling · as1 grow_Lane_into_Road_1px = +0.2459 S (all 10 directed sides
p<0.5) · mg1 barrier-key 0.0000% matched-byte advantage. **Asymmetry is a PRIOR on a
per-site actuator, never the actuator.**

Queryability: importable typed rows + `python -m tac.witness_control.force_class_edge_ledger`
CLI (`--coverage`, `--harms`, `--unprotected`, `--force F`, `--scope S`, `--jsonl OUT`).

Axis discipline: every evidence string cites its artifact; nothing here is a score
claim; pointer 0.1910828242 [contest-CPU] UNMOVED by this module.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path

# --------------------------------------------------------------------------- #
# vocabulary (closed sets; validated in ForceLedgerRow.__post_init__)
# --------------------------------------------------------------------------- #

CLASSES: tuple[str, ...] = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
# canonical comma10k order, indices 0..4 — NEVER luma-sorted (CLAUDE.md scorer section)

EDGES: tuple[str, ...] = (
    "Road<->Lane",
    "Road<->Undrivable",
    "Road<->MyCar",
    "Undrivable<->Movable",
    "Road<->Movable",
    "Lane<->MyCar",
    "Lane<->Movable",
    "Movable<->MyCar",
    "Lane<->Undrivable",
)
# ordered by live cx1 n600 flip mass (gt2_verbs.json edge_indexed); Movable<->MyCar
# and the 3 smallest carry 0.356% of flips together.

# ddm_gt2's verb lexicon (gt2_verbs.json:verb_lexicon), adopted verbatim.
VERBS: tuple[str, ...] = (
    "DISPLACE",     # separatrix moved; mass locally conserved      (symmetric)
    "TRANSFER",     # one side genuinely lost area                  (antisymmetric)
    "ERODE",        # surviving component lost a shallow rim
    "GOUGE",        # surviving component lost interior
    "ANNIHILATE",   # a whole GT component gone — a dropped word
    "BIRTH",        # a component with no GT counterpart
    "FRAGMENT",     # SIGNED: + = word split, - = consolidation (receiver consolidates 4/5 classes)
    "AMPLITUDE",    # pre-argmax margin moved without crossing (OUT of argmax scope; hg1 owns)
    "PHASE",        # sub-pixel position (OUT of argmax scope; pc2 owns)
    # NOTE: the per-FRAME axis is CLOSED (refuted 3x, latest at edge level by gt2r:
    # top-10 frames <= 15.4% of TRANSFER, all 600 frames touched on every big edge).
    # No per-frame rows belong in this ledger.
    "AGGREGATE",    # ledger-local: the force acts on/through an aggregate scalar,
                    # i.e. it has NO verb channel — that absence is itself the finding
)

VERDICTS: tuple[str, ...] = (
    "IMPROVES",     # measured to improve the scope's representation
    "HARMS",        # measured to harm it
    "NEUTRAL",      # measured, no effect at the stated resolution
    "INERT",        # measured: the force cannot act at all (e.g. zero gradient)
    "UNMEASURED",   # no admissible evidence for a sign on this vehicle
)

VERDICT_SCOPES: tuple[str, ...] = (
    "INSTANCE",     # this vehicle/config/corpus only
    "FORMULATION",  # this formulation of the force
    "FAMILY",       # the whole family (strongest measured scope here)
    "DERIVATION",   # analytic, not measured — never a licence to build or kill
    "STRUCTURAL",   # property of the frozen scorer/operators, verified at source
    "N_A",          # for UNMEASURED rows
)

PROTECTIONS: tuple[str, ...] = (
    "BUILT",          # a guard exists and is wired on the live path
    "BUILT_UNFIRED",  # a guard exists but has never fired / is not on the live path
    "DESIGNED",       # a guard is designed (memo/derivation) but not built
    "ABSENT",         # no protection exists
    "N_A",            # protection is not meaningful for this row
)

FORCE_KINDS: tuple[str, ...] = (
    "OBJECTIVE",       # a loss/metric term
    "LEVER",           # a config knob on an objective
    "ALLOCATOR",       # a byte/capacity allocation rule
    "PRODUCTION",      # a grammar production (how a class is *represented*)
    "GUARD",           # a protection mechanism (rows account the guard itself)
    "ACTUATOR_PROBE",  # a one-shot actuation used as measurement
    "DERIVED_FORCE",   # derived on paper, not yet an implemented lever
    "METRIC",          # the scorer's own pricing (the field every force is scored in)
    "FAMILY",          # a whole family of forces (family-level verdicts)
)

_SCOPE_KINDS: tuple[str, ...] = ("GLOBAL", "CLASS", "EDGE", "DIRECTED_EDGE", "FAMILY")


@dataclass(frozen=True)
class ForceLedgerRow:
    """One accountable cell: a force acting on a scope through a verb."""

    row_id: str
    scope_kind: str        # GLOBAL | CLASS | EDGE | DIRECTED_EDGE | FAMILY
    scope: str             # "Lane" | "Road<->Lane" | "Lane->Road" | "ALL" | family name
    force: str             # canonical force id
    force_kind: str
    verb: str
    verdict: str
    measured: bool         # True iff the verdict rests on an executed measurement
    evidence: str          # the load-bearing numbers, inline
    evidence_source: str   # artifact path(s) + git hash where known
    verdict_scope: str
    protection: str
    protection_ref: str = ""
    magnitude_s: float | None = None  # |effect| in S units where known (ranking key)

    def __post_init__(self) -> None:
        if self.scope_kind not in _SCOPE_KINDS:
            raise ValueError(f"{self.row_id}: bad scope_kind {self.scope_kind!r}")
        if self.verb not in VERBS:
            raise ValueError(f"{self.row_id}: bad verb {self.verb!r}")
        if self.verdict not in VERDICTS:
            raise ValueError(f"{self.row_id}: bad verdict {self.verdict!r}")
        if self.verdict_scope not in VERDICT_SCOPES:
            raise ValueError(f"{self.row_id}: bad verdict_scope {self.verdict_scope!r}")
        if self.protection not in PROTECTIONS:
            raise ValueError(f"{self.row_id}: bad protection {self.protection!r}")
        if self.force_kind not in FORCE_KINDS:
            raise ValueError(f"{self.row_id}: bad force_kind {self.force_kind!r}")
        if self.scope_kind == "CLASS" and self.scope not in CLASSES:
            raise ValueError(f"{self.row_id}: bad class {self.scope!r}")
        if self.scope_kind == "EDGE" and self.scope not in EDGES:
            raise ValueError(f"{self.row_id}: bad edge {self.scope!r}")
        if self.scope_kind == "DIRECTED_EDGE":
            a, _, b = self.scope.partition("->")
            if a not in CLASSES or b not in CLASSES or a == b:
                raise ValueError(f"{self.row_id}: bad directed edge {self.scope!r}")
        if self.verdict == "UNMEASURED" and self.measured:
            raise ValueError(f"{self.row_id}: UNMEASURED row cannot claim measured=True")
        if self.verdict in ("IMPROVES", "HARMS", "NEUTRAL", "INERT") and not (
            self.measured or self.verdict_scope == "DERIVATION"
        ):
            raise ValueError(
                f"{self.row_id}: signed verdict requires measured=True or DERIVATION scope"
            )


# --------------------------------------------------------------------------- #
# evidence source shorthands
# --------------------------------------------------------------------------- #

_GT2 = ".omx/research/ddm_gt2_gt_tongue_induction_20260803.md + /Volumes/VertigoDataTier/pact/ddm_gt2_20260803/gt2_verbs.json"
_MG1 = ".omx/research/ddm_mg1_margin_geometry_cure_20260803.md (c942796dbc)"
_CG1R = ".omx/research/ddm_cg1_directed_edge_margin_n600.json (this arm; controls ALL PASS, conf-matrix == pu2 receipt exactly)"
_CG1 = ".omx/research/ddm_cg1_perclass_margin_reach_n600.json (predecessor, 26412a36b0)"
_PU2 = ".omx/research/ddm_pu2_pose_tail_floor_probe_20260803.md + /Volumes/VertigoDataTier/pact/ddm_pu2_20260803/argmax_cache/cx1_directed_flip_receipt.json"
_PH4 = ".omx/research/ddm_ph4_physics_photometrics_dynamics_20260803.md (8e715267ec)"
_AS1 = ".omx/research/ddm_as1_asymmetry_synergy_20260803.md"
_HG1 = ".omx/research/ddm_hg1_negatives_as_geometry_20260803.md + ddm_hg1_signed_depth_profile_n600.json"
_PC2 = ".omx/research/ddm_pc2_perclass_road_edges_20260802.md"
_RT2 = ".omx/research/ddm_rt2_realized_dseg_discarded_20260803.md"
_MODULES = "upstream/modules.py:112-113 (verified at source by this arm)"


def _r(**kw) -> ForceLedgerRow:
    return ForceLedgerRow(**kw)


# --------------------------------------------------------------------------- #
# THE LEDGER
# --------------------------------------------------------------------------- #
# NOTE ON DENOMINATORS (m50): coverage() computes the honest cell universe and how
# few of its cells carry evidence. UNMEASURED rows are written explicitly where a
# sign is *owed*; cells with neither evidence nor an explicit row are counted by
# coverage() as UNMEASURED-implicit.

LEDGER: tuple[ForceLedgerRow, ...] = (
    # ------------------------------------------------------------------ #
    # 0. GOVERNING LAW + STRUCTURAL FACTS
    # ------------------------------------------------------------------ #
    _r(
        row_id="law.aggregate_cashing_dead",
        scope_kind="FAMILY",
        scope="aggregate_asymmetry_cashing",
        force="any_aggregate_reweighting_of_d_seg",
        force_kind="FAMILY",
        verb="AGGREGATE",
        verdict="NEUTRAL",
        measured=True,
        evidence=(
            "7 independent instruments, every one null or harmful when an asymmetry was cashed "
            "through an aggregate: hg1 Simpson reversal; sx2 negative marginal value; pu2 r^2=0.000; "
            "#900; mf1 tau_end pooling; as1 grow_Lane_into_Road_1px +0.2459 S (all 10 directed sides "
            "p<0.5); mg1 barrier-key 0.0000% matched-byte advantage vs a shuffle control resolving "
            "5.6-65.4%. Asymmetry = PRIOR on a per-site actuator, never the actuator."
        ),
        evidence_source=f"{_HG1}; {_AS1}; {_MG1} §4; {_PU2}",
        verdict_scope="FAMILY",
        protection="N_A",
    ),
    _r(
        row_id="struct.d_seg_pricing",
        scope_kind="GLOBAL",
        scope="ALL",
        force="scorer_d_seg_pricing",
        force_kind="METRIC",
        verb="AGGREGATE",
        verdict="HARMS",
        measured=True,
        evidence=(
            "d_seg = uniform per-pixel argmax flip mean; logits discarded; no class/edge/verb term. "
            "Consequence measured on live cx1 n600: it prices Lane word-ANNIHILATION and Road "
            "rim-ERODE at the identical 1/117,964,800 per px; annihilating 26.90% of the entire Lane "
            "population costs only 0.199 S. Per-site pricing is fair (cg1r: realized flip depth "
            "parity 1.011-1.306x across all 9 edges) — the harm is exclusively verb/geometry "
            "blindness, not site mispricing."
        ),
        evidence_source=f"{_MODULES}; {_GT2}; {_CG1R}",
        verdict_scope="STRUCTURAL",
        protection="ABSENT",
        protection_ref="no per-class/per-verb term exists or can be added to the frozen scorer; protection must live in OUR objectives/guards",
    ),
    _r(
        row_id="struct.depth_parity_directed",
        scope_kind="EDGE",
        scope="Road<->Lane",
        force="scorer_d_seg_pricing",
        force_kind="METRIC",
        verb="AMPLITUDE",
        verdict="NEUTRAL",
        measured=True,
        evidence=(
            "THE DECISIVE $0 MEASUREMENT OF THIS ARM: per-site GT-margin depth of realized flips is "
            "direction-symmetric — Road<->Lane depth asym 1.074x (Lane->Road mean 0.3036 vs "
            "Road->Lane 0.3262) while count asym is 3.65x and ph4's prospective BARRIER asym is "
            "10.05x. All 9 edges: depth asym 1.011-1.633, count asym 1.27-15.88. The 10x 'discount' "
            "is NOT per-site pricing; it is realized as VOLUME (3.65x more Lane->Road flips) and as "
            "the VERB (ANNIHILATE of a 2.51-px-thin class). Independently corroborates as1's "
            "'geometric not energetic' (potential symmetric to 2%)."
        ),
        evidence_source=f"{_CG1R}; {_PH4}; {_AS1}",
        verdict_scope="INSTANCE",
        protection="N_A",
    ),
    # ------------------------------------------------------------------ #
    # 1. THE LIVE VEHICLE'S REALIZED SEG FORCES (real outcome + telemetry)
    #    force = tr1_seg_leg_composite: CE + margin hinge @0.05, scored by d_seg.
    #    Outcomes from gt2_verbs.json (verb masses), pu2 (net flows), cg1r (depths).
    # ------------------------------------------------------------------ #
    _r(
        row_id="tr1.lane.annihilate",
        scope_kind="CLASS",
        scope="Lane",
        force="tr1_seg_leg_composite",
        force_kind="OBJECTIVE",
        verb="ANNIHILATE",
        verdict="HARMS",
        measured=True,
        evidence=(
            "9,655 of 16,581 GT Lane components annihilated = 58.23% word-annihilation rate "
            "(47,226 px), vs Road 5.45% and MyCar 0.00%. Lane loses 26.90% of its own population to "
            "misclassification (Road: 0.559%) — a 48.1x per-pixel reliability disparity across a "
            "shared separatrix. ANNIHILATE:BIRTH = 10,124:616 words = 16.4x — the failure is "
            "RECALL, not precision: a force that suppresses false birth aims at the small side."
        ),
        evidence_source=_GT2,
        verdict_scope="INSTANCE",
        protection="ABSENT",
        protection_ref=(
            "the ANNIHILATE channel specifically has NO live protection: lane_guard's born_lane "
            "mask up-weights currently-WON support (a rim-peel guard — does nothing for a whole "
            "component being lost), and its lambda_Lane budget was MEASURED INERT as shipped "
            "(bs2/#871: lambda==0.0 on 64/64 burn-4 gates, budget pinned at the starting level, "
            "licensed giving back ALL 0.050213 S of won Lane). No existence-carrying primitive "
            "exists (see as1.lane_presence_gap)."
        ),
        magnitude_s=0.1575,  # Lane's total flip mass: 185,801 x 8.477e-7 S/flip
    ),
    _r(
        row_id="tr1.lane_to_road.transfer",
        scope_kind="DIRECTED_EDGE",
        scope="Lane->Road",
        force="tr1_seg_leg_composite",
        force_kind="OBJECTIVE",
        verb="TRANSFER",
        verdict="HARMS",
        measured=True,
        evidence=(
            "184,613 Lane->Road flips (36.30% of ALL flips — the single largest directed side). "
            "Edge verb split: TRANSFER 134,078 px (57.0%) > DISPLACE 101,070 px (43.0%), direction "
            "Lane->Road — net area handover, not conserved wobble (gt2 frontmatter: "
            "'boundary-displacement is REFUTED_AS_THE_DOMINANT_MODE on this edge'). Net class flow "
            "Lane -26.50% / Road +27.22% of total flips (pu2), = -19.52% of Lane's own GT population. "
            "Per-site depth of these flips equals the reverse direction (cg1r 1.074x): the erasure is "
            "bought at fair per-site prices, in volume."
        ),
        evidence_source=f"{_GT2}; {_PU2}; {_CG1R}",
        verdict_scope="INSTANCE",
        protection="BUILT_UNFIRED",
        protection_ref=(
            "the TRANSFER guard is lane_guard's lambda_Lane aggregate budget — MEASURED INERT as "
            "shipped (bs2/#871: correct KKT on a mis-specified problem; budget pinned at the "
            "starting level goes monotonically slack). The built fix, tr1_lane_guard_ratchet "
            "(spec_tr1_renderer_20260728.py:933, sigma measured online 0.00142148 S, calibrated k "
            "E[max lambda]=0.0657, detects +0.005 S where the constant budget detects nothing up "
            "to +0.050 S), has NEVER FIRED."
        ),
        magnitude_s=0.157,  # 184,613 flips / 117,964,800 * 100
    ),
    _r(
        row_id="tr1.lane.erode",
        scope_kind="CLASS",
        scope="Lane",
        force="tr1_seg_leg_composite",
        force_kind="OBJECTIVE",
        verb="ERODE",
        verdict="HARMS",
        measured=True,
        evidence=(
            "ERODE 135,683 px on surviving Lane components (73% of Lane's 185,801 flip px); GOUGE "
            "only 2,926 px — Lane has no interior to gouge (rz1: 6.92% of Lane at depth>=2 vs Road "
            "63.64%). TWO verbs, TWO protections (gt2r): by WORD count the Lane drop is ANNIHILATE "
            "(58.23%), by PIXEL count it is ERODE (73.0%) — a rim-peel guard does nothing for "
            "whole-component annihilation and vice versa."
        ),
        evidence_source=_GT2,
        verdict_scope="INSTANCE",
        protection="BUILT",
        protection_ref=(
            "lane_guard's born_lane mask (up-weights won Lane support, weighted by measured head "
            "sensitivity 1.196x) + margin-floor emphasis ARE rim-peel guards, wired at trainer "
            "L2731, default-OFF byte-identical when disabled. Caveat: the sibling lambda budget "
            "piece measured INERT (#871)."
        ),
    ),
    _r(
        row_id="tr1.lane.fragment",
        scope_kind="CLASS",
        scope="Lane",
        force="tr1_seg_leg_composite",
        force_kind="OBJECTIVE",
        verb="FRAGMENT",
        verdict="HARMS",
        measured=True,
        evidence=(
            "Component delta -6,038 (16,581 GT -> 10,543 cx1): net WORD LOSS. FRAGMENT is SIGNED "
            "(gt2r): the receiver CONSOLIDATES for 4/5 classes — a consolidation force and a "
            "fragmentation force are opposite rows. Structure-changing and can be area-neutral, "
            "hence partially invisible to d_seg."
        ),
        evidence_source=_GT2,
        verdict_scope="INSTANCE",
        protection="ABSENT",
        protection_ref="no component-count/existence guard exists on any live path",
    ),
    _r(
        row_id="tr1.road.transfer_gain",
        scope_kind="CLASS",
        scope="Road",
        force="tr1_seg_leg_composite",
        force_kind="OBJECTIVE",
        verb="TRANSFER",
        verdict="HARMS",
        measured=True,
        evidence=(
            "Road net +27.22% of total flips (+138,461 px) — but this is FALSE Road repainted over "
            "Lane/Movable/others (hallucinated area), while true Road holds well: only 0.559% of own "
            "population misclassified, ANNIHILATE 69/1,266 components (5.45%), ERODE 141,180 px. "
            "The 'gain' harms the SCENE's representation; Road's own words survive."
        ),
        evidence_source=f"{_GT2}; {_PU2}",
        verdict_scope="INSTANCE",
        protection="BUILT_UNFIRED",
        protection_ref="road_undriv_bulk_field.py (probe P-C) built, never fired on a live run",
    ),
    _r(
        row_id="tr1.undrivable.hold",
        scope_kind="CLASS",
        scope="Undrivable",
        force="tr1_seg_leg_composite",
        force_kind="OBJECTIVE",
        verb="ERODE",
        verdict="NEUTRAL",
        measured=True,
        evidence="0.128% of own population misclassified; word-annihilation 6.00% (39/650, 197 px); net flow +0.418% of flips. Best-held large class; region production fits it (area/perimeter 176.2).",
        evidence_source=_GT2,
        verdict_scope="INSTANCE",
        protection="N_A",
    ),
    _r(
        row_id="tr1.movable.gouge",
        scope_kind="CLASS",
        scope="Movable",
        force="tr1_seg_leg_composite",
        force_kind="OBJECTIVE",
        verb="GOUGE",
        verdict="HARMS",
        measured=True,
        evidence=(
            "5.40% of own population misclassified; word-annihilation 16.36% (361 comps, 8,180 px); "
            "GOUGE 16,718 px is the LARGEST gouge mass of any class (cars lose interiors, not rims); "
            "net flow -7.418% of flips. as1: Movable deficit is CONCENTRATED (p90 0.569) where Lane's "
            "is diffuse — a different failure geometry needing a different cure."
        ),
        evidence_source=f"{_GT2}; {_AS1}; {_PU2}",
        verdict_scope="INSTANCE",
        protection="ABSENT",
        protection_ref="no Movable-specific guard exists; mf1's component-native production (81.4x) is the designed cure, unbuilt on the live path",
        magnitude_s=0.0668,
    ),
    _r(
        row_id="tr1.mycar.hold",
        scope_kind="CLASS",
        scope="MyCar",
        force="tr1_seg_leg_composite",
        force_kind="OBJECTIVE",
        verb="ERODE",
        verdict="NEUTRAL",
        measured=True,
        evidence="0.054% of own population misclassified; ZERO annihilations (0/600); FRAGMENT +2. The static-hood core (#139) is the best-represented class.",
        evidence_source=_GT2,
        verdict_scope="INSTANCE",
        protection="N_A",
    ),
    _r(
        row_id="tr1.road_lane.hub",
        scope_kind="EDGE",
        scope="Road<->Lane",
        force="tr1_seg_leg_composite",
        force_kind="OBJECTIVE",
        verb="TRANSFER",
        verdict="HARMS",
        measured=True,
        evidence=(
            "235,148 flips = 46.23% of all flips = 32.21% of the whole gap (0.19934 S). Road "
            "participates in 87.48% of ALL flips (m91 hub confirmed on this corpus at 0.3pp "
            "agreement). Decompose per EDGE, never per class — a per-class table splits this one "
            "separatrix across two rows."
        ),
        evidence_source=f"{_GT2}; {_PC2}",
        verdict_scope="INSTANCE",
        protection="BUILT",
        protection_ref="lane_guard (lg1) covers the Lane side only",
        magnitude_s=0.199,
    ),
    # ------------------------------------------------------------------ #
    # 2. OBJECTIVE-COMPONENT FORCES (levers on the seg leg)
    # ------------------------------------------------------------------ #
    _r(
        row_id="mg1.margin_floor.inert",
        scope_kind="GLOBAL",
        scope="ALL",
        force="margin_floor_hinge",
        force_kind="LEVER",
        verb="AMPLITUDE",
        verdict="INERT",
        measured=True,
        evidence=(
            "REFUTED AS A LEVER AT ANY VALUE (FORMULATION scope): a site is a flip iff margin<0, so "
            "flip coverage is already 100% at the shipped floor 0.1; the relu gradient is flat "
            "(exactly one distinct non-zero magnitude at floors 0.1 and 2.0582); lim f->inf = "
            "f - mean(margin) EXACTLY (abs_diff 0.0) = the bulk objective er1 diagnosed. Raising the "
            "floor recruits only already-correct sites (at f=2.0582, 91.3% of hinge support never flips)."
        ),
        evidence_source=f"{_MG1} §1",
        verdict_scope="FORMULATION",
        protection="N_A",
    ),
    _r(
        row_id="mg1.margin_weight.derived",
        scope_kind="GLOBAL",
        scope="ALL",
        force="margin_hinge_weight",
        force_kind="LEVER",
        verb="AMPLITUDE",
        verdict="UNMEASURED",
        measured=False,
        evidence=(
            "Derived no-free-parameter: w* in [0.50, 0.80] = 10-16x the shipped 0.05 (CE-parity at "
            "the separatrix; p_target in [1/C, 1/2] at m=0). TRAINER-SCOPED per bo1's seal "
            "(07362ff659): the under-parity diagnosis applies to the CORRECTION SOLVER's CE + "
            "0.05*hinge; the burn trainer's tau_softplus form is ALREADY inside the parity band — "
            "do not port the 10-16x claim across trainers. A/B OWED (mg1 §7.1) with the RELATIVE "
            "kill criterion (1% of remaining gap = 7,301 flips) and the d_pose + L7-portability "
            "co-reports; a seg-only A/B is forbidden."
        ),
        evidence_source=f"{_MG1} §3, §7.1; bo1 07362ff659",
        verdict_scope="DERIVATION",
        protection="N_A",
    ),
    _r(
        row_id="mg1.per_site_margin_weight",
        scope_kind="GLOBAL",
        scope="ALL",
        force="per_site_margin_weight",
        force_kind="LEVER",
        verb="AMPLITUDE",
        verdict="UNMEASURED",
        measured=False,
        evidence=(
            "The named useful object: head margin (top1-top2, free) selects realized failures at 98.1x "
            "enrichment / 42.51% precision in [0,0.096) vs 0.000257% in [4,8) — 165,165x ratio, 8.5x "
            "sharper than Morse-Smale. As a per-SITE weight it is untested (needs candidate margins = "
            "a scorer pass; forbidden here without MAIN)."
        ),
        evidence_source=f"{_MG1} §2, §7.3; {_CG1} (per-class reach of the same key)",
        verdict_scope="N_A",
        protection="N_A",
    ),
    _r(
        row_id="mg1.squared_hinge.antialigned",
        scope_kind="GLOBAL",
        scope="ALL",
        force="squared_hinge",
        force_kind="LEVER",
        verb="AMPLITUDE",
        verdict="HARMS",
        measured=False,
        evidence=(
            "DERIVATION ONLY: relu(.)^2 gradient is proportional to deficit -> prioritizes the DEEPEST "
            "(most expensive) flips, while d_seg is a COUNT so the score-optimal order under scarce "
            "capacity is cheapest-first. A convex hinge is anti-aligned. Untested; not a licence."
        ),
        evidence_source=f"{_MG1} §6.3",
        verdict_scope="DERIVATION",
        protection="N_A",
    ),
    _r(
        row_id="rt2.realized_dseg.inert",
        scope_kind="GLOBAL",
        scope="ALL",
        force="realized_dseg_term",
        force_kind="OBJECTIVE",
        verb="AGGREGATE",
        verdict="INERT",
        measured=True,
        evidence="rt2 EXECUTED adding the realized d_seg to the objective: gradient identically zero (argmax discards logits). Forbidden cure #4 in mg1 §6.",
        evidence_source=f"{_RT2}; {_MG1} §6",
        verdict_scope="FORMULATION",
        protection="N_A",
    ),
    # ------------------------------------------------------------------ #
    # 3. ALLOCATOR FORCES
    # ------------------------------------------------------------------ #
    _r(
        row_id="wr1.flip_count_key.lane",
        scope_kind="CLASS",
        scope="Lane",
        force="wr1_766_waterfill_flip_count_key",
        force_kind="ALLOCATOR",
        verb="ANNIHILATE",
        verdict="HARMS",
        measured=True,
        evidence=(
            "The #766 drop key is flip-count-first (wr1:93 lexsort((-residual_mass, flip_mass))). "
            "d_seg prices Lane erasure and Road boundary-nudge identically while their prospective "
            "barriers differ 11.4x (ph4/hg1: Road->Lane 51.26 vs Lane->Road 5.10, the minimum of all "
            "11 sides) — a flip-count-driven allocator is structurally paid to erase lanes at a "
            "discount. REFINED by cg1r: the discount is NOT per-site (realized depth parity 1.074x); "
            "it is the verb channel — a whole Lane word costs ~2.5 px of depth to erase (no interior), "
            "so equal flip budgets buy ANNIHILATE on Lane and only DISPLACE on Road."
        ),
        evidence_source=f"{_PH4}; {_HG1}; {_CG1R}; {_GT2} §6",
        verdict_scope="FORMULATION",
        protection="ABSENT",
        protection_ref="no verb-aware or existence-aware term in the #766 key; lane_guard does not gate the waterfill",
    ),
    _r(
        row_id="mg1.barrier_key.neutral",
        scope_kind="GLOBAL",
        scope="ALL",
        force="barrier_integral_rank_key",
        force_kind="ALLOCATOR",
        verb="AMPLITUDE",
        verdict="NEUTRAL",
        measured=True,
        evidence=(
            "hg1 F4 REFUTED per its own pre-registered kill: barrier-key advantage 0.0000% of damage "
            "out to 30% of bytes freed (bit-identical drop order for the first 384/768 cells), against "
            "a shuffle control resolving 5.6-65.4% — the instrument is ~1000x more sensitive than the "
            "effect. Mechanism: barrier is a per-SIDE constant and flips are spatially segregated by "
            "side, so barrier mass ~ constant x flip count."
        ),
        evidence_source=f"{_MG1} §4",
        verdict_scope="INSTANCE",
        protection="N_A",
    ),
    _r(
        row_id="mg1.per_class_pair_scalar.dead",
        scope_kind="FAMILY",
        scope="per_class_pair_scalar_reweight",
        force="per_class_pair_scalar_reweight",
        force_kind="FAMILY",
        verb="AGGREGATE",
        verdict="NEUTRAL",
        measured=True,
        evidence=(
            "The WHOLE family 're-weight the flip count by a per-class-pair scalar' is dead by the "
            "§4 mechanism (per-side constant x segregated flips = near-monotone rescaling; rho 0.90-0.95 "
            "across grains 1-256 px with matched-byte damage unmoved). Repair cost is per-SITE (|m|); "
            "any per-side average destroys it."
        ),
        evidence_source=f"{_MG1} §4.3",
        verdict_scope="FAMILY",
        protection="N_A",
    ),
    # ------------------------------------------------------------------ #
    # 4. ACTUATOR PROBES (measured one-shot forces)
    # ------------------------------------------------------------------ #
    _r(
        row_id="as1.grow_lane.harms",
        scope_kind="DIRECTED_EDGE",
        scope="Road->Lane",
        force="grow_lane_into_road_1px",
        force_kind="ACTUATOR_PROBE",
        verb="TRANSFER",
        verdict="HARMS",
        measured=True,
        evidence=(
            "Growing Lane into Road by 1px = +0.2459 S; ALL 10 directed grow probes p<0.5 (no "
            "directed aggregate growth helps). The Lane deficit cannot be repaired by aggregate "
            "re-inflation — instrument #6 of the governing law. Lane asymmetry is GEOMETRIC not "
            "energetic (potential symmetric to 2%, domain 7:1)."
        ),
        evidence_source=_AS1,
        verdict_scope="FORMULATION",
        protection="N_A",
        magnitude_s=0.2459,
    ),
    # ------------------------------------------------------------------ #
    # 5. PRODUCTION FORCES (how a class is representable at all)
    # ------------------------------------------------------------------ #
    _r(
        row_id="gt2.region_paint.lane",
        scope_kind="CLASS",
        scope="Lane",
        force="region_paint_production",
        force_kind="PRODUCTION",
        verb="ANNIHILATE",
        verdict="HARMS",
        measured=True,
        evidence=(
            "REFUTED AT FAMILY SCOPE for Lane: area/perimeter = 1.407 (vs Undrivable 176.2) makes any "
            "region production 125x less efficient on Lane — a property of the class geometry, not an "
            "implementation. Lane at 2.51 px median minor axis has no interior; a production that "
            "resolves contests by area loses all of them at once (71/102 components annihilated in "
            "the smoke; 58.23% word rate at n600)."
        ),
        evidence_source=_GT2,
        verdict_scope="FAMILY",
        protection="DESIGNED",
        protection_ref="gt2 §2.4: erasure-immune productions = stroke (cost ~ length) or component-existence symbol",
    ),
    _r(
        row_id="gt2.region_paint.undrivable",
        scope_kind="CLASS",
        scope="Undrivable",
        force="region_paint_production",
        force_kind="PRODUCTION",
        verb="DISPLACE",
        verdict="IMPROVES",
        measured=True,
        evidence="Region production delivers 176.2 px per boundary px on Undrivable (125x better than on Lane) — the right production for the region-like classes (Undrivable/MyCar/Road: 176.2/97.4/21.9).",
        evidence_source=_GT2,
        verdict_scope="FAMILY",
        protection="N_A",
    ),
    _r(
        row_id="gt2.stroke_production.lane",
        scope_kind="CLASS",
        scope="Lane",
        force="stroke_curve_production",
        force_kind="PRODUCTION",
        verb="ANNIHILATE",
        verdict="UNMEASURED",
        measured=False,
        evidence=(
            "DESIGNED erasure-immune for Lane (existence decided by the program, not by area contest; "
            "cost scales with length, aspect 25.48). gt2_lane.json prices the polyline program: "
            "eps=0.5px -> 129,864 B lzma9e with 24,048 err px (0.0204 S if all became flips) vs "
            "G_raster 189,460 B. Sign on realized d_seg through a real byte-close: UNMEASURED."
        ),
        evidence_source=f"{_GT2}; /Volumes/VertigoDataTier/pact/ddm_gt2_20260803/gt2_lane.json",
        verdict_scope="N_A",
        protection="N_A",
    ),
    _r(
        row_id="mf1.component_production.movable",
        scope_kind="CLASS",
        scope="Movable",
        force="component_existence_production",
        force_kind="PRODUCTION",
        verb="ANNIHILATE",
        verdict="IMPROVES",
        measured=True,
        evidence=(
            "mf1: component-native production on Movable (2.47 components/frame) measured 81.4x "
            "better than W. A dropped component becomes a missing SYMBOL, detectable at encode time — "
            "where a dropped region is silently absorbed. The designed cure for the ANNIHILATE channel."
        ),
        evidence_source=f"{_GT2} §2.4 (citing mf1); mf1 40f26020d3",
        verdict_scope="INSTANCE",
        protection="N_A",
    ),
    _r(
        row_id="pc2.shared_grid_tokens.lane",
        scope_kind="CLASS",
        scope="Lane",
        force="shared_grid_token_production",
        force_kind="PRODUCTION",
        verb="ANNIHILATE",
        verdict="HARMS",
        measured=True,
        evidence=(
            "err_rate ~ 2.79e-4 * area^(-1.258), r(log,log) = -0.934 (n=5, INSTANCE regularity): "
            "Lane occupies 1.5 of the 256 px a 16x16 token cell describes with 4 numbers -> averaged "
            "toward the majority, loses the argmax (within-class err 25.72% vs Undrivable 0.101%). "
            "No affordable grid refinement rescues a 0.585%-area class (ds=8 quadruples the token "
            "member and still leaves ~6 px/cell)."
        ),
        evidence_source=_PC2,
        verdict_scope="FORMULATION",
        protection="DESIGNED",
        protection_ref="SPEC_v8 §2 'Lane1 = analytic band ~1-2 KB' (the only affordable option); analytic_lane_render_band BUILT 101.4KB, ABSENT from live tr1_config.json",
    ),
    _r(
        row_id="pc2.shared_grid_tokens.mycar",
        scope_kind="CLASS",
        scope="MyCar",
        force="shared_grid_token_production",
        force_kind="PRODUCTION",
        verb="DISPLACE",
        verdict="IMPROVES",
        measured=True,
        evidence=(
            "MyCar sits 0.23x BELOW the area law fit — the one class with a structure-matched "
            "carrier (static hood, tokens_base, IoU 0.994) beats the law. A working in-vehicle "
            "existence proof for the per-class-carrier thesis."
        ),
        evidence_source=_PC2,
        verdict_scope="INSTANCE",
        protection="N_A",
    ),
    _r(
        row_id="pc2.tr1_phase_dof.road_lane",
        scope_kind="EDGE",
        scope="Road<->Lane",
        force="tr1_seg_leg_composite",
        force_kind="OBJECTIVE",
        verb="PHASE",
        verdict="HARMS",
        measured=True,
        evidence=(
            "tokens_delta (98.63% of parameters, 96.2% of scored bytes) is 4 numbers per 16x16 cell: "
            "per-pair AMPLITUDE DOF with NO per-pair positional DOF. Road<->Lane is the most "
            "flicker-typed edge (57.6% = 0.11025 S on pixels whose GT label changes between frames) "
            "and least tie-like (30.6% near-tie) — cannot be bought with a bias. Phase-faithfulness "
            "debt and the missing contour carrier are ONE deficit."
        ),
        evidence_source=_PC2,
        verdict_scope="INSTANCE",
        protection="ABSENT",
        protection_ref="no contour DOF anywhere in the live representation (49 config keys: lane_render_band/per_class/carrier all ABSENT)",
        magnitude_s=0.110,
    ),
    _r(
        row_id="pc2.tie_calibration",
        scope_kind="EDGE",
        scope="Road<->Undrivable",
        force="per_edge_tie_calibration",
        force_kind="LEVER",
        verb="AMPLITUDE",
        verdict="IMPROVES",
        measured=True,
        evidence=(
            "ru1 tier-1 measured the per-atlas version: 11.9% of flips, dS -0.046 at ~0 B, +yield in "
            "17/18 cells. Road<->Undriv and Road<->MyCar are 45.4%/45.7% near-tie (runner-up within "
            "0.25 logits on a boundary already in the right place) — the tie-calibration edges. "
            "Required RGB direction differs per edge: edge-resolve the existing sign rule."
        ),
        evidence_source=f"{_PC2}; ddm_ru1 tier-1",
        verdict_scope="INSTANCE",
        protection="N_A",
        magnitude_s=0.046,
    ),
    # ------------------------------------------------------------------ #
    # 6. PER-SIDE PREDICTORS / WEIGHT FAMILIES (hg1 + as1)
    # ------------------------------------------------------------------ #
    _r(
        row_id="hg1.sr_field.chance",
        scope_kind="GLOBAL",
        scope="ALL",
        force="sR_signed_reachability_field",
        force_kind="LEVER",
        verb="AMPLITUDE",
        verdict="NEUTRAL",
        measured=True,
        evidence=(
            "At chance as a flip predictor on every major directed side (|rho_sR| <= 0.0808, mean "
            "0.0250) while the FREE cached margin scores |rho| 0.2326-0.3787 with correct sign "
            "everywhere — 3-4x worse at 450 MB. H2 (misindexed) refuted at 2.9 sigma; H1 (genuinely "
            "chance) stands. Instrument caveat: its declared positive_controls metadata was NEVER "
            "read by the verdict code (silent-guard, m50 vacuity at one remove)."
        ),
        evidence_source=f"{_HG1} (q1 correlator, witness 96f, 6bd8ddd86e)",
        verdict_scope="INSTANCE",
        protection="N_A",
    ),
    _r(
        row_id="hg1.b16_density_hinge",
        scope_kind="DIRECTED_EDGE",
        scope="Lane->Road",
        force="density_weighted_signed_hinge_b16",
        force_kind="DERIVED_FORCE",
        verb="ANNIHILATE",
        verdict="HARMS",
        measured=False,
        evidence=(
            "ANTI-BUILD by derivation: the density asymmetry runs the wrong way — a density hinge "
            "would DE-weight the Lane side because margin_d1=0.593 looks expensive, exactly where a "
            "bounded perturbation annihilates the class (Lane->Road is the ONLY truncating side, "
            "barrier 5.10 vs 33.26-58.39, yet steepest per-px margin recovery of all 14 sides). "
            "Correct primitive = extent/existence term, not a signed cost hinge."
        ),
        evidence_source=f"{_HG1} §9.2",
        verdict_scope="DERIVATION",
        protection="N_A",
    ),
    _r(
        row_id="as1.depth_weight_rule",
        scope_kind="FAMILY",
        scope="per_side_weighting",
        force="per_side_weight_any",
        force_kind="FAMILY",
        verb="AMPLITUDE",
        verdict="UNMEASURED",
        measured=False,
        evidence=(
            "BINDING DESIGN RULE (positive dual of hg1's negative): any per-side weight must be a "
            "function of AVAILABLE DEPTH (a domain quantity), NEVER of the margin field (an energy "
            "quantity) — the potential is symmetric to ~2% across the separatrix (mf1) while the "
            "domain is 7:1 (Road reach 7 px vs Lane 1 px; shell 75.04% vs <=9.7%). The Lane "
            "asymmetry is ENTIRELY GEOMETRIC. Corroborated at realized-flip level by cg1r: per-site "
            "depth parity 1.011-1.633x across all 9 edges while count asym runs 1.27-15.88x."
        ),
        evidence_source=f"{_AS1} §2.1; {_HG1}; {_CG1R}",
        verdict_scope="N_A",
        protection="DESIGNED",
        protection_ref="depth-weight-not-margin-weight rule binds mg1's hinge-weight A/B (as1 §8.3)",
    ),
    _r(
        row_id="law.asymmetry_vehicle_transfer",
        scope_kind="FAMILY",
        scope="per_side_asymmetry_priors",
        force="directed_asymmetry_as_prior",
        force_kind="FAMILY",
        verb="AGGREGATE",
        verdict="NEUTRAL",
        measured=True,
        evidence=(
            "VALIDITY LAW, not a sign: directed asymmetries are VEHICLE-DEPENDENT — Road<->Undriv "
            "count asym 1.038x (witness n96) -> 1.31x (tb1 ep399) -> 1.681x (live cx1); "
            "Road<->Movable 1.106x -> 1.47x -> 1.606x. hg1's 'asymmetry is localized to Lane' is a "
            "witness-vehicle fact that does NOT transfer; all five major edges earn per-side rows on "
            "the live vehicle. Also: rate ratio != count ratio != flip-mass share (hg1 Road<->Lane "
            "5.85x rate = 2.19x count; annulus imbalance 2.67x is the cause). Every per-side row "
            "must state its vehicle and its ratio kind."
        ),
        evidence_source=f"{_AS1} §2.4; {_HG1}; {_CG1R}",
        verdict_scope="FAMILY",
        protection="N_A",
    ),
    # ------------------------------------------------------------------ #
    # 7. ACTUATORS AND CARRIERS (as1's measured probes + the named gaps)
    # ------------------------------------------------------------------ #
    _r(
        row_id="as1.class_prior_logit_shift",
        scope_kind="GLOBAL",
        scope="ALL",
        force="class_prior_logit_shift",
        force_kind="ACTUATOR_PROBE",
        verb="TRANSFER",
        verdict="HARMS",
        measured=True,
        evidence=(
            "src/tac/mask_prior.py:144 apply_prior_weighting (logits += alpha*log(prior)) — UNWIRED "
            "(sole importer is its own test) and now MEASURED DEAD in unconditional form: F-ACT-1 is "
            "the test of exactly this mechanism (+0.2459 S). Gated-at-16x16 form bounded at "
            "-0.001678 S total. Orphan closed with a measured disposition, not a queue slot."
        ),
        evidence_source=f"{_AS1} §5.1a",
        verdict_scope="FORMULATION",
        protection="N_A",
        magnitude_s=0.2459,
    ),
    _r(
        row_id="as1.gated_area_move.movable",
        scope_kind="CLASS",
        scope="Movable",
        force="gated_area_move_16px",
        force_kind="ACTUATOR_PROBE",
        verb="TRANSFER",
        verdict="IMPROVES",
        measured=True,
        evidence=(
            "F-CELL-1: Movable's deficit is spatially CONCENTRATED enough to clear break-even — 7 "
            "cells p>0.5 into Road (-0.001060 S) + 13 into Undrivable (-0.000511 S); p90 = 0.569 on "
            "Movable->Undriv (the 90th-percentile cell is already profitable). Small but real; "
            "closes 16x16, does NOT close finer cells or per-component."
        ),
        evidence_source=f"{_AS1} §8.2 (200 SPREAD frames, subset ratio 0.9986)",
        verdict_scope="INSTANCE",
        protection="N_A",
        magnitude_s=0.0016,
    ),
    _r(
        row_id="as1.gated_area_move.lane",
        scope_kind="CLASS",
        scope="Lane",
        force="gated_area_move_16px",
        force_kind="ACTUATOR_PROBE",
        verb="TRANSFER",
        verdict="NEUTRAL",
        measured=True,
        evidence=(
            "Lane's deficit is spatially DIFFUSE at 16x16: only 3 profitable cells / 160 candidates, "
            "median cell p = 0.212, p95 = 0.425 < 0.5 (-0.000071 S). Flip-mass concentration != "
            "precision concentration — hs1's 91.22%-at-62B is mass, not precision; do not let one "
            "row's evidence license the other's verb."
        ),
        evidence_source=f"{_AS1} §8.2, A15",
        verdict_scope="INSTANCE",
        protection="N_A",
    ),
    _r(
        row_id="as1.displacement_carrier.lane",
        scope_kind="CLASS",
        scope="Lane",
        force="displacement_carrier",
        force_kind="PRODUCTION",
        verb="DISPLACE",
        verdict="NEUTRAL",
        measured=True,
        evidence=(
            "Structurally blind to 56.92% of Lane's flip mass: deletion is area-changing and "
            "translation is area-preserving. CLOSED for Lane (as1 §8.3). For Movable the ceiling is "
            "<=68.5% reachable — a correction to the ceiling, not a kill. The carrier does not exist "
            "as code anywhere (10,260 .py sweep); sx2 refuted the integer-offset form (displacement "
            "is sub-pixel PHASE, d ~ 0.65 px, 76% of the mixture)."
        ),
        evidence_source=f"{_AS1} S2/S5; ddm_sx2 (.omx/research/ddm_sx2_g2_gate_and_displacement_carrier_20260803.md)",
        verdict_scope="FORMULATION",
        protection="N_A",
    ),
    _r(
        row_id="as1.lane_presence_gap",
        scope_kind="CLASS",
        scope="Lane",
        force="lane_presence_existence_carrier",
        force_kind="PRODUCTION",
        verb="ANNIHILATE",
        verdict="UNMEASURED",
        measured=False,
        evidence=(
            "THE GAP THE FOUR-WAY LANE LAW NAMES (S5): Lane is a 1-px ribbon with no interior — no "
            "region to paint, no depth to displace; the vehicle DELETES it (-134,786 px net) and "
            "deletion is invisible to displacement carriers and region-paint alike. Lane survives "
            "only where ADDRESSED, and it is addressable cheaply because it is spatially static. NO "
            "shipped primitive addresses an area-changing defect on a no-interior class (10,260-file "
            "sweep). dd1 owns Lane component decomposability; gt2 prices the stroke production."
        ),
        evidence_source=f"{_AS1} §4 S5; {_GT2} §2.4",
        verdict_scope="N_A",
        protection="ABSENT",
        protection_ref="no existence-carrying primitive for Lane exists on any path; the highest-ranked protection debt in this ledger",
    ),
    _r(
        row_id="as1.directional_entropy_prior",
        scope_kind="GLOBAL",
        scope="ALL",
        force="directional_entropy_prior",
        force_kind="ALLOCATOR",
        verb="TRANSFER",
        verdict="UNMEASURED",
        measured=False,
        evidence=(
            "The one aggregate use of asymmetry that is NOT in the dead family, because entropy "
            "coding acts per element: 5 per-edge direction priors (p_major 0.6163-0.7851) are worth "
            "a MEASURED 84,520 bits = 10,565 B = 1.137% of the gap at 0.0208 B/flip. Absorption "
            "UNMEASURED: needs a direction-coding field; r7's per-context coder exists but is "
            "dormant on the live path. Free-vs-counted call is wf2's."
        ),
        evidence_source=f"{_AS1} §4 S1",
        verdict_scope="N_A",
        protection="N_A",
    ),
    _r(
        row_id="as1.road_undriv_bulk_field",
        scope_kind="EDGE",
        scope="Road<->Undrivable",
        force="road_undriv_bulk_field",
        force_kind="GUARD",
        verb="TRANSFER",
        verdict="UNMEASURED",
        measured=False,
        evidence=(
            "Per-EDGE directional carrier (one signed phi_bulk lifted to two per-class SDF channels "
            "by signed per-side scales; argmax parity proved exact for any s_R, s_U > 0). PREMISE "
            "REACTIVATED by as1 §2.4: hg1's witness-vehicle 1.05x symmetry does not hold on the live "
            "vehicle (1.681x). research_only=True, gated on probe P-C, UNRUN. 'The single most "
            "concrete positive lead the arm produced.'"
        ),
        evidence_source=f"{_AS1} §5.1a; src/tac/boundary_math/road_undriv_bulk_field.py",
        verdict_scope="N_A",
        protection="BUILT_UNFIRED",
        protection_ref="probe P-C unrun; carrier never fired on a live run",
    ),
    # ------------------------------------------------------------------ #
    # 8. ALLOCATOR SUPPORT/INDEX FORCES (rs2 + hs1 + sx2) — rs2's rows are
    #    GLOBAL: its memo carries NO class/edge decomposition (stated).
    # ------------------------------------------------------------------ #
    _r(
        row_id="rs2.wr1_tile_support",
        scope_kind="GLOBAL",
        scope="ALL",
        force="wr1_tile_support_pricing",
        force_kind="ALLOCATOR",
        verb="AGGREGATE",
        verdict="HARMS",
        measured=True,
        evidence=(
            "The #766 waterfill prices a cell drop over its own 256-px tile while the measured "
            "receptive field through the real receiver is 84x82 px (support ratio 24.19x; the key "
            "sees 4.13% of the real footprint; median RF area 7,056 px over 49/384 cells). "
            "One-sided and structural: only FALSE SAFETY — 144 of 486 'provably safe' knee-A cells "
            "(29.6%) are not safe. Class/edge granularity: ABSENT in rs2 (honest UNMEASURED)."
        ),
        evidence_source=".omx/research/ddm_rs2_flip_damage_rerank_and_drop_seg_leg_20260803.md (c423c153e4)",
        verdict_scope="INSTANCE",
        protection="DESIGNED",
        protection_ref="corrected box-sum key written (rf_half=0 reduces exactly to wr1's bincount) but deliberately unlanded: sister session holds uncommitted edits consuming wr1's drop_rank",
    ),
    _r(
        row_id="rs2.drop_more_beyond_knee",
        scope_kind="GLOBAL",
        scope="ALL",
        force="drop_more_beyond_knee",
        force_kind="ALLOCATOR",
        verb="AGGREGATE",
        verdict="HARMS",
        measured=True,
        evidence=(
            "Twice-corroborated: dropping past the cell_drop50 knee realizes 0.63-0.65 B/flip vs "
            "W = 1.2731 (dominated 1.96-2.02x; ba31 n600 + gr1 cell sweep, with the measured n48 -> "
            "n600 understatement of 9.2% applied)."
        ),
        evidence_source=".omx/research/ddm_rs2_flip_damage_rerank_and_drop_seg_leg_20260803.md",
        verdict_scope="INSTANCE",
        protection="N_A",
    ),
    _r(
        row_id="rs2.gr1_gsum_key",
        scope_kind="GLOBAL",
        scope="ALL",
        force="gr1_gsum_gradient_key",
        force_kind="ALLOCATOR",
        verb="AGGREGATE",
        verdict="UNMEASURED",
        measured=False,
        evidence=(
            "The allocator that CHOSE the live base. Support-correct by construction (backprop), "
            "n600 (reproduces exactly from gr1_sensitivity_gabs.npy). Known blindnesses, stated: "
            "POSE-BLIND (compute_pose=False) and a smoothed surrogate (tau_softplus) — a third proxy "
            "layer; computed on the ANCESTOR (pre-drop) lattice. Per-class/per-edge consequences "
            "of its choices: UNMEASURED (this ledger's tr1.* rows are the realized outcome)."
        ),
        evidence_source=".omx/research/ddm_rs2_flip_damage_rerank_and_drop_seg_leg_20260803.md §3",
        verdict_scope="N_A",
        protection="N_A",
    ),
    _r(
        row_id="hs1.per_pair_seg_carrier",
        scope_kind="FAMILY",
        scope="per_pair_seg_selection",
        force="per_pair_seg_correction_carrier",
        force_kind="FAMILY",
        verb="AGGREGATE",
        verdict="NEUTRAL",
        measured=True,
        evidence=(
            "FAMILY dead: the PAIR index for seg over every form tested (raw per-pair debt, per-pair "
            "cell selection, static-mask + exception list). seg x PAIR Gini 0.0858 vs pose 0.8271 "
            "(9.6x flatter, 15,500x flatter max/min); top-6 pairs = 1.66% of seg debt vs 62.01% of "
            "pose. Mechanism: 93.86% of flips sit ON the GT boundary and total boundary length is "
            "frame-invariant. Do not build a per-pair seg correction carrier."
        ),
        evidence_source=".omx/research/ddm_hs1_heterogeneous_storage_20260803.md",
        verdict_scope="FAMILY",
        protection="N_A",
    ),
    _r(
        row_id="hs1.static_cell_index",
        scope_kind="GLOBAL",
        scope="ALL",
        force="static_spatial_cell_index",
        force_kind="ALLOCATOR",
        verb="AGGREGATE",
        verdict="IMPROVES",
        measured=True,
        evidence=(
            "seg debt is STATIC-SPATIAL: cell Gini 0.8581, 63.3% of cells hold zero flips, top-128 "
            "cells capture 91.22% for 62 B (vs per-pair selection: more capture, 379x the address). "
            "sx2's 49 B static per-cell prior removes 57.2% of the 36,798 B map term = 430x exchange "
            "rate; held-out (odd frames) 49.5%. CAVEAT (same index, opposite verdicts): the same "
            "CELL index used to DROP payload realizes 32.53 B/flip = 25.5x WORSE than W (qd1) — "
            "adding a shared description != removing per-pair payload."
        ),
        evidence_source=".omx/research/ddm_hs1_heterogeneous_storage_20260803.md; ddm_sx2 (49 B prior)",
        verdict_scope="FORMULATION",
        protection="N_A",
    ),
    _r(
        row_id="sx2.free_extractor_blend",
        scope_kind="GLOBAL",
        scope="ALL",
        force="free_generic_extractor_blend",
        force_kind="ALLOCATOR",
        verb="AGGREGATE",
        verdict="HARMS",
        measured=True,
        evidence=(
            "Canny's marginal contribution ON TOP of the 49 B counted static prior is NEGATIVE: "
            "union 49.1% < 57.2% static alone; every blend below the better part. Counted-but-tiny "
            "beats free-but-generic; do not blend a coarse free predictor into a precise counted "
            "one. Also: rho>0.5 gate PASSES (canny 0.6210) while the conclusion it gated is REFUTED "
            "(28.1% at best budget; rho and free-fraction anti-correlate above budget) — a gate "
            "metric that can be bought while destroying what it certifies."
        ),
        evidence_source=".omx/research/ddm_sx2_g2_gate_and_displacement_carrier_20260803.md",
        verdict_scope="FORMULATION",
        protection="N_A",
    ),
    _r(
        row_id="conflict.d_null_space",
        scope_kind="GLOBAL",
        scope="ALL",
        force="scorer_D_null_space",
        force_kind="METRIC",
        verb="AGGREGATE",
        verdict="NEUTRAL",
        measured=True,
        evidence=(
            "CROSS-MEMO CONFLICT, encoded so it cannot recur: D's 80.674% null fraction lives on the "
            "CAMERA plane (1,017,336-dim) and is STRUCTURALLY UNREACHABLE from the token lattice — "
            "M = D.U has gain range [0.6866, 1.0283], 0.0% of render directions attenuated below "
            "1e-3 (rs2, operator validated at 1.7e-07 relative). hs1 cites the same fraction as a "
            "live constraint on capture realizability — valid ONLY for camera-plane mechanisms "
            "(e.g. paint-on-camera carriers), NEVER for token-lattice mechanisms."
        ),
        evidence_source=".omx/research/ddm_rs2_...md §5.2 vs .omx/research/ddm_hs1_...md §5.2",
        verdict_scope="STRUCTURAL",
        protection="N_A",
    ),
    # ------------------------------------------------------------------ #
    # 9. TR1-REACHABLE, NEVER-FIRED LEVERS (pt2) — duty-to-measure queue
    # ------------------------------------------------------------------ #
    *(
        _r(
            row_id=f"pt2.{name}",
            scope_kind="GLOBAL",
            scope="ALL",
            force=name,
            force_kind="LEVER",
            verb="AMPLITUDE",
            verdict="UNMEASURED",
            measured=False,
            evidence=evidence,
            evidence_source=".omx/research/ddm_pt2_lever_port_to_tr1_20260803.md (08a472aa29)",
            verdict_scope="N_A",
            protection="N_A",
        )
        for name, evidence in (
            (
                "seg_focal_gamma",
                "Ported to TR1 (--seg-focal-gamma -> seg_pixel_w, all forms; closed-form ratio "
                "verified rel err 1.73e-05). DSL Lever lever_seg_focal_gamma, default-OFF, "
                "never-fired. Targets concentration — against pc2's measured ~0.058% interior flip "
                "share, the focal re-weighting acts almost entirely on the separatrix.",
            ),
            (
                "fisher_density_weight",
                "Ported to TR1 (--fisher-density-weight/--fisher-density-source; exact sech^2 law, "
                "rel err 4.57e-07). DSL Lever lever_fisher_density, default-OFF, never-fired.",
            ),
            (
                "head_natural_grad",
                "Ported to TR1 (--head-natural-grad/-eps; fwd diff 0.0, grad diff 6.10e-06). DSL "
                "Lever lever_head_natural_grad, default-OFF, never-fired. Own-optimum caveat: "
                "re-tune lr per arm.",
            ),
            (
                "tau_softplus_tau",
                "The live seg form's own scalar (--tau-softplus-tau, inherited default 0.3). "
                "Governs ~100% of the live lineage's epochs and has NEVER been swept on this "
                "vehicle. lever_tau_softplus_tau. Separatrix share monotone 1.000->0.5007 across "
                "the sweep range.",
            ),
        )
    ),
    # ------------------------------------------------------------------ #
    # 10. #382 sigma_cc' — the charter's adjudication row
    # ------------------------------------------------------------------ #
    _r(
        row_id="adjud.sigma_ccprime_382",
        scope_kind="GLOBAL",
        scope="ALL",
        force="perclass_pair_surface_tension_sigma_ccprime",
        force_kind="DERIVED_FORCE",
        verb="DISPLACE",
        verdict="UNMEASURED",
        measured=False,
        evidence=(
            "ADJUDICATED against mg1's dead family, per the charter: sigma_cc' is a per-CLASS-PAIR "
            "(edge) coefficient on a per-SITE boundary-length gradient IN TRAINING — NOT a per-side "
            "scalar reweighting of a flip-count RANKING (mg1 §4.3's dead mechanism is side-averaging "
            "inside an allocator key). So it is NOT automatically dead. Status (sigma_ccprime_build_"
            "20260709.md): BUILT AND FULLY WIRED on the RETIRED levelset trainer only "
            "(--length-sigma-matrix, levelset:19608; grep on TR1 = 0 — a GUARDED exclusion, "
            "test_spec_v9_cgauge.py:109), NEVER RACED. The two derivation laws DISAGREE on the "
            "decisive pair: Young's-angle sigma[Road-Lane]=0.377 (lower tension -> anti-erosion) vs "
            "fragility 1.029 (not lowered) — the A/B (fitted vs fragility vs all-ones, n600 through "
            "R) is the owed arbiter. Constraints when it fires on TR1: (a) per-PAIR is the only "
            "correct form (five edges differ 1.5x in near-tie, 25x in near_3px — pc2); (b) TR1's "
            "erasure is IMPLICIT (capacity/quantization/conv smoothing), so sigma enters as a "
            "COUNTER-weight on fragile pairs, not a multiplier on an explicit length term — same "
            "5x5 matrix, INVERTED sign convention (fh1 F-B); "
            "(c) depth/domain-derived coefficients, never margin-derived (as1 §2.1); (d) fire WITH "
            "rung 2, never before; POOLS LAW: race, never stack."
        ),
        evidence_source=f".omx/research/sigma_ccprime_build_20260709.md (3571e5b65); {_PC2} §3 rung 3; {_MG1} §4.3; {_AS1} §2.1; ddm_fh1 F-B",
        verdict_scope="N_A",
        protection="N_A",
    ),
    # ------------------------------------------------------------------ #
    # 11. THE GUARDS THEMSELVES, ACCOUNTED (the protection column's own rows)
    # ------------------------------------------------------------------ #
    _r(
        row_id="guard.lane_guard_lambda_inert",
        scope_kind="CLASS",
        scope="Lane",
        force="lane_guard_lambda_primal_dual",
        force_kind="GUARD",
        verb="TRANSFER",
        verdict="INERT",
        measured=True,
        evidence=(
            "The guard itself, measured (bs2/#871, 64 lane_guard gate rows over burn-4 windows "
            "01-03): lambda_lane == 0.0 on 64/64 gates, g < 0 on 64/64, budget_s_units pinned at "
            "one value (0.12589) while realized Lane descended 0.122438 -> 0.072225 — correct KKT "
            "on a mis-specified problem; a budget pinned at the STARTING level goes monotonically "
            "slacker as the primal wins. It licensed giving back ALL 0.050213 S of won Lane. The "
            "ratchet lever (budget(t) = min(budget(t-1), mean(last m)+k*sigma), sigma online "
            "0.00142148 S, calibrated k) is the built, never-fired fix. FLAGGED, not inherited: "
            "lane_guard's EROSION_S_MEASURED=0.00151 is LADDER CLASS 4 UNVERIFIED — it appears in "
            "no machine-readable artifact (all 23 xp1_verdict.json keys checked); it scales only "
            "the dual step rate, not the budget."
        ),
        evidence_source="ddm_bs2/#871 via src/tac/witness_dsl/spec_tr1_renderer_20260728.py:933-968; src/tac/optimization/lane_guard.py",
        verdict_scope="INSTANCE",
        protection="BUILT_UNFIRED",
        protection_ref="tr1_lane_guard_ratchet lever built, never fired",
    ),
    _r(
        row_id="fh1.class_weight_lane",
        scope_kind="CLASS",
        scope="Lane",
        force="class_weight_lane",
        force_kind="LEVER",
        verb="ANNIHILATE",
        verdict="UNMEASURED",
        measured=False,
        evidence=(
            "--class-weight-lane EXISTS as a wired DSL lever (spec_tr1_renderer:80, default 1.0) "
            "and has NEVER FIRED — fh1 R6 names it 'the default-off-is-orphaned-signal case'. "
            "Derived race start: per-class birth weight ~ (P/A)_c with (P/A)_Lane/(P/A)_Movable = "
            "8.9 (GT geometry). Pool LANE-BIRTH (0.042 S measured, QA92); competes with KD-from-"
            "birth, never stacks. $0 to race."
        ),
        evidence_source=".omx/research/ddm_fh1_forces_harvest_20260731.md R6/A6",
        verdict_scope="N_A",
        protection="N_A",
    ),
    _r(
        row_id="fh1.tie_locus_edge_weighted",
        scope_kind="FAMILY",
        scope="per_edge_loss_weighting",
        force="tie_locus_edge_weighted_loss",
        force_kind="DERIVED_FORCE",
        verb="DISPLACE",
        verdict="UNMEASURED",
        measured=False,
        evidence=(
            "TR1's live loss is PAIR-BLIND: _live_margin_weight weights by margin MAGNITUDE only — "
            "no per-class-PAIR structure anywhere in the loss graph, while d_seg's Gamma-limit has "
            "per-pair structure (fh1 F-B) and the five edges differ 1.5-25x in type (pc2). "
            "TieLocusEdgeWeighted (v9 Force 3+4, W raced in {uniform, flip-mass, Young's-sigma}) is "
            "a DESIGNED-STUB: fh1_tie_locus_edge_weighted, flags exist on NEITHER trainer "
            "(fail-closed by never-invent-flags, which is the point). Pool PLACEMENT; the one "
            "genuinely new LOSS term fh1 ranked worth a burn-4 lever."
        ),
        evidence_source=".omx/research/ddm_fh1_forces_harvest_20260731.md R1+R4; src/tac/witness_dsl/fh1_adapted_force_levers_20260731.py:36",
        verdict_scope="N_A",
        protection="N_A",
    ),
    _r(
        row_id="fh1.birth_plateau_knee",
        scope_kind="CLASS",
        scope="Lane",
        force="birth_plateau_knee_conjunct",
        force_kind="GUARD",
        verb="BIRTH",
        verdict="UNMEASURED",
        measured=False,
        evidence=(
            "Event guard (fh1 R8, #430 coherent-cascade order): require birth-plateau AND loss-knee "
            "before curriculum advance. Motivation MEASURED: Lane births +8.75/gate STILL RISING at "
            "ep399, 599/985 erased — advancing the curriculum before births plateau erases them. "
            "DESIGNED-STUB (fh1_birth_plateau_knee_conjunct); typed birth_completion key is P2, "
            "rides tp1."
        ),
        evidence_source=".omx/research/ddm_fh1_forces_harvest_20260731.md R8; src/tac/witness_dsl/fh1_adapted_force_levers_20260731.py:100",
        verdict_scope="N_A",
        protection="DESIGNED",
        protection_ref="stub lever exists; flags on neither trainer",
    ),
    _r(
        row_id="law.address_cost",
        scope_kind="FAMILY",
        scope="location_addressed_protections",
        force="any_transmitted_location_mask",
        force_kind="FAMILY",
        verb="AGGREGATE",
        verdict="NEUTRAL",
        measured=True,
        evidence=(
            "THE ADDRESS LAW (gt2r, rate-side bound on every protection design): implicit coding of "
            "the label field (410,584 B lzma9e whole-corpus) BEATS static-explicit (+5.7%) and "
            "temporal-explicit (+37.6%) addressing; the address is 76-78% of explicit cost "
            "(sx2/hs1 convergent: per-pair explicit address 331,236 B = 76.4% of address+payload). "
            "Protections that require TRANSMITTING a location mask are structurally expensive; the "
            "viable forms are receiver-generated addressing (dilate(boundary(already-sent payload)) "
            "= GENERIC, free) or tiny counted priors (sx2's 49 B static map, 430x)."
        ),
        evidence_source=f"{_GT2} (gt2_mdl/gt2_split.json); ddm_sx2; ddm_hs1",
        verdict_scope="FORMULATION",
        protection="N_A",
    ),
)


# --------------------------------------------------------------------------- #
# queries
# --------------------------------------------------------------------------- #

def validate_ledger(rows: tuple[ForceLedgerRow, ...] = LEDGER) -> None:
    """Raise on duplicate row ids (dataclass validation runs at construction)."""
    seen: set[str] = set()
    for row in rows:
        if row.row_id in seen:
            raise ValueError(f"duplicate row_id {row.row_id}")
        seen.add(row.row_id)


def by_force(force: str, rows: tuple[ForceLedgerRow, ...] = LEDGER) -> list[ForceLedgerRow]:
    return [r for r in rows if r.force == force]


def by_scope(scope: str, rows: tuple[ForceLedgerRow, ...] = LEDGER) -> list[ForceLedgerRow]:
    return [r for r in rows if r.scope == scope or (r.scope_kind == "GLOBAL" and scope == "ALL")]


def harms(rows: tuple[ForceLedgerRow, ...] = LEDGER) -> list[ForceLedgerRow]:
    out = [r for r in rows if r.verdict == "HARMS"]
    out.sort(key=lambda r: (-(r.magnitude_s or 0.0), not r.measured))
    return out


def unprotected_harms(rows: tuple[ForceLedgerRow, ...] = LEDGER) -> list[ForceLedgerRow]:
    """HARMS rows whose protection is ABSENT or BUILT_UNFIRED — the ranked debt list."""
    rank = {"ABSENT": 0, "BUILT_UNFIRED": 1, "DESIGNED": 2}
    out = [r for r in rows if r.verdict == "HARMS" and r.protection in rank]
    out.sort(key=lambda r: (rank[r.protection], -(r.magnitude_s or 0.0)))
    return out


def unmeasured(rows: tuple[ForceLedgerRow, ...] = LEDGER) -> list[ForceLedgerRow]:
    return [r for r in rows if r.verdict == "UNMEASURED"]


def forces(rows: tuple[ForceLedgerRow, ...] = LEDGER) -> list[str]:
    return sorted({r.force for r in rows})


def coverage(rows: tuple[ForceLedgerRow, ...] = LEDGER) -> dict:
    """The m50 denominator: how many cells exist vs carry evidence.

    Cell universe: for each non-guard force, 5 classes + 9 edges = 14 sign cells.
    A cell counts as covered iff an explicit row exists for that (force, scope);
    GLOBAL/FAMILY rows cover their force at the aggregate level only and are
    reported separately — they do NOT count as per-class/per-edge coverage.
    """
    per_force: dict[str, dict] = {}
    for f in forces(rows):
        f_rows = by_force(f, rows)
        class_cells = {r.scope for r in f_rows if r.scope_kind == "CLASS"}
        edge_cells = {r.scope for r in f_rows if r.scope_kind in ("EDGE", "DIRECTED_EDGE")}
        measured_cells = sum(
            1 for r in f_rows if r.scope_kind in ("CLASS", "EDGE", "DIRECTED_EDGE") and r.measured
        )
        per_force[f] = {
            "rows": len(f_rows),
            "class_cells_covered": len(class_cells),
            "class_cells_possible": len(CLASSES),
            "edge_cells_covered": len(edge_cells),
            "edge_cells_possible": len(EDGES),
            "measured_scope_cells": measured_cells,
            "has_global_or_family_row": any(
                r.scope_kind in ("GLOBAL", "FAMILY") for r in f_rows
            ),
        }
    n_forces = len(per_force)
    cells_possible = n_forces * (len(CLASSES) + len(EDGES))
    cells_covered = sum(
        v["class_cells_covered"] + v["edge_cells_covered"] for v in per_force.values()
    )
    return {
        "forces": n_forces,
        "rows": len(rows),
        "scope_cells_possible": cells_possible,
        "scope_cells_with_explicit_rows": cells_covered,
        "scope_cells_unmeasured_implicit": cells_possible - cells_covered,
        "honest_statement": (
            f"{cells_covered}/{cells_possible} (force x class|edge) cells carry an explicit row; "
            f"the remaining {cells_possible - cells_covered} are UNMEASURED-implicit. An empty cell "
            "is VACUOUS, never PASS (m50)."
        ),
        "per_force": per_force,
    }


def to_jsonl(path: str | Path, rows: tuple[ForceLedgerRow, ...] = LEDGER) -> Path:
    p = Path(path)
    with p.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(dataclasses.asdict(row)) + "\n")
    return p


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--coverage", action="store_true")
    ap.add_argument("--harms", action="store_true")
    ap.add_argument("--unprotected", action="store_true")
    ap.add_argument("--unmeasured", action="store_true")
    ap.add_argument("--force", type=str, default=None)
    ap.add_argument("--scope", type=str, default=None)
    ap.add_argument("--jsonl", type=str, default=None)
    args = ap.parse_args(argv)

    validate_ledger()
    picked: list[ForceLedgerRow] | None = None
    if args.harms:
        picked = harms()
    elif args.unprotected:
        picked = unprotected_harms()
    elif args.unmeasured:
        picked = unmeasured()
    elif args.force:
        picked = by_force(args.force)
    elif args.scope:
        picked = by_scope(args.scope)

    if args.coverage or picked is None and not args.jsonl:
        print(json.dumps(coverage(), indent=1))
    if picked is not None:
        for r in picked:
            print(
                f"[{r.verdict:10s}] {r.scope_kind}:{r.scope:22s} force={r.force:40s} "
                f"verb={r.verb:10s} prot={r.protection:13s} measured={r.measured}"
            )
    if args.jsonl:
        out = to_jsonl(args.jsonl)
        print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
