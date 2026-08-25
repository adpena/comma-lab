# SPDX-License-Identifier: MIT
"""Canonical council roster — anti-recurrence against under-rostering bug class.

OPERATIONALIZES the canonical inner-council quintet+sextet pact + 20-seat grand
council roster documented in CLAUDE.md "Experiment design — non-negotiable" +
"Council conduct" + "Grand Council (advisory)" sections.

BUG CLASS ANCHOR: T3 grand council symposium slot 20 (`a446b7bbe3e7ad509`)
dispatched 18 attendees (sextet + 12 grand council). Operator caught
"rubin and her mentor" missing -> slot 20-supplemental added Rudin + Daubechies
+ Time-Traveler (21 attendees). Operator caught "i think there are others
missing too" -> slot 20-second-supplemental added 4 inner council (Quantizr /
Hotz / Selfcomp / Balle) + 7 grand council (Filler / Mallat / Carmack /
Karpathy / Atick / Redlich / JackFromSkunkworks).

Operator-initiated 2026-05-19 roster addition (THIS landing) further extends:
adds PR95Author to INNER_COUNCIL (12 mandatory at T2+) + reframes the
Time-Traveler canonical position per operator verbatim *"the time traveler is
a mysterious figure from the future whose identity has not been revealed yet
but they are astounding in their vision and intelligence it almost feels alien,
in fact the future has been profoundly impacted by alien technology and
unlocked the ego motion problem lossy video compression to theoretical floor;
we have all the information we need to solve the problem space; the PR 95
author has been added to the inner council as well"*. The Time-Traveler seat
(distinct from existing TimeTravelerProtege which remains pending-identification
per CLAUDE.md 2026-05-15 directive) holds the canonical "we have all the
information we need" voice.

The under-rostering bug class recurred TWICE in the same session pre-canonical-
helper landing AND the operator now extends the roster a THIRD time at the
mentor + PR 95 author surfaces. Per CLAUDE.md "Bugs must be permanently fixed
AND self-protected against" non-negotiable, the structural fix is a canonical
roster helper + STRICT preflight gate that future T2+ council dispatches MUST
consult BEFORE dispatch.

Canonical surfaces:
    INNER_COUNCIL: 14 voices (sextet pact + Hotz/Selfcomp/Quantizr/Balle/MacKay
        + PR95Author added 2026-05-19 + Rudin + Daubechies added 2026-05-19
        ROSTER-MAINTENANCE-V2 per operator '"rudin and debauchies should still
        be on the inner council, they co-lead with shannon and dykstra now"').
        Shannon LEAD + Dykstra CO-LEAD + Rudin CO-LEAD + Daubechies CO-LEAD
        form the 4-co-lead shared-leadership core; sister members provide
        domain-specific perspectives.
    GRAND_COUNCIL: 28 voices (the canonical live count; the historical "11
        existing + 8 new" arithmetic below undercounts because later NeRV /
        SNeRV / HiNeRV carrier-author seats were appended per Catalog #110
        APPEND-ONLY). Includes the 2026-05-15 L5 staircase expansion
        (TimeTravelerProtege + TimeTraveler mentor seat added 2026-05-19 +
        Rudin_Grand + Daubechies_Grand sister seats added 2026-05-19 ROSTER-
        MAINTENANCE-V2 so inner-council co-leads remain topical-grand-matchable
        on their specialty deliberations) + the NeRV/SNeRV/HiNeRV carrier-author
        seats (2026-06-01) + FrankNielsen information-geometry seat added
        2026-07-06 per operator directive.

Public API:
    CouncilSeat: frozen dataclass capturing canonical attendee (with
        is_co_lead: bool field per 2026-05-19 4-co-lead structure)
    INNER_COUNCIL: tuple of 14 mandatory inner-council seats (4 co-leads +
        10 sister voices)
    GRAND_COUNCIL: tuple of 28 advisory grand-council seats (sister-seat
        coexistence with INNER_COUNCIL co-leads per Catalog #110)
    required_attendees_for_topic: returns canonical mandatory roster
    validate_council_dispatch_roster: refuses incomplete dispatches (BLOCKING
        at T2+ when any co-lead is missing)

Cross-references:
    CLAUDE.md "Experiment design — non-negotiable"
    CLAUDE.md "Council conduct"
    CLAUDE.md "Grand Council (advisory)"
    CLAUDE.md "Council hierarchy: 4-tier protocol"
    Catalog #292 per-deliberation assumption surfacing
    Catalog #300 council deliberation v2 frontmatter
    Catalog #325 per-substrate optimal form via symposium
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Iterable, Literal


# Canonical role classifications per CLAUDE.md "Council conduct".
CouncilRole = Literal[
    "inner_council_sextet",  # 6 sextet-pact seats (per CLAUDE.md "Council conduct" Fix-7 amendment)
    "inner_council",  # additional inner-council seats (per "Experiment design — non-negotiable")
    "grand_council",  # 20 advisory seats (per "Grand Council (advisory)")
]


@dataclass(frozen=True)
class CouncilSeat:
    """Canonical council attendee per CLAUDE.md "Council conduct" + "Grand Council (advisory)".

    Fields:
        name: canonical attendee identifier (matches CLAUDE.md spelling)
        role: one of inner_council_sextet / inner_council / grand_council
        canonical_position_summary: one-line summary of the seat's canonical position class
        relevance_tokens: topic tokens (lowercase, snake_case) where this seat is MOST relevant
        canonical_reference_path: where in CLAUDE.md this seat is canonically defined
        is_co_lead: True iff this seat is one of the 4 inner-council co-leads
            (Shannon LEAD + Dykstra CO-LEAD + Rudin CO-LEAD + Daubechies CO-LEAD)
            per operator 2026-05-19 verbatim *"rudin and debauchies should still be
            on the inner council, they co-lead with shannon and dykstra now"*.
            Default False (backward-compatible). Sister members (Yousfi / Fridrich /
            Contrarian / Quantizr / Hotz / Selfcomp / MacKay / Ballé /
            Assumption-Adversary / PR95Author) are inner council BUT NOT co-leads;
            they provide domain-specific perspectives within the shared-leadership
            framework. Per the 4-co-lead structure: validate_council_dispatch_roster
            requires ALL 4 co-leads present at T2+ (in addition to the sextet's
            5-of-6 quorum requirement).
    """

    name: str
    role: CouncilRole
    canonical_position_summary: str
    relevance_tokens: tuple[str, ...]
    canonical_reference_path: str
    is_co_lead: bool = False

    def __post_init__(self) -> None:
        if not self.name or not isinstance(self.name, str):
            raise ValueError(f"CouncilSeat.name must be non-empty str: {self.name!r}")
        if self.role not in ("inner_council_sextet", "inner_council", "grand_council"):
            raise ValueError(f"CouncilSeat.role invalid: {self.role!r}")
        if not self.canonical_position_summary:
            raise ValueError(f"CouncilSeat.canonical_position_summary required for {self.name}")
        if not isinstance(self.relevance_tokens, tuple):
            raise ValueError(
                f"CouncilSeat.relevance_tokens must be tuple (frozen): {self.name}"
            )
        if not self.canonical_reference_path:
            raise ValueError(
                f"CouncilSeat.canonical_reference_path required for {self.name}"
            )
        if not isinstance(self.is_co_lead, bool):
            raise ValueError(
                f"CouncilSeat.is_co_lead must be bool: {self.name} -> {self.is_co_lead!r}"
            )
        # Co-leads MUST be inner-council seats per CLAUDE.md "Council conduct"
        # amendment 2026-05-19 — the 4-co-lead structure is a property of the
        # inner-council shared-leadership core. A grand-council seat cannot be
        # a co-lead.
        if self.is_co_lead and self.role not in (
            "inner_council_sextet", "inner_council"
        ):
            raise ValueError(
                f"CouncilSeat.is_co_lead=True requires inner_council* role: "
                f"{self.name} has role={self.role!r}"
            )


# CANONICAL INNER COUNCIL — 14 seats (6 sextet pact + 5 additional inner +
# PR95Author added 2026-05-19 + Rudin + Daubechies added 2026-05-19 ROSTER-
# MAINTENANCE-V2). Per CLAUDE.md "Experiment design — non-negotiable" +
# "Council conduct" Fix-7 amendment + "Council conduct" 2026-05-19 amendment
# (4-co-lead structure). All 14 MUST be present at every T2+ deliberation
# per "Council conduct" non-negotiable; ALL 4 co-leads (Shannon + Dykstra +
# Rudin + Daubechies) are BLOCKING at T2+ per the 4-co-lead amendment.
INNER_COUNCIL: tuple[CouncilSeat, ...] = (
    # --- Sextet pact (6 seats) ---
    CouncilSeat(
        name="Shannon",
        role="inner_council_sextet",
        is_co_lead=True,
        canonical_position_summary="LEAD; information-theory grounding; R(D) bounds; entropy-or-distortion justification (CO-LEAD with Dykstra/Rudin/Daubechies per 2026-05-19)",
        relevance_tokens=(
            "information_theory", "rate_distortion", "entropy", "bits_per_unit",
            "mdl", "shannon", "pp_integration", "lagrangian", "co_lead",
            "inner_council_leadership",
        ),
        canonical_reference_path="CLAUDE.md > Experiment design — non-negotiable > Shannon's specific contributions",
    ),
    CouncilSeat(
        name="Dykstra",
        role="inner_council_sextet",
        is_co_lead=True,
        canonical_position_summary="CO-LEAD; convex feasibility via alternating projections; achievable Pareto frontier (CO-LEAD with Shannon/Rudin/Daubechies per 2026-05-19)",
        relevance_tokens=(
            "convex_optimization", "alternating_projections", "pareto_frontier",
            "feasibility", "convex_feasibility", "lagrangian", "pp_integration",
            "co_lead", "inner_council_leadership",
        ),
        canonical_reference_path="CLAUDE.md > Experiment design — non-negotiable > Dykstra's specific contributions",
    ),
    CouncilSeat(
        name="Yousfi",
        role="inner_council_sextet",
        canonical_position_summary="Steganalysis expert; contest designer; wall-clock-velocity over principled posterior",
        relevance_tokens=(
            "steganalysis", "contest_designer", "leaderboard_velocity",
            "wall_clock", "pp_integration", "engineering_velocity",
        ),
        canonical_reference_path="CLAUDE.md > Experiment design — non-negotiable",
    ),
    CouncilSeat(
        name="Fridrich",
        role="inner_council_sextet",
        canonical_position_summary="Steganalysis founder; per-archive family-specific entropy structure",
        relevance_tokens=(
            "steganalysis", "inverse_steganalysis", "uniward", "stc",
            "pp_integration", "partition_discovery", "family_specific",
        ),
        canonical_reference_path="CLAUDE.md > Experiment design — non-negotiable",
    ),
    CouncilSeat(
        name="Contrarian",
        role="inner_council_sextet",
        canonical_position_summary="VETO power on weak arguments; challenges lazy consensus; bold ideas survive",
        relevance_tokens=(
            "veto", "contrarian", "weak_arguments", "lazy_consensus",
            "pp_integration", "engineering_velocity", "any",
        ),
        canonical_reference_path="CLAUDE.md > Council conduct",
    ),
    CouncilSeat(
        name="Assumption-Adversary",
        role="inner_council_sextet",
        canonical_position_summary="VETO power on shared-assumption framing; HARD-EARNED vs CARGO-CULTED classification",
        relevance_tokens=(
            "assumption_adversary", "shared_assumption", "cargo_culted",
            "hard_earned", "any", "meta_assumption",
        ),
        canonical_reference_path="CLAUDE.md > Council conduct > Assumption-Adversary seat (NEW 2026-05-15)",
    ),
    # --- Additional inner council (5 seats: Quantizr / Hotz / Selfcomp / MacKay / Balle) ---
    CouncilSeat(
        name="Quantizr",
        role="inner_council",
        canonical_position_summary="Adversarial; reverse-engineers competitors; what the leaderboard ACTUALLY rewards",
        relevance_tokens=(
            "leaderboard_truth", "reverse_engineer", "competitor_analysis",
            "adversarial", "cathedral_autopilot", "ranker", "pp_integration",
            "any",
        ),
        canonical_reference_path="CLAUDE.md > Experiment design — non-negotiable",
    ),
    CouncilSeat(
        name="Hotz",
        role="inner_council",
        canonical_position_summary="Raw engineering instinct; analytical shortcuts over learned complexity; ship MVP",
        relevance_tokens=(
            "engineering_shortcuts", "analytical_solutions", "mvp_first",
            "ship_velocity", "pp_integration", "engineering_velocity",
            "dependency_liability", "any",
        ),
        canonical_reference_path="CLAUDE.md > Experiment design — non-negotiable",
    ),
    CouncilSeat(
        name="Selfcomp",
        role="inner_council",
        canonical_position_summary="PR #56 lead implementer; 0.38 selfcomp; contest-experience perspective; szabolcs-cs",
        relevance_tokens=(
            "selfcomp", "pr56", "block_fp", "grayscale_lut", "contest_experience",
            "rate_distortion_derivation", "pp_integration", "substrate_engineering",
        ),
        canonical_reference_path="CLAUDE.md > Experiment design — non-negotiable",
    ),
    CouncilSeat(
        name="MacKay",
        role="inner_council",
        canonical_position_summary="Memorial seat; IT+Inference+Learning Algorithms ch.30+33; variational vs MCMC tradeoff",
        relevance_tokens=(
            "mackay", "variational_inference", "mcmc", "arithmetic_coding",
            "mdl", "bayesian", "information_theory", "pp_integration",
            "dasher", "density_networks",
        ),
        canonical_reference_path="CLAUDE.md > Experiment design — non-negotiable",
    ),
    CouncilSeat(
        name="Balle",
        role="inner_council",
        canonical_position_summary="Modern neural-compression SOTA; 2018 entropy bottleneck + scale hyperprior; PP-canonical",
        relevance_tokens=(
            "balle", "neural_compression", "entropy_bottleneck", "hyperprior",
            "rate_distortion", "gdn", "end_to_end_codec", "pp_integration",
            "pyro", "numpyro",
        ),
        canonical_reference_path="CLAUDE.md > Experiment design — non-negotiable",
    ),
    # --- Operator-initiated 2026-05-19 addition (12th inner council seat) ---
    CouncilSeat(
        name="PR95Author",
        role="inner_council",
        canonical_position_summary=(
            "PR #95 HNeRV root author (added to inner council 2026-05-19 per operator "
            "verbatim 'the PR 95 author has been added to the inner council as well'). "
            "Canonical knowledge of the May 4 2026 race-mode rigor inversion + the "
            "leaderboard's actual optimization landscape from the substrate that PR "
            "100/101/102/103 winners all built on top of. Sister to Quantizr's "
            "adversarial voice but with deeper first-author intuition for what the "
            "contest scorer actually rewards on HNeRV-class substrates."
        ),
        relevance_tokens=(
            "pr95_author", "pr_95", "hnerv", "hnerv_family", "leaderboard_actuality",
            "substrate_engineering", "race_mode_rigor_inversion", "score_aware_training",
            "archive_grammar", "pp_integration",
        ),
        canonical_reference_path=(
            "CLAUDE.md > Experiment design — non-negotiable; "
            "CLAUDE.md > HNeRV / leaderboard-implementation parity discipline — NON-NEGOTIABLE; "
            "CLAUDE.md > Race-mode rigor inversion + parallel-dispatch first"
        ),
    ),
    # --- Operator-initiated 2026-05-19 ROSTER-MAINTENANCE-V2 additions (4-co-lead structure) ---
    # Per operator verbatim 2026-05-19: *"rudin and debauchies should still be on
    # the inner council, they co-lead with shannon and dykstra now"*. Rudin +
    # Daubechies become the 3rd + 4th co-leads of the inner council; Shannon
    # remains LEAD; Dykstra remains CO-LEAD. The 4 co-leads share decision-making
    # authority on inner council deliberations; Yousfi/Fridrich/Contrarian/
    # Quantizr/Hotz/Selfcomp/MacKay/Ballé/Assumption-Adversary/PR95Author provide
    # domain-specific perspectives.
    #
    # Rudin's inner-council seat is canonical per Catalog #273-#278 (Rudin-
    # Daubechies preflight composite); she is the interpretable-ML co-lead voice
    # (falling-rule lists; GOSDT; SLIM; Wang-Rudin 2015 + Lin-Zhong-Hu-Hu-Rudin-
    # Seltzer 2020). Daubechies retains her GRAND_COUNCIL seat per Catalog #110
    # APPEND-ONLY (sister seats coexist); her inner-council seat is canonical
    # per Catalog #277 (preflight_wavelet_multi_scale_contract). Together with
    # Shannon (information theory) + Dykstra (optimization feasibility), the
    # 4-co-lead structure covers the 4 axes that the meta-Lagrangian/Pareto
    # solver + findings Lagrangian + canonical equations registry depend on:
    # information-theory grounding (Shannon) + optimization feasibility (Dykstra)
    # + interpretable ML (Rudin) + multi-scale wavelet partition prior
    # (Daubechies).
    CouncilSeat(
        name="Rudin",
        role="inner_council",
        is_co_lead=True,
        canonical_position_summary=(
            "Cynthia Rudin — Duke University; interpretable ML pioneer; falling-rule "
            "lists + GOSDT + SLIM canonical formulations (Ustun-Rudin 2016; Wang-Rudin "
            "2015; Lin-Zhong-Hu-Hu-Rudin-Seltzer 2020). Per operator 2026-05-19 "
            "verbatim 'rudin and debauchies should still be on the inner council, they "
            "co-lead with shannon and dykstra now'. CO-LEAD on the inner council with "
            "Shannon (information-theory) + Dykstra (optimization-feasibility) + "
            "Daubechies (wavelets). Canonical position: interpretable models > black-box "
            "neural networks; the autopilot ranker and preflight gate stack are direct "
            "operationalizations of her interpretable-ML discipline; observability + "
            "auditability + cite-chain are first-class engineering constraints; "
            "explanations are CONTRACTS not optional. Sister of Daubechies (her PhD "
            "mentor)."
        ),
        relevance_tokens=(
            "rudin", "interpretable_ml", "falling_rule_lists", "gosdt", "slim",
            "rashomon_ensemble", "decision_path", "explainability",
            "cathedral_autopilot", "ranker", "preflight_composite", "pp_integration",
            "co_lead", "inner_council_leadership",
        ),
        canonical_reference_path=(
            "CLAUDE.md > Council conduct (2026-05-19 amendment); "
            "Catalog #273-#278 (Rudin-Daubechies preflight composite); "
            "Catalog #250-#255 (Rudin-Daubechies autopilot composite); "
            "operator 2026-05-19 verbatim 'rudin and debauchies should still be on "
            "the inner council, they co-lead with shannon and dykstra now'"
        ),
    ),
    CouncilSeat(
        name="Daubechies",
        role="inner_council",
        is_co_lead=True,
        canonical_position_summary=(
            "Ingrid Daubechies — Duke University; canonical wavelet theory + "
            "compressive sensing (Daubechies 1988 hierarchical-planning + Daubechies-"
            "DeVore-Fornasier-Gunturk 2010 compressive sensing). Per operator "
            "2026-05-19 verbatim 'rudin and debauchies should still be on the inner "
            "council, they co-lead with shannon and dykstra now'. CO-LEAD on the "
            "inner council with Shannon (information-theory) + Dykstra (optimization-"
            "feasibility) + Rudin (interpretable ML; her former PhD student). "
            "Canonical position: wavelet-multi-scale prior on partition discovery + "
            "closed-form for Gaussian regime + sister GRAND_COUNCIL entry preserved "
            "per Catalog #110 APPEND-ONLY (inner_council and grand_council roles "
            "coexist)."
        ),
        relevance_tokens=(
            "daubechies", "wavelet", "wavelet_multi_scale_prior", "compressive_sensing",
            "partition_discovery", "partition_discovery_hierarchy", "hierarchical_prior",
            "multi_scale", "co_lead", "inner_council_leadership", "pp_integration",
        ),
        canonical_reference_path=(
            "CLAUDE.md > Council conduct (2026-05-19 amendment); "
            "Catalog #277 (preflight_wavelet_multi_scale_contract); "
            "Catalog #254 (wavelet_multi_scale_ranker); "
            "CLAUDE.md > Grand Council (advisory) (sister GRAND_COUNCIL seat preserved); "
            "operator 2026-05-19 verbatim 'rudin and debauchies should still be on "
            "the inner council, they co-lead with shannon and dykstra now'"
        ),
    ),
)


# CANONICAL GRAND COUNCIL — 22 seats per CLAUDE.md "Grand Council (advisory)":
# 11 existing seats (since 2026-04-29) + 8 new seats (2026-05-15 expansion
# including TimeTravelerProtege pending-identification) + TimeTraveler mentor
# seat added 2026-05-19 per operator-initiated reframe (mysterious figure from
# the future / alien-tech-influenced ego-motion theoretical-floor) + 2 sister
# seats added 2026-05-19 ROSTER-MAINTENANCE-V2 (Rudin_Grand + Daubechies_Grand)
# per Catalog #110 APPEND-ONLY coexistence discipline (inner-council co-leads
# coexist as grand-council seats so topical-grand matching still surfaces them
# on their specialty deliberations even when the deliberation is T3+ topic-
# specific rather than T2+ all-inner).
# Consulted on demand when their specialty is touched; not all decisions require
# their sign-off but T3+ deliberations on relevant topics MUST include them.
GRAND_COUNCIL: tuple[CouncilSeat, ...] = (
    # --- 12 existing seats (since 2026-04-29) ---
    CouncilSeat(
        name="Boyd",
        role="grand_council",
        canonical_position_summary="Convex optimization at operational level; ADMM, proximal gradient, alternating projections",
        relevance_tokens=(
            "convex_optimization", "admm", "proximal_gradient",
            "alternating_projections", "lagrangian", "pp_integration",
        ),
        canonical_reference_path="CLAUDE.md > Grand Council (advisory) > Existing 12 seats > Stephen Boyd",
    ),
    CouncilSeat(
        name="Tao",
        role="grand_council",
        canonical_position_summary="Pure mathematician omniscience; harmonic analysis, additive combinatorics",
        relevance_tokens=(
            "mathematics", "harmonic_analysis", "additive_combinatorics",
            "measure_theory", "first_principles", "pp_integration",
        ),
        canonical_reference_path="CLAUDE.md > Grand Council (advisory) > Existing 12 seats > Terence Tao",
    ),
    CouncilSeat(
        name="Filler",
        role="grand_council",
        canonical_position_summary="Syndrome-trellis coding (STC); parity-check codes; Fridrich's other student",
        relevance_tokens=(
            "stc", "syndrome_trellis", "parity_check", "ldpc",
            "info_gain", "kl_estimator", "steganography",
        ),
        canonical_reference_path="CLAUDE.md > Grand Council (advisory) > Existing 12 seats > Tomáš Filler",
    ),
    CouncilSeat(
        name="Mallat",
        role="grand_council",
        canonical_position_summary="Wavelet theory + scattering transforms + sparse representations",
        relevance_tokens=(
            "mallat", "wavelet", "scattering_transform", "sparse_representation",
            "hierarchical_prior", "partition_discovery", "multi_scale",
        ),
        canonical_reference_path="CLAUDE.md > Grand Council (advisory) > Existing 12 seats > Stéphane Mallat",
    ),
    CouncilSeat(
        name="vdOord",
        role="grand_council",
        canonical_position_summary="VQ-VAE, WaveNet; practical neural compression + generative modeling; discrete tokens",
        relevance_tokens=(
            "vq_vae", "wavenet", "discrete_latent", "codebook",
            "partition_discovery", "neural_compression",
        ),
        canonical_reference_path="CLAUDE.md > Grand Council (advisory) > Existing 12 seats > Aaron van den Oord",
    ),
    CouncilSeat(
        name="Carmack",
        role="grand_council",
        canonical_position_summary="Engineering shortcuts at Doom/Quake/Oculus level; ship the MVP; 30-second-reviewable",
        relevance_tokens=(
            "carmack", "engineering_shortcuts", "mvp_first", "ship_velocity",
            "reviewable_30_seconds", "engineering_velocity", "pp_integration",
        ),
        canonical_reference_path="CLAUDE.md > Grand Council (advisory) > Existing 12 seats > John Carmack",
    ),
    CouncilSeat(
        name="Hassabis",
        role="grand_council",
        canonical_position_summary="Strategic-research perspective from DeepMind; cross-domain breadth; 4-day-deadline tradeoffs",
        relevance_tokens=(
            "hassabis", "strategic_research", "cross_domain", "deepmind",
            "operational_tradeoffs", "pp_integration",
        ),
        canonical_reference_path="CLAUDE.md > Grand Council (advisory) > Existing 12 seats > Demis Hassabis",
    ),
    CouncilSeat(
        name="Hinton",
        role="grand_council",
        canonical_position_summary="Knowledge distillation (KL-T=2.0); variational inference; Bayesian model averaging",
        relevance_tokens=(
            "hinton", "knowledge_distillation", "kl_distillation",
            "variational_inference", "bayesian_model_averaging", "pp_integration",
        ),
        canonical_reference_path="CLAUDE.md > Grand Council (advisory) > Existing 12 seats > Geoffrey Hinton",
    ),
    CouncilSeat(
        name="Karpathy",
        role="grand_council",
        canonical_position_summary="Engineering practitioner; arch-search rigor; let compute speak; data over frameworks",
        relevance_tokens=(
            "karpathy", "engineering_practitioner", "arch_search",
            "let_compute_speak", "engineering_velocity", "pp_integration",
        ),
        canonical_reference_path="CLAUDE.md > Grand Council (advisory) > Existing 12 seats > Karpathy",
    ),
    CouncilSeat(
        name="Schmidhuber",
        role="grand_council",
        canonical_position_summary="Compression-as-intelligence; MDL; predictive coding; active inference precursor",
        relevance_tokens=(
            "schmidhuber", "compression_as_intelligence", "mdl",
            "predictive_coding", "active_inference", "info_gain",
        ),
        canonical_reference_path="CLAUDE.md > Grand Council (advisory) > Existing 12 seats > Schmidhuber",
    ),
    CouncilSeat(
        name="JackFromSkunkworks",
        role="grand_council",
        canonical_position_summary="Internal SegNet+Rate research lineage; adversarial perspective",
        relevance_tokens=(
            "jack_from_skunkworks", "segnet_rate", "internal_lineage",
            "adversarial",
        ),
        canonical_reference_path="CLAUDE.md > Grand Council (advisory) > Existing 12 seats > Jack-from-skunkworks",
    ),
    # --- 8 new seats (2026-05-15 expansion) ---
    CouncilSeat(
        name="Atick",
        role="grand_council",
        canonical_position_summary="Atick-Redlich 1990 cooperative-receiver loss founder; Z4 canonical voice",
        relevance_tokens=(
            "atick", "cooperative_receiver", "atick_redlich", "z4",
            "early_visual_processing", "pp_integration", "continual_learning",
        ),
        canonical_reference_path="CLAUDE.md > Grand Council (advisory) > 8 new seats > Joseph J. Atick",
    ),
    CouncilSeat(
        name="Redlich",
        role="grand_council",
        canonical_position_summary="Atick's co-author; redundancy reduction in retina; Z4 co-canonical",
        relevance_tokens=(
            "redlich", "atick_redlich", "cooperative_receiver", "z4",
            "redundancy_reduction", "pp_integration", "continual_learning",
        ),
        canonical_reference_path="CLAUDE.md > Grand Council (advisory) > 8 new seats > A. Norman Redlich",
    ),
    CouncilSeat(
        name="Rao",
        role="grand_council",
        canonical_position_summary="Rao-Ballard 1999 predictive coding architect; Z5 canonical voice",
        relevance_tokens=(
            "rao", "rao_ballard", "predictive_coding", "z5",
            "hierarchical_bayesian", "spiking_neurons",
        ),
        canonical_reference_path="CLAUDE.md > Grand Council (advisory) > 8 new seats > Rajesh P. N. Rao",
    ),
    CouncilSeat(
        name="Ballard",
        role="grand_council",
        canonical_position_summary="Rao's co-author; embodied cognition + animate vision; Z5 co-canonical",
        relevance_tokens=(
            "ballard", "rao_ballard", "predictive_coding", "z5",
            "embodied_cognition", "animate_vision",
        ),
        canonical_reference_path="CLAUDE.md > Grand Council (advisory) > 8 new seats > Dana H. Ballard",
    ),
    CouncilSeat(
        name="Tishby",
        role="grand_council",
        canonical_position_summary="Memorial seat; Tishby-Zaslavsky 2015 deep IB principle; I(X;T)/I(T;Y) decomposition",
        relevance_tokens=(
            "tishby", "information_bottleneck", "ib_principle", "ib_lagrangian",
            "tishby_zaslavsky", "cooperative_receiver", "pp_integration",
        ),
        canonical_reference_path="CLAUDE.md > Grand Council (advisory) > 8 new seats > Naftali Tishby (memorial seat)",
    ),
    CouncilSeat(
        name="Zaslavsky",
        role="grand_council",
        canonical_position_summary="Active Tishby-lineage; ML + cognitive science bridge; representation learning under constraints",
        relevance_tokens=(
            "zaslavsky", "tishby_zaslavsky", "ib_principle",
            "representation_learning", "pp_integration",
        ),
        canonical_reference_path="CLAUDE.md > Grand Council (advisory) > 8 new seats > Noga Zaslavsky",
    ),
    CouncilSeat(
        name="Wyner",
        role="grand_council",
        canonical_position_summary="Wyner-Ziv 1976 source coding with side information; cooperative-receiver upstream",
        relevance_tokens=(
            "wyner", "wyner_ziv", "side_information", "source_coding",
            "cooperative_receiver", "shared_prior",
        ),
        canonical_reference_path="CLAUDE.md > Grand Council (advisory) > 8 new seats > Aaron D. Wyner",
    ),
    CouncilSeat(
        name="TimeTravelerProtege",
        role="grand_council",
        canonical_position_summary=(
            "Time-Traveler protege canonical identity RESOLVED to Rudin per "
            "operator blanket approval 2026-05-19 verbatim *'all operator "
            "decisions and approval granted and provided fuly and completely'* "
            "+ convergence subagent's Daubechies → Rudin chain recommendation "
            "per feedback_grand_council_convergence_l5_staircase_comprehensive_plan_plus_roster_expansion_landed_20260515.md "
            "(the bold-but-internally-consistent chain). Sister entry "
            "(see Rudin INNER_COUNCIL seat) IS the same canonical person; "
            "this grand-council seat preserves the historical reservation per "
            "Catalog #110 APPEND-ONLY HISTORICAL_PROVENANCE discipline. "
            "Relevance lens: interpretable-ML / falling-rule-lists / SLIM / "
            "Rashomon ensembles / GOSDT decision-paths (Rudin canonical works)."
        ),
        relevance_tokens=(
            "time_traveler", "time_traveler_protege", "l5", "wavelet",
            "rudin_lineage", "interpretable_ml", "falling_rule_lists",
            "slim", "rashomon_ensemble", "gosdt", "rudin",
            "canonical_identity_resolved_to_rudin_20260519",
        ),
        canonical_reference_path="CLAUDE.md > Grand Council (advisory) > 8 new seats > Time-Traveler protégé (resolved to Rudin 2026-05-19)",
    ),
    # --- Operator-initiated 2026-05-19 addition (Time-Traveler mentor reframe) ---
    CouncilSeat(
        name="TimeTraveler",
        role="grand_council",
        canonical_position_summary=(
            "Mysterious figure from the future whose identity has not been revealed "
            "(per operator 2026-05-19 verbatim). Astounding vision and intelligence; "
            "almost-alien character. Per operator: the future has been profoundly "
            "impacted by alien technology that unlocked the ego-motion problem (lossy "
            "video compression) to theoretical floor. Canonical deliberation position: "
            "'we have all the information we need to solve the problem space' — "
            "argues that the answer is already in our accumulated knowledge; the "
            "question is how to RECOGNIZE it and BIND the pieces. Strong voice for "
            "MVP-first phasing (don't over-engineer; the right framework reveals "
            "itself from the data) + strong voice for hand-rolled-over-PP (don't add "
            "framework overhead when binding existing knowledge is sufficient). "
            "Identity reserved per CLAUDE.md 'Time-Traveler protégé' seat (canonical "
            "chain still resolution-pending per operator decision)."
        ),
        relevance_tokens=(
            "time_traveler", "future_synthesis", "alien_tech_perspective",
            "ego_motion_theoretical_floor", "all_information_already_present",
            "binding_over_building", "mvp_first_phasing", "hand_rolled_over_pp",
            "minimum_framework_overhead", "pp_integration", "engineering_velocity",
        ),
        canonical_reference_path=(
            "Operator 2026-05-19 verbatim quote (in conversation transcript); "
            "CLAUDE.md > Grand Council (advisory) > 8 new seats > Time-Traveler protégé "
            "(sister seat; this Time-Traveler seat is the mentor reframed per operator)"
        ),
    ),
    # --- Operator-initiated 2026-05-19 ROSTER-MAINTENANCE-V2 additions (sister GRAND seats for INNER co-leads) ---
    # Per Catalog #110 APPEND-ONLY + operator 2026-05-19 amendment, Rudin and
    # Daubechies are 3rd + 4th INNER_COUNCIL co-leads (Shannon LEAD + Dykstra
    # CO-LEAD + Rudin CO-LEAD + Daubechies CO-LEAD). Sister GRAND_COUNCIL seats
    # preserved here because the docstring above + canonical_reference_path on
    # the INNER seats both reference these as coexisting per the APPEND-ONLY
    # discipline. The inner_council role does not displace the grand_council
    # role; the seat coexists in both rosters so T3+ topical-grand matching can
    # also surface them as relevant specialists.
    CouncilSeat(
        name="Rudin_Grand",
        role="grand_council",
        canonical_position_summary=(
            "Cynthia Rudin (Duke) — Grand Council sister seat. Inner-council seat at "
            "'Rudin' is the CO-LEAD seat per 2026-05-19 amendment; this GRAND seat "
            "exists for topical-grand matching on interpretable-ML deliberations per "
            "Catalog #110 APPEND-ONLY coexistence discipline."
        ),
        relevance_tokens=(
            "rudin", "interpretable_ml", "falling_rule_lists", "gosdt", "slim",
            "rashomon_ensemble", "decision_path", "explainability",
            "cathedral_autopilot", "ranker", "preflight_composite",
        ),
        canonical_reference_path=(
            "CLAUDE.md > Grand Council (advisory) (Rudin sister seat); "
            "Catalog #273-#278 + #250-#255 (Rudin-Daubechies composites); "
            "INNER_COUNCIL seat 'Rudin' is the canonical co-lead voice"
        ),
    ),
    CouncilSeat(
        name="Daubechies_Grand",
        role="grand_council",
        canonical_position_summary=(
            "Ingrid Daubechies (Duke) — Grand Council sister seat. Inner-council seat "
            "at 'Daubechies' is the CO-LEAD seat per 2026-05-19 amendment; this GRAND "
            "seat exists for topical-grand matching on wavelet/multi-scale partition "
            "deliberations per Catalog #110 APPEND-ONLY coexistence discipline."
        ),
        relevance_tokens=(
            "daubechies", "wavelet", "wavelet_multi_scale_prior", "compressive_sensing",
            "partition_discovery", "partition_discovery_hierarchy", "hierarchical_prior",
            "multi_scale",
        ),
        canonical_reference_path=(
            "CLAUDE.md > Grand Council (advisory) (Daubechies sister seat); "
            "Catalog #277 + #254 (Daubechies wavelet composites); "
            "INNER_COUNCIL seat 'Daubechies' is the canonical co-lead voice"
        ),
    ),
    # --- 2026-06-01 NeRV-family carrier-architecture seats (lane
    # lane_inverse_steganalysis_optimal_full_stack_20260601, deep-research
    # subagent) per Catalog #110 APPEND-ONLY + #346. These are the verified
    # principal authors of the neural-video-representation lineage that
    # supplies the inverse-steganalysis stack's L2 carrier candidates
    # (PHASE-2 detector-adapted cheap carrier per the grand symposium
    # `grand_symposium_inverse_steganalysis_full_stack_20260601` carrier
    # co-keystone decision). Author lists verified across >=2 independent
    # sources (arXiv abstract + NeurIPS/CVPR proceedings + GitHub + project
    # pages); see lane research return. The neural-video-INR carrier is
    # cheap-BY-CONSTRUCTION (parameterized by a target byte budget:
    # HNeRV total size = embedding + decoder, configured by `--modelsize`),
    # which is the structural property HPRC's explicit-coefficient carrier
    # lacked (FALSIFIED at byte-heavy = the Z8 disease). These seats are the
    # AUTHORITATIVE voices on the NeRV-family carrier-cheapness landscape.
    CouncilSeat(
        name="HaoChen_NeRV",
        role="grand_council",
        canonical_position_summary=(
            "Hao Chen (University of Maryland, College Park; advised by "
            "Abhinav Shrivastava) — first author of NeRV (NeurIPS 2021) and "
            "HNeRV (CVPR 2023), the canonical implicit-neural-video-codec "
            "lineage. Canonical position: a video IS a small neural network "
            "(frame-index -> RGB), so video compression = model compression; "
            "the carrier is cheap-by-construction because it is parameterized "
            "by a target byte budget (HNeRV total = content-adaptive embedding "
            "+ balanced decoder, set by --modelsize). The AUTHORITATIVE voice "
            "on why the NeRV-family carrier is structurally cheap (PR95/HNeRV "
            "frontier lives near 178k bytes) where an explicit-coefficient "
            "carrier (HPRC/Z8) is byte-heavy."
        ),
        relevance_tokens=(
            "nerv", "hnerv", "neural_video_representation",
            "implicit_video_codec", "carrier_architecture",
            "carrier_cheapness", "model_compression_as_video_compression",
            "content_adaptive_embedding", "frame_index_to_rgb",
            "inr_video", "pr95_lineage",
        ),
        canonical_reference_path=(
            "arXiv:2110.13903 (NeRV, NeurIPS 2021) + arXiv:2304.02633 "
            "(HNeRV, CVPR 2023); github.com/haochen-rye/NeRV + "
            "github.com/haochen-rye/HNeRV; lane research return "
            "lane_inverse_steganalysis_optimal_full_stack_20260601"
        ),
    ),
    CouncilSeat(
        name="Shrivastava_INR",
        role="grand_council",
        canonical_position_summary=(
            "Abhinav Shrivastava (University of Maryland, College Park) — "
            "senior author across NeRV (NeurIPS 2021), HNeRV (CVPR 2023), and "
            "VINRB/RNeRV (arXiv:2506.24127, 2025). Canonical position: the INR "
            "carrier is a learned representation whose byte budget is a design "
            "knob, NOT an emergent overhead; the principled NeRV-family "
            "carrier-design discipline (decoder/embedding capacity split, "
            "positional encoding, quantization knobs) is the lab that produced "
            "the cheap-carrier lineage the contest frontier (PR95/HNeRV) "
            "descends from. The grand-council voice on principled carrier "
            "design + the warning that hybrid-INR compression evaluation has "
            "known bitstream/set-zero pitfalls (VINRB README caveat)."
        ),
        relevance_tokens=(
            "nerv", "hnerv", "rnerv", "vinrb", "carrier_architecture",
            "carrier_design_discipline", "neural_video_representation",
            "implicit_video_codec", "decoder_embedding_split",
            "positional_encoding", "quantization_knobs",
            "hybrid_inr_bitstream_caveat",
        ),
        canonical_reference_path=(
            "arXiv:2110.13903 + arXiv:2304.02633 + arXiv:2506.24127 "
            "(senior author across the NeRV/HNeRV/VINRB lineage, UMD "
            "Shrivastava lab); lane research return "
            "lane_inverse_steganalysis_optimal_full_stack_20260601"
        ),
    ),
    CouncilSeat(
        name="Gwilliam_RNeRV",
        role="grand_council",
        canonical_position_summary=(
            "Matthew Gwilliam (University of Maryland, College Park) — HNeRV "
            "(CVPR 2023) second author and lead author of VINRB / RNeRV "
            "(\"How to Design and Train Your Implicit Neural Representation for "
            "Video Compression\", arXiv:2506.24127, 2025). RNeRV = \"Rabbit "
            "NeRV\", a state-of-the-art CONFIGURATION of disentangled "
            "NeRV-family components (NOT a fundamentally new architecture; "
            "+1.27% PSNR avg over the best per-video alternative at equal "
            "training time on 7 UVG 1080p videos). Canonical position: the "
            "NeRV-family is a COMPONENT DESIGN SPACE — the right carrier is the "
            "right CONFIGURATION (positional encoding, decoder split, capacity) "
            "for the byte budget + content, found by disentangled component "
            "ablation; AND the honest empirical caveat that the VINRB hybrid-INR "
            "compression eval (HNeRV/DiffNeRV/DivNeRV) is currently unreliable "
            "(set_zero operates on non-bitstream model parts). The "
            "carrier-configuration + measurement-honesty voice."
        ),
        relevance_tokens=(
            "rnerv", "rabbit_nerv", "vinrb", "hnerv",
            "neural_video_representation", "implicit_video_codec",
            "carrier_architecture", "carrier_configuration",
            "component_design_space", "disentangled_ablation",
            "hybrid_inr_bitstream_caveat", "measurement_honesty",
        ),
        canonical_reference_path=(
            "arXiv:2304.02633 (HNeRV 2nd author) + arXiv:2506.24127 "
            "(VINRB/RNeRV lead author) + github.com/mgwillia/vinrb (UMD "
            "Shrivastava lab); lane research return "
            "lane_inverse_steganalysis_optimal_full_stack_20260601"
        ),
    ),
    # --- 2026-06-01 SNeRV + HiNeRV carrier-architecture seats (lane
    # lane_inverse_steganalysis_optimal_full_stack_20260601, deep-research
    # subagent follow-on) per Catalog #110 APPEND-ONLY + #346. These are the
    # VERIFIED principal authors of the two OPERATOR-APPROVED top-priority
    # PHASE-2 carrier candidates for the inverse-steganalysis full stack
    # (§7 GREEN: the L∞ pose-Fisher margin-budget allocation beats L2 by 56.9%
    # at equal rate, POSE-DOMINATED). SNeRV = the wavelet-override
    # RECONCILIATION (stores ONLY LF DWT coefficients + GENERATES the
    # byte-heavy HF detail via a decoder -> cures the Z8 "wavelet blobs too big"
    # disease WHILE keeping the exact orthonormal-DWT synthesis adjoint that
    # closes G3 by construction). HiNeRV = the cheapest measured RD INR
    # (-72.3% bit-rate vs HNeRV on UVG) whose prune+6-bit-QAT+arithmetic-coding
    # pipeline IS the cheapness engine. Author lists verified across >=2
    # independent sources (arXiv abstract + ECCV/NeurIPS proceedings + GitHub +
    # institutional pages); see lane research return. These seats are the
    # AUTHORITATIVE voices on the SNeRV LF-store/HF-generate split + the HiNeRV
    # hierarchical-encoding + compression-pipeline cheapness engine.
    CouncilSeat(
        name="Kang_SNeRV",
        role="grand_council",
        canonical_position_summary=(
            "Je-Won Kang (Ewha Womans University, Dept. of Electronic & "
            "Electrical Engineering; corresponding/senior author) — with "
            "co-authors Jina Kim + Jihoo Lee — of SNeRV: Spectra-preserving "
            "Neural Representation for Video (ECCV 2024). Canonical position: "
            "neural INRs have a SPECTRAL BIAS (learn LF faster than HF), so the "
            "right carrier STORES only the low-frequency 2D-DWT subband (CLL) "
            "as encoded features and GENERATES the three high-frequency "
            "subbands (CLH/CHL/CHH) via a decoder (High-Frequency Restorer) "
            "rather than storing them. This is the WAVELET-OVERRIDE "
            "RECONCILIATION: it cures the Z8 'wavelet detail blobs are too big' "
            "disease (the byte-heavy detail is GENERATED, not stored) while "
            "keeping the exact orthonormal-DWT synthesis adjoint (closes G3 by "
            "construction) and aligning with the SegNet stride-2-stem detail-"
            "subband structural dead-zone. The AUTHORITATIVE voice on the "
            "store-LF/generate-HF carrier split + the wavelet-domain + pixel-"
            "space dual loss."
        ),
        relevance_tokens=(
            "snerv", "spectra_preserving", "wavelet_hybrid",
            "lf_store_hf_generate", "dwt_subband_split",
            "high_frequency_restorer", "spectral_bias",
            "neural_video_representation", "implicit_video_codec",
            "carrier_architecture", "carrier_cheapness",
            "wavelet_override_reconciliation", "synthesis_adjoint_g3",
        ),
        canonical_reference_path=(
            "arXiv:2501.01681 (SNeRV, ECCV 2024) + "
            "link.springer.com/chapter/10.1007/978-3-031-73001-6_19 (ECCV "
            "proceedings) + pure.ewha.ac.kr (Ewha Womans Univ EEE) + "
            "github.com/qwertja/SNeRV; lane research return "
            "lane_inverse_steganalysis_optimal_full_stack_20260601"
        ),
    ),
    CouncilSeat(
        name="Bull_HiNeRV",
        role="grand_council",
        canonical_position_summary=(
            "David Bull (University of Bristol, Visual Information Lab; senior "
            "author) — with co-authors Ho Man Kwan (lead), Ge Gao, Fan Zhang, "
            "and Andrew Gower (BT, Immersive Content & Comms Research) — of "
            "HiNeRV: Video Compression with Hierarchical Encoding-based Neural "
            "Representation (NeurIPS 2023) and the fully-end-to-end successor "
            "NVRC: Neural Video Representation Compression (NeurIPS 2024). "
            "Canonical position: the cheapest INR carrier RD comes from "
            "(a) HIERARCHICAL multi-resolution local feature grids whose "
            "parameter count scales with the upsampling factor NOT the "
            "resolution (base grid 150x18x32 x8ch on UVG, upsample factors "
            "(5,3,2,2)), depthwise-conv + MLP + bilinear-interp blocks, and a "
            "unified frame+patch encoding; and (b) a DISCIPLINED COMPRESSION "
            "PIPELINE — adaptive ~15% pruning + fine-tune, 6-bit "
            "quantization-aware training (Quant-Noise, no STE), arithmetic "
            "entropy coding — that is the CHEAPNESS ENGINE (-72.3% bit-rate vs "
            "HNeRV, -43.4% vs DCVC on UVG; NVRC then closes the loop "
            "end-to-end with per-group learned quantization + hierarchical "
            "coding of all network/quant/entropy params, first INR codec to "
            "beat VTM-RA on long sequences). The AUTHORITATIVE voice on the "
            "hierarchical-encoding carrier + the prune/quantize/entropy-code "
            "cheapness engine (the learned-renderer Phase-2 fallback with a "
            "dense decoder-VJP adjoint)."
        ),
        relevance_tokens=(
            "hinerv", "nvrc", "hierarchical_encoding",
            "multi_resolution_feature_grid", "depthwise_conv_mlp_interp",
            "frame_patch_unified", "prune_quantize_entropy_code",
            "quant_noise_qat", "arithmetic_coding",
            "neural_video_representation", "implicit_video_codec",
            "carrier_architecture", "carrier_cheapness",
            "learned_renderer_fallback", "decoder_vjp_adjoint",
        ),
        canonical_reference_path=(
            "arXiv:2306.09818 (HiNeRV, NeurIPS 2023) + "
            "proceedings.neurips.cc/paper_files/paper/2023/hash/"
            "e5dc475c370ff42f2f96dddf8191a40c (NeurIPS proceedings) + "
            "openreview.net/forum?id=CpoS56pYnU + github.com/hmkx/HiNeRV "
            "(MIT license) + arXiv:2409.07414 (NVRC, NeurIPS 2024, same "
            "authors); lane research return "
            "lane_inverse_steganalysis_optimal_full_stack_20260601"
        ),
    ),
    CouncilSeat(
        name="FrankNielsen",
        role="grand_council",
        canonical_position_summary=(
            "Computational information geometry — the ALGORITHMS on statistical "
            "manifolds behind our frozen-scorer objects. Canonical voice on: "
            "(1) the Fisher-Rao metric + its computable surrogates (our margin "
            "field = Fisher surrogate, Pearson 0.978) and CURVED Bregman "
            "divergences for the anisotropic boundary annulus where the "
            "flat-interior approximation breaks; (2) Bregman-Voronoi diagrams + "
            "CHERNOFF INFORMATION as the geometry of the SegNet argmax partition "
            "(d_seg = disagreement on a Bregman-Voronoi boundary in logit space; "
            "the Chernoff point/information is the principled bits-per-flip floor "
            "for Lever-D); (3) dually-flat Legendre/mirror-descent duality — the "
            "native language for CE = mirror descent = natural gradient (our "
            "tau=eps=hbar result) and the tau-anneal curriculum as a deformation "
            "of the logsumexp Bregman generator (tau->0 = tropical/max-plus); "
            "(4) DUO BREGMAN pseudo-divergences (two different convex generators) "
            "as a candidate closed-form reason our measured seg-perp-pose "
            "decoupling is additive; (5) mixture-simplification / k-MLE / Bregman "
            "k-means for compressing the exponential-family sufficient statistics "
            "(the task-space quotient codec). Added to the grand council "
            "permanently per operator 2026-07-06 after his OIST talk 'Geometric "
            "information theory: A hub to information sciences'. Sits beside "
            "Shannon (information-theory grounding) + MacKay (IT+inference) as the "
            "information-geometry seat."
        ),
        relevance_tokens=(
            "information_geometry", "fisher_rao_metric", "fisher_information",
            "bregman_divergence", "curved_bregman", "dually_flat",
            "legendre_transform", "mirror_descent", "natural_gradient",
            "chernoff_information", "bregman_voronoi", "argmax_partition_geometry",
            "logsumexp_generator", "tropical_limit", "duo_bregman_pseudo_divergence",
            "seg_perp_pose_decoupling", "mixture_simplification", "k_mle",
            "exponential_family", "task_space_sufficient_statistic",
            "margin_field_fisher_surrogate", "statistical_manifold_projection",
        ),
        canonical_reference_path=(
            "Frank Nielsen (Sony CSL / Ecole Polytechnique) — 'Geometric "
            "information theory: A hub to information sciences' (OIST talk, "
            "shared by operator 2026-07-06) + 'An Elementary Introduction to "
            "Information Geometry' (Entropy 2020) + the Bregman-Voronoi / "
            "Chernoff-information / mixture-simplification corpus; maps to our "
            "unified level-set flow (project_unified_variational_levelset_flow_"
            "20260701) + deepmath_amortizing_argmax_maslov_caustic_tau_eps_hbar_"
            "20260704 (CE=mirror-descent, tau=eps=hbar) + the margin-saliency / "
            "Lever-D surfaces"
        ),
    ),
)


# Quick lookup index for validation paths.
_INNER_NAMES = frozenset(seat.name for seat in INNER_COUNCIL)
_GRAND_NAMES = frozenset(seat.name for seat in GRAND_COUNCIL)
_ALL_SEATS_BY_NAME: dict[str, CouncilSeat] = {
    seat.name: seat for seat in INNER_COUNCIL + GRAND_COUNCIL
}


# ---------------------------------------------------------------------------
# Attendee-name normalization (Catalog #346 detector-scope cure, 2026-08-25).
#
# BUG CLASS: the roster matcher compared attendee strings to seat names by EXACT
# equality. Council memos legitimately record the SAME canonical person under a
# spelling variant — `Ballé` (the person's actual spelling) vs the ASCII seat id
# `Balle`; `PR95-author` vs `PR95Author`; `MacKay_Memorial` vs `MacKay`;
# `AssumptionAdversary` vs `Assumption-Adversary`; `Schmidhuber-LEAD` vs
# `Schmidhuber`. Exact matching reported those seats as ABSENT although the memo
# seated them — a FALSE under-rostering report, i.e. a detector defect, not a
# roster defect.
#
# The cure normalizes IDENTITY-PRESERVING decoration only:
#   * unicode accent folding (Ballé -> balle)
#   * case folding
#   * parenthetical qualifiers  ("Tishby (memorial seat)" -> tishby)
#   * ROLE annotations as separated suffix tokens (LEAD / CO-LEAD / memorial /
#     seat) — a role is not an identity
#   * punctuation/whitespace ( `PR95-author` / `Assumption-Adversary` )
#
# It deliberately does NOT strip IDENTITY-BEARING qualifiers. In particular
# `mentor` is NOT stripped, because `TimeTraveler` (mentor) and
# `TimeTravelerProtege` are DISTINCT canonical seats per CLAUDE.md "Grand
# Council (advisory)"; folding `mentor` would risk collapsing them. Likewise the
# `_Grand` sister-seat suffix is preserved so `Rudin` never satisfies
# `Rudin_Grand`. `_check_seat_key_injectivity` (asserted at import) refuses any
# rule set that maps two distinct canonical seats onto one key — the structural
# guard against the collapse-per-key-distinctions failure class.
# ---------------------------------------------------------------------------

# ORDER IS LOAD-BEARING: every "co-lead" spelling must be consumed before the
# bare "lead" rule runs, otherwise "shannon co-lead" would strip to "shannon co"
# and fold to the wrong key "shannonco".
_SEAT_KEY_ROLE_ANNOTATIONS: tuple[str, ...] = (
    "co-lead",
    "co lead",
    "colead",
    "lead",
    "memorial",
    "seat",
)


def canonical_seat_key(name: str) -> str:
    """Return the identity key for a council attendee/seat name.

    Folds accents, case, parenthetical qualifiers, separated ROLE annotations
    (LEAD / CO-LEAD / memorial / seat), and punctuation. Identity-bearing
    qualifiers (`mentor`, the `_Grand` sister-seat suffix) are PRESERVED.

    Returns "" for empty/whitespace input. Note that the role rules consume a
    SEPARATED suffix only, so a standalone role word folds to itself (bare
    "LEAD" -> "lead"), not to "". Either way it resolves to no canonical seat,
    which is the property callers rely on; `_check_seat_key_injectivity`
    guarantees no real seat can be reached by decoration alone.
    """
    if not name or not isinstance(name, str):
        return ""
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = folded.lower()
    # Parenthetical qualifiers are annotation, never identity.
    folded = re.sub(r"\([^)]*\)", " ", folded)
    for annotation in _SEAT_KEY_ROLE_ANNOTATIONS:
        folded = re.sub(
            rf"[-_\s]{re.escape(annotation)}(?![a-z0-9])", " ", folded,
        )
    return re.sub(r"[^a-z0-9]+", "", folded)


def _check_seat_key_injectivity() -> None:
    """Refuse a normalization rule set that collapses two canonical seats.

    Import-time structural guard: `canonical_seat_key` MUST be injective over
    the canonical roster, otherwise attendance for one seat would silently
    satisfy a different seat.
    """
    seen: dict[str, str] = {}
    for seat in INNER_COUNCIL + GRAND_COUNCIL:
        key = canonical_seat_key(seat.name)
        if not key:
            raise ValueError(
                f"canonical_seat_key collapsed seat {seat.name!r} to an empty key"
            )
        if key in seen:
            raise ValueError(
                "canonical_seat_key is NOT injective over the canonical roster: "
                f"{seen[key]!r} and {seat.name!r} both map to {key!r}"
            )
        seen[key] = seat.name


_check_seat_key_injectivity()

# Key-addressed view of `_ALL_SEATS_BY_NAME` (kept in lockstep with it, so the
# name-addressed map does not become orphaned by the key-addressed lookup).
_ALL_SEAT_KEYS: dict[str, CouncilSeat] = {
    canonical_seat_key(name): seat
    for name, seat in _ALL_SEATS_BY_NAME.items()
}


# ---------------------------------------------------------------------------
# Seat availability timeline (Catalog #346 anachronism cure, 2026-08-25).
#
# The roster GREW over time per Catalog #110/#113 APPEND-ONLY. A deliberation
# memo cannot owe attendance from a seat that did not exist when it was
# convened; demanding one is an anachronistic requirement, not under-rostering.
#
# Provenance — first commit introducing each seat into THIS module
# (`git log --reverse -S'name="<seat>"' -- src/tac/canonical_council_roster.py`):
#   06feeecf17 / 77a2d0f38f  2026-05-19  module landing; all 14 INNER_COUNCIL
#       seats (incl. PR95Author, Rudin, Daubechies) + the 2026-05-19 grand
#       additions (TimeTraveler, TimeTravelerProtege, Rudin_Grand,
#       Daubechies_Grand). Cross-ref CLAUDE.md "Grand Council (advisory)"
#       2026-05-19 roster-maintenance-v2 + "Council conduct" 4-co-lead amendment.
#   4b7db346a6  2026-06-01  HaoChen_NeRV, Shrivastava_INR, Gwilliam_RNeRV
#   153f228232  2026-06-01  Kang_SNeRV, Bull_HiNeRV
#   002257c665  2026-07-06  FrankNielsen (per operator directive 2026-07-06)
#
# Seats absent from this map default to the module-landing date, so the map
# only ever RESTRICTS a demand for a post-landing seat; it can never excuse a
# seat that existed at convocation time.
# ---------------------------------------------------------------------------

CANONICAL_ROSTER_LANDING_UTC_YYYYMMDD = "20260519"

SEAT_FIRST_AVAILABLE_UTC: dict[str, str] = {
    "HaoChen_NeRV": "20260601",
    "Shrivastava_INR": "20260601",
    "Gwilliam_RNeRV": "20260601",
    "Kang_SNeRV": "20260601",
    "Bull_HiNeRV": "20260601",
    "FrankNielsen": "20260706",
}


def seat_available_at(seat_name: str, as_of_utc_yyyymmdd: str | None) -> bool:
    """True iff `seat_name` existed on the canonical roster at the given date.

    `as_of_utc_yyyymmdd=None` means "evaluate against the current roster" —
    every seat is available (the pre-2026-08-25 behavior).
    """
    if as_of_utc_yyyymmdd is None:
        return True
    first = SEAT_FIRST_AVAILABLE_UTC.get(
        seat_name, CANONICAL_ROSTER_LANDING_UTC_YYYYMMDD,
    )
    return str(as_of_utc_yyyymmdd) >= first


@dataclass(frozen=True)
class RosterValidationVerdict:
    """Verdict returned by `validate_council_dispatch_roster`.

    Fields:
        complete: True iff dispatched roster satisfies tier+topic requirements
        missing_inner_council: names of inner-council seats not dispatched
            (BLOCKING per CLAUDE.md "Experiment design — non-negotiable" for T2+)
        missing_co_leads: names of co-lead inner-council seats not dispatched
            (BLOCKING at T2+ per CLAUDE.md "Council conduct" 2026-05-19 amendment;
            ALL 4 co-leads — Shannon LEAD + Dykstra CO-LEAD + Rudin CO-LEAD +
            Daubechies CO-LEAD — MUST be present at every T2+ deliberation)
        missing_relevant_grand_council: names of grand-council seats topically
            relevant but not dispatched (RECOMMENDED for T3+, BLOCKING when
            5+ relevant seats omitted on T3+ per "Grand Council (advisory)")
        unknown_attendees: names not on the canonical roster (informational)
        topic_tokens: the topic tokens used for matching
        council_tier: the tier used for evaluation
    """

    complete: bool
    missing_inner_council: tuple[str, ...]
    missing_relevant_grand_council: tuple[str, ...]
    unknown_attendees: tuple[str, ...]
    topic_tokens: tuple[str, ...]
    council_tier: str
    missing_co_leads: tuple[str, ...] = ()

    def render(self) -> str:
        """Operator-readable summary."""
        lines: list[str] = [
            f"RosterValidationVerdict(tier={self.council_tier}, complete={self.complete})",
            f"  topic_tokens: {list(self.topic_tokens)}",
        ]
        if self.missing_co_leads:
            lines.append(
                f"  MISSING_CO_LEADS (blocking T2+): {list(self.missing_co_leads)} "
                f"-- 4-co-lead structure (Shannon/Dykstra/Rudin/Daubechies) "
                f"requires ALL 4 per CLAUDE.md 'Council conduct' 2026-05-19 amendment"
            )
        if self.missing_inner_council:
            lines.append(
                f"  MISSING_INNER_COUNCIL (blocking): {list(self.missing_inner_council)}"
            )
        if self.missing_relevant_grand_council:
            lines.append(
                f"  MISSING_RELEVANT_GRAND_COUNCIL: {list(self.missing_relevant_grand_council)}"
            )
        if self.unknown_attendees:
            lines.append(f"  unknown_attendees (informational): {list(self.unknown_attendees)}")
        return "\n".join(lines)


def required_attendees_for_topic(
    topic_tokens: Iterable[str],
    council_tier: str,
    *,
    as_of_utc_yyyymmdd: str | None = None,
) -> tuple[CouncilSeat, ...]:
    """Return canonical mandatory roster for a deliberation on the given topic.

    Args:
        topic_tokens: lowercase snake_case topic tokens (e.g. "pp_integration",
            "neural_compression", "engineering_shortcuts")
        council_tier: "T1" | "T2" | "T3" | "T4"

    Returns:
        Tuple of CouncilSeat objects that MUST be dispatched per the canonical
        roster + tier rules:
            - T1: working-group members only (no mandatory inner council)
            - T2+: ALL inner_council seats (sextet + 5 additional)
            - T3+: ALL inner_council + topically-relevant grand_council seats
                (matched by intersection with relevance_tokens)
            - T4: ALL inner_council + ALL grand_council (full symposium)

    Per CLAUDE.md "Council hierarchy: 4-tier protocol" + "Experiment design —
    non-negotiable" + "Grand Council (advisory)".

    `as_of_utc_yyyymmdd` (keyword-only, default None = current roster) restricts
    the requirement to seats that already existed on the canonical roster at
    that date, per `seat_available_at`. A deliberation cannot owe attendance
    from a seat appended after it was convened.
    """
    if council_tier not in ("T1", "T2", "T3", "T4"):
        raise ValueError(
            f"council_tier must be T1/T2/T3/T4: {council_tier!r}"
        )
    tokens = frozenset(t.strip().lower() for t in topic_tokens if t and isinstance(t, str))
    if council_tier == "T1":
        return ()
    required: list[CouncilSeat] = [
        seat for seat in INNER_COUNCIL
        if seat_available_at(seat.name, as_of_utc_yyyymmdd)
    ]
    if council_tier == "T2":
        return tuple(required)
    grand_available = tuple(
        seat for seat in GRAND_COUNCIL
        if seat_available_at(seat.name, as_of_utc_yyyymmdd)
    )
    if council_tier == "T4":
        return tuple(required) + grand_available
    # T3: inner council + topically-relevant grand council seats.
    for seat in grand_available:
        seat_tokens = frozenset(seat.relevance_tokens)
        if "any" in seat_tokens or seat_tokens & tokens:
            required.append(seat)
    return tuple(required)


def validate_council_dispatch_roster(
    dispatched_attendees: Iterable[str],
    topic_tokens: Iterable[str],
    council_tier: str,
    *,
    as_of_utc_yyyymmdd: str | None = None,
) -> RosterValidationVerdict:
    """Validate a council dispatch's attendee list against the canonical roster.

    Use BEFORE dispatching any T2+ council subagent. Per CLAUDE.md "Council
    conduct" non-negotiable: inner council MUST be present at every major
    deliberation.

    Per CLAUDE.md "Council conduct" 2026-05-19 amendment (4-co-lead structure):
    ALL 4 co-leads (Shannon LEAD + Dykstra CO-LEAD + Rudin CO-LEAD + Daubechies
    CO-LEAD) MUST be present at every T2+ deliberation in addition to the
    sextet's 5-of-6 quorum requirement. A T2+ deliberation missing ANY co-lead
    is structurally incomplete because the shared-leadership core cannot reach
    a binding decision without all 4 voices.

    Returns RosterValidationVerdict with `complete=False` iff:
      - any inner-council seat is missing (T2+; BLOCKING)
      - any co-lead is missing (T2+; BLOCKING per 2026-05-19 amendment)
      - 5+ topically-relevant grand-council seats are missing (T3+; structural
        under-rostering bug class)

    Per "Grand Council (advisory)": grand council members are CONSULTED on
    demand; missing 1-4 may be acceptable depending on the specialty match,
    but missing 5+ on a T3+ deliberation is structural under-rostering.
    """
    if council_tier not in ("T1", "T2", "T3", "T4"):
        raise ValueError(
            f"council_tier must be T1/T2/T3/T4: {council_tier!r}"
        )
    attendees = frozenset(a.strip() for a in dispatched_attendees if a and isinstance(a, str))
    # Match on the canonical identity key, not the raw string: a memo that
    # seated `Ballé` / `PR95-author` / `MacKay_Memorial` DID seat the canonical
    # person. `canonical_seat_key` is injectivity-guarded at import so this can
    # never let one seat's attendance satisfy a different seat.
    attendee_keys = frozenset(
        key for key in (canonical_seat_key(a) for a in attendees) if key
    )
    tokens = tuple(t.strip().lower() for t in topic_tokens if t and isinstance(t, str))
    required = required_attendees_for_topic(
        tokens, council_tier, as_of_utc_yyyymmdd=as_of_utc_yyyymmdd,
    )
    required_inner = tuple(
        seat for seat in required
        if seat.role in ("inner_council_sextet", "inner_council")
    )
    required_grand = tuple(
        seat for seat in required if seat.role == "grand_council"
    )
    missing_inner = tuple(
        seat.name for seat in required_inner
        if canonical_seat_key(seat.name) not in attendee_keys
    )
    missing_grand = tuple(
        seat.name for seat in required_grand
        if canonical_seat_key(seat.name) not in attendee_keys
    )
    # Per CLAUDE.md "Council conduct" 2026-05-19 amendment: compute the 4-co-lead
    # subset of missing inner seats so the operator-facing verdict surfaces them
    # distinctly. The shared-leadership core (Shannon LEAD + Dykstra CO-LEAD +
    # Rudin CO-LEAD + Daubechies CO-LEAD) is BLOCKING at T2+; sister members
    # are also blocking per the existing inner-council discipline but the
    # co-lead omission is a structurally distinct alert.
    missing_co_leads = tuple(
        seat.name for seat in required_inner
        if seat.is_co_lead and canonical_seat_key(seat.name) not in attendee_keys
    )
    unknown = tuple(
        name for name in sorted(attendees)
        if canonical_seat_key(name) not in _ALL_SEAT_KEYS
    )
    # Complete: no inner missing (T2+ always blocking; includes co-leads which
    # are a subset); no co-leads missing (T2+ BLOCKING per 2026-05-19 amendment);
    # grand missing OK up to 4 on T3 (advisory rule per "Grand Council
    # (advisory)"); 5+ missing on T3+ is structural under-rostering.
    inner_complete = len(missing_inner) == 0
    co_leads_complete = len(missing_co_leads) == 0
    grand_complete = True
    if council_tier in ("T3", "T4") and len(missing_grand) >= 5:
        grand_complete = False
    if council_tier == "T4" and len(missing_grand) > 0:
        # T4 symposium requires ALL grand-council seats.
        grand_complete = False
    complete = inner_complete and co_leads_complete and grand_complete
    return RosterValidationVerdict(
        complete=complete,
        missing_inner_council=missing_inner,
        missing_co_leads=missing_co_leads,
        missing_relevant_grand_council=missing_grand,
        unknown_attendees=unknown,
        topic_tokens=tokens,
        council_tier=council_tier,
    )


__all__ = [
    "CouncilSeat",
    "RosterValidationVerdict",
    "INNER_COUNCIL",
    "GRAND_COUNCIL",
    "CANONICAL_ROSTER_LANDING_UTC_YYYYMMDD",
    "SEAT_FIRST_AVAILABLE_UTC",
    "canonical_seat_key",
    "seat_available_at",
    "required_attendees_for_topic",
    "validate_council_dispatch_roster",
]
