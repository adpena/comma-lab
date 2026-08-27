# SPDX-License-Identifier: MIT
"""Canonical equation: the CLICK-POLISH RD law (equations leg of the first pointer-move since 2026-06-10).

The DAG owed this anchor at ``FEED-clickpolish-n8row`` / ``FEED-legdisposition-n8val`` (2026-07-10):
"equations OWED on the n600 row (click-polish RD law + waterline equivalents)". This module mints
that law and anchors it on the FIRST pointer-beating exact row.

LAW — ``clickpolish_exact_gated_discrete_latent_ratchet_v1``.
An exact-eval-gated discrete ``+/-1`` latent-click search that repacks BYTE-EXACT (``|archive|``
invariant) is a monotone confirmed-S ratchet driven PURELY by ``d_seg``: with ``ΔB = 0`` the rate
term drops out of ``ΔS = 100·Δd_seg + 25·ΔB/N``, leaving ``ΔS = 100·Δd_seg``; a click is accepted
iff ``ΔS < 0``. This is the SAME confirmed-ratchet family as the witness-scoped
``exact_metric_mc_finisher_v1`` but a DISTINCT member: a different vehicle (PR110-lineage entropy-recoded
frontier archive under the borrowed PR128 click-polish mechanism, NOT the witness head/palette) and a
weaker accept instrument (n8 in-container candidate SELECTION, with the n600 exact eval as the
pointer-moving CONFIRM — not an n600-only accept test). It is therefore NOT registered as an in-domain
anchor of ``exact_metric_mc_finisher_v1``.

EMPIRICAL ANCHOR — the first pointer-beating exact row since 2026-06-10 (this IS the measured evidence,
[contest-CPU] authoritative). Candidate archive sha ``ad02b012…`` (177,169 B; our frontier +
8 exact-gated ``±1`` latent clicks over the incumbent ``b46897…``) evaluated by ``upstream/evaluate.py
(deterministic-bytes acceptable: deliberate contest-CPU authority receipt (upstream evaluate.py --device cpu axis), not a CUDA-fallback path.)
--device cpu`` over 600 samples in a Modal Linux x86_64 gVisor container (torch 2.5.1+cpu; selection +
eval in the SAME container per the determinism mandate): SegNet 0.00055961 · PoseNet 0.00002942 ·
rate 0.00471878 → S = 0.19108282419209976 (canonical ``tac.contest_score.compute_contest_score``),
a net ΔS = −1.70e-5 vs the prior pointer 0.19109982419209975 from ONE ±1 round on 8 pairs, with
archive bytes unchanged (177,169 → 177,169) confirming the constant-|archive| clause.

BORROWED-SUBSTRATE ACCOUNTING (NO-FAKE #7 — DEFENSIVE BANK, not innovation): the click-polish
MECHANISM is borrowed (PR128 exact-score-gated discrete latent click-polish, external-unverified) and
the SUBSTRATE is OURS (PR110-lineage frontier, ``lane_pr110_payload_entropy_recode_20260610``); the
row banks a lower exact score, it does not claim an original method. See
``harvest_result.json:search.borrowed_substrate_accounting`` for the itemized reuse ledger.

KNOWN GAP: the CUDA axis is UNMEASURED for this candidate (same gap class as the prior CPU-pointer
lineage). means != ends: this DID move the contest-CPU pointer (0.19109982 → 0.19108282); the
in-flight n600-selection click-polish run is expected to supersede this row within hours.
"""
from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    ProvenanceEvidenceGrade,
    build_provenance_for_archive_member,
)

_UTC = "2026-07-10T16:42:45Z"
_ARCHIVE = "experiments/results/clickpolish_pr110_20260710/n8_validation/candidate_archive.zip"
_HARVEST = "experiments/results/clickpolish_pr110_20260710/n8_validation/harvest_result.json"
_HW = "linux_x86_64_cpu"
_AXIS = "[contest-CPU]"

CLICKPOLISH_EXACT_GATED_RATCHET_EQUATION_ID = (
    "clickpolish_exact_gated_discrete_latent_ratchet_v1"
)


def _contest_cpu_prov() -> object:
    return build_provenance_for_archive_member(
        _ARCHIVE,
        "x",
        _AXIS,
        _HW,
        ProvenanceEvidenceGrade.PROMOTABLE_EXACT_CONTEST_CPU,
        captured_at_utc=_UTC,
    )


def build_clickpolish_exact_gated_discrete_latent_ratchet_v1() -> CanonicalEquation:
    """Exact-gated discrete latent clicks at constant |archive| ratchet S via pure d_seg."""

    prov = _contest_cpu_prov()
    anchor = EmpiricalAnchor(
        anchor_id="clickpolish_pr110_n8click_pointer_move_20260710",
        measurement_utc=_UTC,
        inputs={
            "incumbent_archive_sha256": (
                "b46897267ded1e73a581dad57143f6c1cd181b515479d4efce40e4536d50e73e"
            ),
            "incumbent_S": 0.19109982419209975,
            "candidate_archive_sha256": (
                "ad02b0124cbb3405c23d3480ac16f12b4e48cbf6f75878dd77a5e621bebd079c"
            ),
            "n_accepted_clicks": 8,
            "archive_bytes": 177169,
            "delta_bytes": 0,
            "borrowed_substrate_accounting": (
                "DEFENSIVE BANK (NO-FAKE #7): mechanism borrowed (PR128 exact-score-gated "
                "discrete latent click-polish, external-unverified); substrate OURS "
                "(lane_pr110_payload_entropy_recode_20260610); see harvest_result.json "
                "search.borrowed_substrate_accounting"
            ),
            "known_gap": "CUDA axis UNMEASURED for this candidate (same class as prior CPU pointer)",
        },
        predicted_output={
            "constant_archive_bytes": True,
            "net_dS_sign": "negative",
            "rate_term_delta": 0.0,
            "mechanism": "dS = 100*d_seg gain at fixed |archive|",
        },
        empirical_output={
            "d_seg": 0.00055961,
            "d_pose": 0.00002942,
            "rate": 0.00471878,
            "S": 0.19108282419209976,
            "delta_S_vs_prior_pointer": -1.6999999999989246e-05,
            "n_samples": 600,
        },
        residual=0.0,
        source_artifact=_HARVEST,
        measurement_method=(
            # deterministic-bytes acceptable: deliberate contest-CPU authority receipt (upstream evaluate.py --device cpu axis), not a CUDA-fallback path
            "upstream/evaluate.py --device cpu, 600 samples, exact archive bytes; Modal Linux "
            "x86_64 gVisor container, torch 2.5.1+cpu; selection+eval SAME container "
            "(determinism mandate). S recomputed via tac.contest_score.compute_contest_score."
        ),
        empirical_verification_status="VERIFIED_VIA_EMPIRICAL_ANCHOR",
        provenance=prov,
    )
    return CanonicalEquation(
        equation_id=CLICKPOLISH_EXACT_GATED_RATCHET_EQUATION_ID,
        name=(
            "Click-polish RD law: exact-score-gated discrete latent ratchet at constant archive "
            "bytes (pure-d_seg descent; borrowed PR128 mechanism on OURS PR110-lineage substrate)"
        ),
        one_line_summary=(
            "At fixed |archive|, exact-eval-gated +/-1 latent clicks ratchet S monotonically via "
            "pure d_seg: dS = 100*d_seg (rate term unchanged); accept iff dS<0."
        ),
        latex_form=(
            r"\Delta S = 100\,\Delta d_{seg} + 25\,\Delta B/N,\ \Delta B=0 \Rightarrow "
            r"\Delta S = 100\,\Delta d_{seg};\ \text{accept iff }\Delta S<0"
        ),
        python_callable_module_path=(
            "tac.canonical_equations.clickpolish_exact_gated_ratchet_20260710:"
            "build_clickpolish_exact_gated_discrete_latent_ratchet_v1"
        ),
        domain_of_validity={
            "vehicle": [
                "PR110-lineage entropy-recoded frontier archive "
                "(lane_pr110_payload_entropy_recode_20260610); borrowed PR128 click-polish mechanism"
            ],
            "mechanism": [
                "exact-score-gated discrete +/-1 latent clicks; byte-exact splice-repack "
                "(|archive| invariant => rate term drops out)"
            ],
            "accept_instrument": [
                "n8 candidate SELECTION in-container; n600 exact eval is the pointer-moving CONFIRM"
            ],
            "measurement_axis": ["[contest-CPU] Linux x86_64 (Modal gVisor container)"],
            "excluded": [
                "witness head/palette mc-finisher (exact_metric_mc_finisher_v1: n600-only accept, "
                "different vehicle) — NOT an anchor of that equation",
                "any archive-bytes-changing edit (then the 25*ΔB/N rate term re-enters)",
                "CUDA axis (UNMEASURED for this candidate)",
            ],
        },
        units_in={"d_seg": "fraction", "archive_bytes": "bytes"},
        units_out={"delta_S": "score_units"},
        empirical_anchors=(anchor,),
        predicted_vs_empirical_residual={"clickpolish_n8click_dS": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            "tac.click_polish",
            "tools/click_polish_exact_search.py",
        ),
        canonical_producers=(_HARVEST,),
        provenance=prov,
    )


def populate_clickpolish_exact_gated_discrete_latent_ratchet_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration of the click-polish RD law.

    Triality legs: DAG = FEED-pointer-move-n8click (+ FEED-clickpolish-n8row / -n8val);
    DSL = N/A-with-rationale (a search TOOL, not a launch-config lever — see FEED-clickpolish-build);
    equations = this module. The measured row IS the pointer-move (contest-CPU 0.19108282).
    """
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_clickpolish_exact_gated_discrete_latent_ratchet_v1()
    register_canonical_equation(
        eq,
        path=path,
        lock_path=lock_path,
        agent=agent,
        subagent_id=subagent_id,
        notes=(
            "clickpolish_exact_gated_ratchet_20260710: DAG-owed click-polish RD law; anchored on "
            "the FIRST pointer-beating exact row since 2026-06-10 (contest-CPU 0.19108282, ΔS "
            "-1.7e-5, 8 exact-gated ±1 latent clicks, |archive| invariant). DEFENSIVE BANK "
            "(borrowed PR128 mechanism on OURS PR110 substrate); CUDA axis UNMEASURED."
        ),
    )
    return eq


__all__ = [
    "CLICKPOLISH_EXACT_GATED_RATCHET_EQUATION_ID",
    "build_clickpolish_exact_gated_discrete_latent_ratchet_v1",
    "populate_clickpolish_exact_gated_discrete_latent_ratchet_equation",
]
