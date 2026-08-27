# SPDX-License-Identifier: MIT
"""PR110-lineage byte-neutral click slack (#399 round-1, 2026-07-11).

THE MEASURED FACT (one anchor, two load-bearing clauses)
--------------------------------------------------------
The PR110-lineage frontier payload carries BYTE-NEUTRAL latent "click" slack: ±1 latent-table
clicks exist that leave the archive bytes IDENTICAL (40/40 single clicks archive-byte-invariant,
177,169 B unchanged) while lowering the full-n600 advisory score. Round-1 (pairs 0-47 only,
37 accepted clicks) measured incumbent advisory S 0.19109312 -> candidate 0.19101380,
**dS = -7.93e-05, seg-carried, at zero rate cost** `[macOS-CPU advisory]` NON-PROMOTABLE.

Credibility clause: the SAME harness's advisory number for the incumbent matches its real
contest-CPU report to ~1e-5 on the same bytes — the advisory axis tracks the exact axis
closely HERE (an axis-calibration fact, not a score claim; the exact row still gates).

Headroom clause (durable — do not re-derive): the CURRENT frontier archive is itself only
the n8 click-polish (pairs 0-7). Round-1 extended coverage to pairs 0-47 at ~2/3 dim
coverage; **592 pairs remain unpolished** and the search ledger is resumable. The mechanism
scales with pair coverage; the per-pair yield beyond 0-47 is UNMEASURED (no extrapolated
claim).

NO-FAKE #7: borrowed mechanism (PR128 click-polish) on borrowed-lineage substrate — a
DEFENSIVE BANK, never an originality claim. Pointer 0.19108282 [contest-CPU] UNMOVED until
the staged exact row runs (MODAL-HOLD).
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import (
    build_provenance_for_research_sidecar,
)

EQUATION_ID = "pr110_lineage_click_polish_byte_neutral_slack_v1"

_UTC = "2026-07-11T22:00:00Z"
_ADVISORY = "[macOS-CPU advisory]"
_MEMO = ".omx/research/click_polish_399_20260711T2200Z.md"
_CANDIDATE_SHA = "0872086672e7a0ff98e9e41e77e4c84d415b756eac3ba24db5ffec0d83e52969"

# The EXACT-ROW confirmation (2026-07-12): the fully-realized version of this same slack — PR128's
# full 2,656-click extraction spliced verbatim onto our byte-identical PR110 base — measured on the
# REAL contest-CPU axis via Modal. This is the reactivation criterion of anchor_round1, resolved.
_EXACT_MEMO = ".omx/research/modal_import_candidate_exact_eval_20260712.md"
_IMPORT_SHA = "196acd18e4ca10a3ab0d826436aa46014a44cba8a55eb4abf9931876cc7e98b5"
_CONTEST_CPU = "[contest-CPU]"


def build_pr110_lineage_click_polish_byte_neutral_slack_v1() -> CanonicalEquation:
    """Build the byte-neutral click-slack fact with its round-1 MEASURED anchor."""

    anchor_round1 = EmpiricalAnchor(
        anchor_id="click_polish_round1_pairs0_47_byte_neutral_ds_measured_20260711",
        measurement_utc=_UTC,
        inputs={
            "mechanism": (
                "±1 latent-table clicks on the PR110-lineage frontier payload (PR128 click-polish "
                "mechanism); K=48-scope search, accept gate = FULL-n600 advisory pass only "
                "(never K-scope extrapolation)"
            ),
            "substrate": "frontier archive ad02b012… (177,169 B), itself only the n8 polish (pairs 0-7)",
            "coverage": "pairs 0-47, ~2/3 dim coverage at the 3600s round cap; ledger resumable",
            "tool": "tools/click_polish_local.py -> src/tac/click_polish.py (commit 08322f68f1)",
        },
        predicted_output={
            "hypothesis": "the n8-polished frontier leaves byte-neutral click slack on unpolished pairs",
        },
        empirical_output={
            "incumbent_advisory_S_n600": 0.19109312,
            "candidate_advisory_S_n600": 0.19101380,
            "delta_S_advisory": -7.93e-05,
            "carried_by": "d_seg (incumbent d_seg 0.00055972; pose unchanged 2.942e-05)",
            "archive_bytes": "IDENTICAL 177,169 (40/40 single clicks archive-byte-invariant)",
            "accepted_clicks": 37,
            "candidate_archive_sha256": _CANDIDATE_SHA,
            "advisory_axis_credibility": (
                "incumbent advisory matches its contest-CPU report to ~1e-5 on the same bytes "
                "(same-harness calibration; NOT a score claim)"
            ),
            "n600_pass_walltime_s": "486-540 under live-run contention (full sweep round 6.5-8h -> K-scope + direct authority passes)",
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method="full_n600_advisory_eval_incumbent_vs_candidate_same_harness_byte_closed",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "the staged exact contest-CPU row on the candidate (MODAL-HOLD queue entry, "
                "operator GO) CONFIRMS or REFUTES the advisory delta; rounds 2+ over pairs "
                "48-599 refresh the candidate + this anchor"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )

    anchor_import_exact = EmpiricalAnchor(
        anchor_id="click_polish_import_full_extraction_contest_cpu_exact_20260712",
        measurement_utc="2026-07-12T06:35:14Z",
        inputs={
            "mechanism": (
                "PR128's FULL 2,656-click extraction (a12dongithub rhnerv_latent_polish, MIT) spliced "
                "VERBATIM onto our PR110 base — base-equality TRUE (0/16,800 latent cells differ; "
                "PR112 = lossless recode of our PR110, code-verified). Borrowed clicks, OUR substrate."
            ),
            # deterministic-bytes acceptable: deliberate contest-CPU authority receipt (upstream evaluate.py --device cpu axis), not a CUDA-fallback path
            "axis": "Modal contest-CPU (linux_x86_64), upstream/evaluate.py --device cpu, n=600",
            "call_id": "fc-01KXAGAT8JQA4BNH64FJ1SDC5N",
            "archive_sha256": _IMPORT_SHA,
        },
        predicted_output={
            "advisory": "macOS-CPU advisory 0.18806993 (this equation's slack, fully realized)",
            "hypothesis": "advisory transfers to real contest-CPU (CPU-selected clicks, no axis drift)",
        },
        empirical_output={
            "contest_cpu_S_n600": 0.1880443980,
            "d_seg": 0.0005334,
            "d_pose": 2.937e-05,
            "archive_bytes": 176564,
            "delta_vs_incumbent": -0.0030384262,
            "incumbent_contest_cpu": 0.19108282419209976,
            "advisory_to_exact_gap": "2.6e-5 (exact MARGINALLY LOWER; no CPU-axis drift — click structure held)",
            "pointer_moved": (
                "our_local_frontier_contest_cpu -> 0.1880443980; submitted_pr_number = null; "
                "NON-SUBMISSION borrowed-substrate defensive bank per operator 2026-07-12 "
                "(PRs reserved for the cgauge witness, never the borrowed import)"
            ),
            "modal_spend_usd": "~0.10 (CPU-only single axis)",
        },
        residual=2.6e-05,  # advisory-to-exact gap on this lineage's CPU axis
        source_artifact=_EXACT_MEMO,
        measurement_method="modal_contest_cpu_upstream_evaluate_recomputed_from_components_custody_verified",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_EXACT_MEMO,
            reactivation_criteria=(
                "rounds 2+ click search ON the import base (>0 further accepted clicks) or a CUDA-axis "
                "cross-check would refresh this anchor; the SUBMITTABLE frontier is the cgauge witness, "
                "measured separately (this bank never becomes a PR)"
            ),
            measurement_axis=_CONTEST_CPU,
            hardware_substrate="linux_x86_64_cpu",
        ),
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "PR110-lineage byte-neutral click slack: ±1 latent clicks with IDENTICAL archive "
            "bytes lower full-n600 advisory S (round-1 pairs 0-47: dS -7.93e-05, 37 clicks); "
            "the frontier itself is only n8-polished -> 592 pairs of unmeasured headroom"
        ),
        one_line_summary=(
            "Byte-neutral ±1 click slack exists in the frontier latent table: dS_advisory "
            "-7.93e-05 at zero rate cost (pairs 0-47); frontier only n8-polished, 592 pairs open."
        ),
        latex_form=(
            r"\exists\,\delta \in \{\pm 1\}^{K}:\ \mathrm{bytes}(A{+}\delta)=\mathrm{bytes}(A)"
            r"\ \wedge\ S_{600}(A{+}\delta) < S_{600}(A)\quad(\Delta S = -7.93\times 10^{-5},\ \text{pairs } 0{-}47)"
        ),
        python_callable_module_path="tac.click_polish",
        domain_of_validity={
            "vehicle": ["pr110_lineage_frontier_payload"],
            "verdict_scope": (
                "INSTANCE -- measured on the ad02b012 frontier archive, pairs 0-47. The slack's "
                "EXISTENCE is established; its magnitude on pairs 48-599 is UNMEASURED."
            ),
            "measurement_axis": ["macOS-CPU advisory"],
            "promotion_eligible": False,
            "note": (
                "defensive bank per NO-FAKE #7 (borrowed mechanism + borrowed lineage). The exact "
                "contest-CPU row is staged MODAL-HOLD; the pointer moves only through it."
            ),
        },
        units_in={"clicks": "latent_table_plus_minus_1_steps", "accept_gate": "full_n600_advisory_S"},
        units_out={"delta_S": "advisory_score_units", "delta_bytes": "archive_bytes (0 by construction)"},
        empirical_anchors=(anchor_round1, anchor_import_exact),
        predicted_vs_empirical_residual={"round1_measured": 0.0, "import_exact_contest_cpu_gap": 2.6e-05},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            _MEMO,
            "staged_exact_eval_queue_MODAL_HOLD",   # the operator-GO exact row
            "click_polish_rounds_2plus_pairs48_599",  # the resumed extension campaign
        ),
        canonical_producers=(
            "tac.click_polish",
            "tools/click_polish_local.py",
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria=(
                "exact contest-CPU row on the staged candidate (operator GO) or a refreshed "
                "candidate from rounds 2+ supersedes this instance-scope anchor"
            ),
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )


def populate_pr110_lineage_click_polish_byte_neutral_slack_equation(
    *,
    path=None,
    lock_path=None,
    agent: str | None = None,
    subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (latest-row-wins). EQUATIONS leg of FEED-399;
    DSL leg = N/A-with-reason (frozen-archive polish tool, not a trainer/curriculum lever)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_pr110_lineage_click_polish_byte_neutral_slack_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="click_polish_byte_neutral_slack_20260711 (#399 round-1 measured; candidate "
              "0872086672e7 staged MODAL-HOLD; 592-pair headroom; advisory NON-PROMOTABLE)",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "build_pr110_lineage_click_polish_byte_neutral_slack_v1",
    "populate_pr110_lineage_click_polish_byte_neutral_slack_equation",
]
