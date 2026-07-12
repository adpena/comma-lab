# SPDX-License-Identifier: MIT
"""PR128 latent-click STRUCTURE law — isotropic slack, geometric magnitude, base-transfer (2026-07-12).

THE MEASURED FACTS (one law, three durable clauses + a transfer anchor)
-----------------------------------------------------------------------
Byte-exact differential forensics of PR128 `rhnerv_latent_polish` (a12dongithub, MIT)
vs its baseline (merged PR112) vs OUR PR110-lineage frontier, all decoded to the 600x28
uint8 latent q-grid (the physical click coordinate). Two independent exact-score-gated
discrete searches (theirs, ~2656 clicks; ours #399) on the SAME frozen HNeRV/PR95 stack.

CLAUSE A -- BASE-EQUALITY (coordinate identity): PR112's base q-grid and our PR110 base
q-grid are BYTE-IDENTICAL (0 of 16 800 cells differ; latent_raw both sha c760cab8). PR112
is a lossless ctx-recode of our PR110 payload. => the author's click set applies VERBATIM
to our latent table; it is not merely "similar", it is the same coordinate system.

CLAUSE B -- DIM-ISOTROPY (the pattern): accepted clicks distribute NEAR-UNIFORMLY over all
28 latent dims (CoV 0.238; hottest 5.6%, coldest 2.1%; 21/28 dims carry 80%; 99.7% of pairs
touched). The residual quantization slack after PR95 QAT is well-mixed -- there is NO few-dim
or few-pair shortcut; a full 28-dim x 600-pair sweep is mandatory (why the author used
diagonal batching, not a targeted subset).

CLAUSE C -- MAGNITUDE-GEOMETRIC + TEMPORAL-INCOHERENCE: |delta| ~ {1:0.863, 2:0.107, 3:0.027,
4:0.0026, 5:0.0008} (geometric-like decay; 13.7% are |>=2|, 3.1% |>=3| up to |5|). Adjacent
same-dim clicked cells 464 vs random-expected 440 (1.05x) and same-sign 51% (chance) => clicks
are per-pair-INDEPENDENT slack, NOT a smooth ego-motion trajectory. Only pair-locality is
exploitable; there is no temporal prior.

TRANSFER ANCHOR: importing PR128's final table onto our base packet (sidecar dropped,
byte-closed) scores advisory S 0.188070 vs our base 0.191110 (dS -0.00304; d_seg -2.6e-5,
bytes -605), reproducing PR128's published 0.187992 to within 8e-5 (cross-axis macOS-vs-Windows
CPU + our +33 B FP11-grammar overhead). Where the two independent searches touched the same
cell they agree on SIGN 93%.

NO-FAKE #7: borrowed mechanism structure on borrowed-lineage substrate -- a DEFENSIVE BANK /
search-accelerator prior, never an originality claim. Pointer 0.19108282 [contest-CPU]
UNMOVED; the exact contest-CPU row on the import candidate is MODAL-HOLD (operator GO).
"""

from __future__ import annotations

from tac.canonical_equations.equation import (
    RECALIBRATE_ON_NEW_ANCHORS,
    VERIFIED_VIA_EMPIRICAL_ANCHOR,
    CanonicalEquation,
    EmpiricalAnchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar

EQUATION_ID = "pr128_latent_click_structure_isotropic_geometric_v1"

_UTC = "2026-07-12T02:15:54Z"
_ADVISORY = "[macOS-CPU advisory]"
_MEMO = ".omx/research/pr128_click_forensics_20260712T021554Z.md"
_PRIORS = "experiments/results/pr128_click_import_forensics_20260712/click_search_priors_pr128.json"
_IMPORT_SHA = "196acd18e4ca"  # import candidate archive sha (12)


def build_pr128_latent_click_structure_v1() -> CanonicalEquation:
    """Build the click-structure law with its two MEASURED anchors."""

    anchor_structure = EmpiricalAnchor(
        anchor_id="pr128_click_structure_isotropy_magnitude_temporal_20260712",
        measurement_utc=_UTC,
        inputs={
            "decode": (
                "byte-exact q-grid (600x28 uint8) of PR128 final (sha cfd941de, latent_raw a7eba972 "
                "= author's git-LFS oid) and baseline (PR112 == our PR110 base, latent_raw c760cab8)"
            ),
            "click_set": "q_pr128_final - q_shared_base = 2656 clicks (includes folded PR101 sidecar)",
            "coordinate": "shared base is byte-identical PR112<->our-PR110 (0/16800 cells differ)",
        },
        predicted_output={
            "prior_hypothesis": "clicks concentrate on a few sloppy latent dims (few-dim shortcut)",
        },
        empirical_output={
            "base_equality_cells_differing": 0,
            "n_clicks_final": 2656,
            "n_clicks_v1": 2031,
            "dim_isotropy": {
                "coeff_of_variation": 0.238, "hottest_dim_frac": 0.056, "coldest_dim_frac": 0.021,
                "n_dims_for_80pct": 21, "pairs_touched_frac": 0.997,
                "verdict": "NEAR-ISOTROPIC -- few-dim shortcut FALSIFIED; full-dim sweep mandatory",
            },
            "magnitude_distribution": {"1": 0.863, "2": 0.107, "3": 0.027, "4": 0.0026, "5": 0.0008},
            "temporal_coherence": {"adjacent_same_dim": 464, "random_expected": 440, "ratio": 1.05,
                                   "adjacent_same_sign_frac": 0.51,
                                   "verdict": "NO temporal coherence; per-pair-independent slack"},
            "trajectory_v1_to_final": {"kept": 1905, "revised": 49, "dropped": 77, "added": 702,
                                       "verdict": "monotone greedy accumulation; depth-limited"},
            "cross_search_sign_agreement": "13/14 = 93% where our #399 and PR128 touched same cell "
                                           "(co-location 2.4x above chance)",
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method="byte_exact_latent_qgrid_diff_over_600x28_codes_shared_base",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria="a new public latent-polish PR or a refreshed #399 click set "
                                  "refreshes the isotropy/magnitude/temporal profile",
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )

    anchor_transfer = EmpiricalAnchor(
        anchor_id="pr128_table_import_onto_our_substrate_n600_advisory_20260712",
        measurement_utc=_UTC,
        inputs={
            "candidate": "PR128 final q-grid spliced onto our base packet (drop_sidecar=True), "
                         "byte-closed via tac.click_polish.FrozenPacket.repack_archive_bytes",
            "eval": "full-n600 chunked CPU render+score (tac.click_polish.render_and_score), "
                    "base reproduce = Q0 + sidecar",
        },
        predicted_output={"hypothesis": "base-equality => PR128 d_seg gain transfers ~fully to us"},
        empirical_output={
            "base_S": 0.1911103, "base_d_seg": 0.0005599, "base_bytes": 177169,
            "import_S": 0.1880699, "import_d_seg": 0.0005337, "import_bytes": 176564,
            "delta_S": -0.0030404, "delta_d_seg": -2.62e-05, "delta_bytes": -605,
            "pr128_published_cpu_S": 0.187992,
            "cross_axis_gap": "our import 0.188070 vs PR128 0.187992 = +8e-5 "
                              "(macOS-vs-Windows CPU + our +33 B FP11 grammar overhead)",
            "import_archive_sha256_12": _IMPORT_SHA,
        },
        residual=0.0,
        source_artifact=_MEMO,
        measurement_method="full_n600_advisory_eval_base_vs_import_byte_closed_own_results_dir",
        empirical_verification_status=VERIFIED_VIA_EMPIRICAL_ANCHOR,
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria="exact contest-CPU (Linux x86_64) row on the import candidate "
                                  "(MODAL-HOLD, operator GO) confirms/refutes the advisory delta",
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )

    return CanonicalEquation(
        equation_id=EQUATION_ID,
        name=(
            "PR128 latent-click structure: base-equality (0/16800 cells) + dim-isotropy (CoV 0.238, "
            "21/28 dims for 80%) + geometric magnitude (86/11/3/0.3/0.08 %) + temporal-incoherence "
            "(1.05x); PR128 table imports onto our substrate at advisory S 0.188070 (dS -0.00304)"
        ),
        one_line_summary=(
            "Exact-gated latent slack is ISOTROPIC over 28 dims (no shortcut), geometric in |delta|, "
            "temporally incoherent; base-identical => PR128 clicks transfer (import adv S 0.18807)."
        ),
        latex_form=(
            r"q^{\mathrm{base}}_{\mathrm{PR112}}=q^{\mathrm{base}}_{\mathrm{PR110}}\ (0/16800);\ "
            r"\mathrm{CoV}_{\dim}=0.238;\ P(|\delta|=k)\!\approx\!\{.86,.11,.03,.003,.0008\};\ "
            r"S^{\mathrm{adv}}_{\mathrm{import}}=0.18807\ (\Delta S=-3.04\times10^{-3})"
        ),
        python_callable_module_path="tac.click_polish",
        domain_of_validity={
            "vehicle": ["pr110_lineage_frontier_payload", "pr95_hnerv_frozen_decoder"],
            "verdict_scope": (
                "INSTANCE/FORMULATION -- measured on the PR128 final table (2656 clicks) vs the "
                "shared PR112/PR110 base and imported onto our frontier. Structure (isotropy, "
                "magnitude, temporal) is a robust property of this frozen stack's residual slack; "
                "the transfer S is [macOS-CPU advisory], exact row pending."
            ),
            "measurement_axis": ["macOS-CPU advisory"],
            "promotion_eligible": False,
            "note": "NO-FAKE #7 defensive bank + search-accelerator prior (borrowed mechanism, "
                    "borrowed lineage). Pointer moves only through the MODAL-HOLD exact row.",
        },
        units_in={"clicks": "latent_qgrid_integer_steps", "coordinate": "600x28_uint8_shared_base"},
        units_out={"dim_CoV": "dimensionless", "delta_S": "advisory_score_units",
                   "delta_bytes": "archive_bytes"},
        empirical_anchors=(anchor_structure, anchor_transfer),
        predicted_vs_empirical_residual={"structure_measured": 0.0, "transfer_measured": 0.0},
        last_calibration_utc=_UTC,
        next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
        canonical_consumers=(
            _MEMO,
            _PRIORS,  # the #399 search proposal distribution
            "tools/click_polish_local.py",  # consumes priors: full-dim sweep, extend to +-3, many rounds
            "staged_exact_eval_queue_MODAL_HOLD",
        ),
        canonical_producers=(
            "tac.click_polish",
            ".omx/research/pr128_click_forensics_20260712T021554Z.md",
        ),
        provenance=build_provenance_for_research_sidecar(
            sidecar_path=_MEMO,
            reactivation_criteria="exact contest-CPU row on the import candidate, a new public "
                                  "latent-polish PR, or a refreshed #399 click set supersedes these anchors",
            measurement_axis=_ADVISORY,
            hardware_substrate="macos_arm64",
        ),
    )


def populate_pr128_latent_click_structure_equation(
    *, path=None, lock_path=None, agent: str | None = None, subagent_id: str | None = None,
) -> CanonicalEquation:
    """Idempotent APPEND-ONLY registration (latest-row-wins). EQUATIONS leg of FEED-pr128forensics;
    DSL leg = N/A-with-reason (forensic differential of a frozen public archive, not a trainer lever)."""
    from tac.canonical_equations.registry import register_canonical_equation

    eq = build_pr128_latent_click_structure_v1()
    register_canonical_equation(
        eq, path=path, lock_path=lock_path, agent=agent, subagent_id=subagent_id,
        notes="pr128_latent_click_structure_20260712 (byte-exact forensics; base-equality TRUE; "
              "isotropic slack + geometric magnitude + temporal-incoherence; import advisory "
              "S 0.188070 dS -0.00304; NON-PROMOTABLE, exact row MODAL-HOLD)",
    )
    return eq


__all__ = [
    "EQUATION_ID",
    "build_pr128_latent_click_structure_v1",
    "populate_pr128_latent_click_structure_equation",
]
