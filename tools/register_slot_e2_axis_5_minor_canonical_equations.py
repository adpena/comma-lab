# SPDX-License-Identifier: MIT
"""Register 11 NEW canonical equations from Slot E2 axis-5-minor findings.

Per CLAUDE.md "Canonical equations + models registry" non-negotiable + Catalog
#344 + operator NON-NEGOTIABLE 2026-05-28 "Ensure no signal loss" cascade +
Slot E2 landing memo `feedback_slot_e2_canonical_equations_l33_l42_registration
_plus_negative_results_audit_v3_memory_falsification_corpus_combined_landed
_20260528.md` Phase D op-routable #1 (rank 1; $0 + ~30 min wall-clock; HIGH
compounding apparatus growth per highest-EV-shortest-wall-clock canonical
metric trichotomy).

Each equation is registered with a single PR-source-as-empirical-anchor
(residual=0.0) per the Slot G L33-L42 Phase 0 pattern. Auto-recalibration
fires when 3+ NEW empirical anchors land per equation from PR111+ training
waves per Catalog #371.

This script is APPEND-ONLY per Catalog #110/#113 HISTORICAL_PROVENANCE; it
emits NEW events only and does NOT mutate existing 302 registry rows.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from tac.canonical_equations.equation import (
    CanonicalEquation,
    EmpiricalAnchor,
    RECALIBRATE_ON_NEW_ANCHORS,
    _utc_now_iso,
)
from tac.canonical_equations.registry import (
    EVENT_REGISTERED,
    register_canonical_equation,
    update_equation_with_empirical_anchor,
)
from tac.provenance.builders import build_provenance_for_research_sidecar


SLOT_E2_MEMO_PATH = (
    ".omx/research/MEMORY_archive_pre_2026_05_26.md"  # canonical pointer; Slot E2 memo lives in ~/.claude/projects memory
)
SUBAGENT_ID = (
    "slot_i_signal_preservation_slot_e2_op_routable_1_plus_catalog_348_"
    "retroactive_sweep_20260529_0030cst"
)
REGISTRATION_NOTES = (
    "Slot E2 Phase D op-routable #1 axis-5-minor canonical equation registration "
    "per Slot E2 landing memo + operator NON-NEGOTIABLE 2026-05-28T23:35Z 'Ensure "
    "no signal loss' cascade; per Catalog #344 + #110/#113 APPEND-ONLY; sister "
    "of Slot G L33-L42 Phase 0 prerequisite registration pattern."
)

REACTIVATION_CRITERIA = (
    "Wave N+48 audit re-run against expanded canonical equation registry; "
    "per-equation calibration via tools/recalibrate_equation.py --equation-id "
    "<eq_id> once 3+ NEW empirical anchors land from PR111+ training waves OR "
    "operator-routed re-eval per CLAUDE.md 'Forbidden premature KILL' "
    "reactivation criteria refresh per Catalog #313 probe outcomes ledger; per "
    "Slot E2 Phase D operator-routable per 'iterate not force' standing directive."
)


# 11 axis-5-minor canonical equations from Slot E2 Phase B verdict matrix
# (rows #01, 02, 07, 09, 11, 17, 18, 20, 22, 23, 25 per the cascade rank-1 plan).
EQUATIONS: list[dict] = [
    # 1. Row #01 — AC bolt-on dominated by brotli q11 on small alphabet
    {
        "equation_id": "ac_dominated_by_brotli_q11_on_small_alphabet_v1",
        "name": "AC bolt-on dominated by brotli q11 on small-alphabet payloads",
        "one_line_summary": (
            "Arithmetic coding bolt-on cannot beat brotli quality=11 baseline on "
            "small-alphabet 24,188-byte payloads; AC pays per-symbol overhead "
            "without sufficient entropy headroom"
        ),
        "latex_form": (
            r"\Delta\text{archive}_{AC} - \Delta\text{archive}_{brotli11} > 0 "
            r"\text{ for } |\Sigma| \leq 256 \text{ and } N_{payload} \leq 30\text{KB}"
        ),
        "python_callable_module_path": (
            "tac.canonical_equations.ac_dominated_by_brotli_q11_on_small_alphabet"
        ),
        "domain_of_validity": {
            "codec_family": ["arithmetic_coding", "brotli_q11"],
            "payload_class": ["small_alphabet_sidecar_under_30kb"],
            "substrate_class": ["pr95_family", "hnerv_family"],
        },
        "units_in": {"alphabet_size": "int", "payload_bytes": "int"},
        "units_out": {"delta_archive_bytes": "int signed (AC - brotli; positive = AC loses)"},
        "memo_anchor": "feedback_ac_bolt_on_real_encoder_smoke_falsified_20260508.md",
    },
    # 2. Row #02 — Verdict re-tagging KILL → DEFER pending research exhaustion META-pattern
    {
        "equation_id": "verdict_re_tagging_kill_to_defer_pending_research_exhaustion_v1",
        "name": (
            "Verdict re-tagging KILL→DEFER-pending-research-exhaustion META-pattern"
        ),
        "one_line_summary": (
            "META-pattern for re-classifying historical KILL verdicts to DEFER when "
            "research-path exhaustion criteria per CLAUDE.md kill-as-last-resort "
            "non-negotiable were not satisfied at original landing"
        ),
        "latex_form": (
            r"V_{re-tagged} = \text{DEFER} \iff V_{original} = \text{KILL} "
            r"\land \exists r \in R_{plausible} : r \notin R_{tested}"
        ),
        "python_callable_module_path": (
            "tac.canonical_equations.verdict_re_tagging_kill_to_defer_pending_research_exhaustion"
        ),
        "domain_of_validity": {
            "verdict_class": ["KILL", "FALSIFIED", "DEAD", "RETIRED"],
            "discipline_axis": ["kill_as_last_resort"],
            "substrate_class": ["all"],
        },
        "units_in": {"original_verdict": "enum", "plausible_reactivation_paths": "list[str]"},
        "units_out": {"re_tagged_verdict": "enum {DEFER, KILL}"},
        "memo_anchor": "feedback_adversarial_audit_4_falsifications_DEFERRED_not_killed_20260507.md",
    },
    # 3. Row #07 — Markov-1 adaptive AAC pays 30KB small-sample cost
    {
        "equation_id": (
            "markov_1_adaptive_aac_pays_30kb_small_sample_cost_under_high_conditional_bins_v1"
        ),
        "name": "Markov-1 adaptive AAC pays 30KB small-sample cost under high conditional bin count",
        "one_line_summary": (
            "Adaptive arithmetic coding with Markov-1 conditioning pays ~30KB "
            "small-sample cost when conditional alphabet bins exceed 1.8M; "
            "oracle-vs-adaptive distinction per Shannon bound"
        ),
        "latex_form": (
            r"C_{adaptive} - C_{oracle} \approx 30000 \text{ bytes when } "
            r"N_{conditional\_bins} > 1.8 \times 10^6"
        ),
        "python_callable_module_path": (
            "tac.canonical_equations.markov_1_adaptive_aac_pays_30kb_small_sample_cost"
        ),
        "domain_of_validity": {
            "codec_family": ["arithmetic_coding_markov_1"],
            "payload_class": ["high_conditional_alphabet"],
            "substrate_class": ["mask_stream", "latent_stream"],
        },
        "units_in": {"conditional_bins": "int", "sample_count": "int"},
        "units_out": {"small_sample_cost_bytes": "int (adaptive cost above oracle bound)"},
        "memo_anchor": "feedback_markov1_aac_falsified_adaptive_smallSampleCost_20260507.md",
    },
    # 4. Row #09 — Hyperprior cannot reconstruct near-iid quantized symbols
    {
        "equation_id": (
            "hyperprior_architecture_cannot_reconstruct_near_iid_quantized_symbols_no_2d_locality_v1"
        ),
        "name": "Hyperprior architecture cannot reconstruct near-iid quantized symbols (no 2D locality)",
        "one_line_summary": (
            "PR101 CompressAI/Balle hyperprior architecture plateaus at "
            "rel_err 0.98-0.99 across 8 configurations because near-iid quantized "
            "symbols offer no exploitable 2D locality the hyperprior can model"
        ),
        "latex_form": (
            r"\text{rel\_err}_{hyperprior}(\theta) \to 0.985 \pm 0.005 "
            r"\text{ for } \theta \in \Theta_{tested,|\Theta|=8} "
            r"\text{ when } I(z_i; z_{i+1}) \approx 0"
        ),
        "python_callable_module_path": (
            "tac.canonical_equations.hyperprior_cannot_reconstruct_near_iid_quantized_symbols"
        ),
        "domain_of_validity": {
            "codec_family": ["compressai_balle_hyperprior", "factorized_prior"],
            "payload_class": ["near_iid_quantized_symbol_stream"],
            "substrate_class": ["pr101", "hnerv_family"],
        },
        "units_in": {"symbol_mutual_information": "float (nats)", "hyperprior_capacity": "int (params)"},
        "units_out": {"rel_err_plateau": "float in [0,1] (0.98-0.99 for near-iid)"},
        "memo_anchor": "feedback_pr101_compressai_balle_full_reactivation_FALSIFIED_with_capacity_20260507.md",
    },
    # 5. Row #11 — PR106 substrate-architecture mismatch (no mask channel)
    {
        "equation_id": "pr106_substrate_no_separate_mask_channel_lane_assumption_falsified_v1",
        "name": "PR106 substrate has no separate mask-channel lane (lane assumption falsified)",
        "one_line_summary": (
            "Lanes #05/#06 presume mask.mkv as separate channel in PR106; falsified "
            "per substrate-architecture mismatch (no separate mask channel; lanes "
            "need reformulation not class-kill)"
        ),
        "latex_form": (
            r"\text{lane}_{05,06} \text{ assumes } \text{archive.zip}/\text{masks.mkv} "
            r"\in \text{pr106\_substrate} = \emptyset \implies \text{lane FALSIFIED}"
        ),
        "python_callable_module_path": (
            "tac.canonical_equations.pr106_substrate_no_separate_mask_channel"
        ),
        "domain_of_validity": {
            "substrate_class": ["pr106", "pr106_latent_sidecar_r2"],
            "lane_class": ["mask_channel_targeted_lanes"],
            "verdict_class": ["substrate_architecture_mismatch"],
        },
        "units_in": {"lane_id": "str", "substrate_archive_members": "list[str]"},
        "units_out": {"mask_channel_present": "bool"},
        "memo_anchor": "feedback_pr106_no_mask_channel_lanes_05_06_falsified_20260504.md",
    },
    # 6. Row #17 — apogee_int4 NAIVE-PTQ falsified (not class-kill)
    {
        "equation_id": (
            "naive_ptq_int4_with_block_size_128_falsified_700x_pose_avg_class_collapse_v1"
        ),
        "name": "Naive PTQ int4 with block-size 128 falsified at 700x pose_avg class collapse",
        "one_line_summary": (
            "Naive PTQ int4 blocksize=128 produces ~700x pose_avg class collapse on "
            "apogee; QAT/LSQ/per-channel/smaller-block/outlier-handling unexplored "
            "— config falsified, paradigm NOT killed"
        ),
        "latex_form": (
            r"\text{pose\_avg}_{naive\_ptq\_int4, blocksize=128} / "
            r"\text{pose\_avg}_{fp16\_baseline} \approx 700"
        ),
        "python_callable_module_path": (
            "tac.canonical_equations.naive_ptq_int4_block_size_128_700x_pose_class_collapse"
        ),
        "domain_of_validity": {
            "quantization_method": ["naive_ptq"],
            "bit_width": [4],
            "block_size": [128],
            "substrate_class": ["apogee"],
        },
        "units_in": {"bit_width": "int", "block_size": "int", "quantization_method": "enum"},
        "units_out": {"pose_avg_amplification_factor": "float (vs fp16 baseline)"},
        "memo_anchor": "project_apogee_int4_FALSIFIED_score_1_43_dispatcher_VALIDATED_20260505.md",
    },
    # 7. Row #18 — Feature-space proxy ≠ contest-score regime (C1 world-model probe)
    {
        "equation_id": (
            "feature_space_proxy_loses_99_98_pct_to_independent_baseline_at_matched_dof_bit_budget_v1"
        ),
        "name": "Feature-space proxy loses 99.98% to independent baseline at matched DoF/bit-budget",
        "one_line_summary": (
            "FAIR-RSSM feature-space proxy loses 99.98% to independent baseline at "
            "matched DoF/bit-budget; proxy != contest-score regime; class-shift "
            "deferred pending contest-scale dispatch"
        ),
        "latex_form": (
            r"\text{score}_{feature\_proxy} / \text{score}_{independent\_baseline} "
            r"\approx 1 + 0.9998 \text{ at matched } DoF \land \text{matched bit-budget}"
        ),
        "python_callable_module_path": (
            "tac.canonical_equations.feature_space_proxy_loses_to_independent_baseline_at_matched_dof"
        ),
        "domain_of_validity": {
            "substrate_class": ["c1_world_model", "fair_rssm"],
            "measurement_axis": ["feature_space_proxy", "matched_dof_bit_budget"],
            "verdict_class": ["proxy_not_contest_score_regime"],
        },
        "units_in": {"dof_matched": "bool", "bit_budget_matched": "bool"},
        "units_out": {"score_amplification_factor": "float (proxy / independent baseline)"},
        "memo_anchor": (
            "project_c1_world_model_probe_v2_FAIR_RSSM_corroborates_loss_but_NOT_falsification_20260514.md"
        ),
    },
    # 8. Row #20 — Lane killed aggressively actually engineering bug not paradigm failure META
    {
        "equation_id": (
            "lane_killed_aggressively_actually_engineering_bug_not_paradigm_failure_class_v1"
        ),
        "name": "Lane killed aggressively when actually engineering bug, not paradigm failure",
        "one_line_summary": (
            "META-pattern: lanes killed at council-clean-3/3 often engineering bugs "
            "(dead-flag false-positives, CUDA OOM, scaffold-missing) not paradigm "
            "failures; per Catalog #307 distinction"
        ),
        "latex_form": (
            r"P(V_{KILL}=\text{engineering\_bug} \mid V_{KILL\_landed}) \approx 0.5 "
            r"\text{ pre-Catalog \#307 hardening 2026-05-16}"
        ),
        "python_callable_module_path": (
            "tac.canonical_equations.lane_killed_aggressively_actually_engineering_bug_not_paradigm"
        ),
        "domain_of_validity": {
            "verdict_class": ["KILL", "FALSIFIED"],
            "discipline_axis": ["paradigm_vs_implementation"],
            "substrate_class": ["all"],
        },
        "units_in": {"verdict_landed_date_utc": "str", "engineering_bug_evidence": "bool"},
        "units_out": {"is_engineering_bug_not_paradigm": "bool"},
        "memo_anchor": "project_killed_lanes_forensic_audit_20260428.md",
    },
    # 9. Row #22 — Half-res bottleneck destroys FastViT PoseNet luma (5x regression)
    {
        "equation_id": "half_res_bottleneck_destroys_fastvit_posenet_luma_5x_regression_v1",
        "name": "Half-res bottleneck destroys FastViT PoseNet luma (5x empirical regression)",
        "one_line_summary": (
            "Lane 7 PSD half-res bottleneck destroys FastViT PoseNet luma stride "
            "(5x regression); reactivation gated on PoseNet-aware luma-skip "
            "variant OR Phase 2 Lane 19 transfer"
        ),
        "latex_form": (
            r"\text{pose}_{half\_res} / \text{pose}_{full\_res} \approx 5 "
            r"\text{ on FastViT-T12-PoseNet at 384x512 } \to \text{ 192x256}"
        ),
        "python_callable_module_path": (
            "tac.canonical_equations.half_res_bottleneck_destroys_fastvit_posenet_luma"
        ),
        "domain_of_validity": {
            "substrate_class": ["lane_7_psd", "pixelshuffle_downscale"],
            "scorer_class": ["fastvit_t12_posenet"],
            "resolution_class": ["half_res_192x256"],
        },
        "units_in": {"resolution_ratio": "float (0.5 for half-res)", "scorer_class": "enum"},
        "units_out": {"pose_regression_factor": "float (vs full-res baseline)"},
        "memo_anchor": "project_lane_7_psd_killed_or_deferred_20260430.md",
    },
    # 10. Row #23 — Non-byte-aligned int7 packing Pareto-dominated by int8 at current packer
    {
        "equation_id": (
            "non_byte_aligned_int7_packing_exceeds_1_bit_savings_at_current_packer_pareto_dominated_by_int8_v1"
        ),
        "name": "Non-byte-aligned int7 packing exceeds 1-bit savings, Pareto-dominated by int8",
        "one_line_summary": (
            "Int7 at non-byte-aligned packer exceeds 1-bit theoretical savings; "
            "Pareto-dominated by int8; pair-int7 14-bit / arith-coded / mixed / "
            "hyperprior-conditioned codebook unexplored"
        ),
        "latex_form": (
            r"C_{int7, current\_packer}(N_{params}) > C_{int8}(N_{params}) - "
            r"\frac{N_{params}}{8} \text{ bytes when packer is non-byte-aligned}"
        ),
        "python_callable_module_path": (
            "tac.canonical_equations.non_byte_aligned_int7_packing_pareto_dominated_by_int8"
        ),
        "domain_of_validity": {
            "quantization_method": ["int7_non_byte_aligned"],
            "bit_width": [7],
            "packer_class": ["current_non_byte_aligned"],
            "substrate_class": ["apogee", "pr101"],
        },
        "units_in": {"bit_width": "int", "packer_alignment_bits": "int"},
        "units_out": {"pareto_dominated_by_int8": "bool"},
        "memo_anchor": "project_lane_apogee_int7_DEFERRED_pending_research_20260505.md",
    },
    # 11. Row #25 — Hard-argmax grayscale-LUT bolt-on does NOT preserve quality on 3ch-trained renderer
    {
        "equation_id": (
            "hard_argmax_grayscale_lut_bolt_on_does_not_preserve_quality_on_3ch_trained_renderer_v1"
        ),
        "name": "Hard-argmax grayscale-LUT bolt-on does NOT preserve quality on 3ch-trained renderer",
        "one_line_summary": (
            "Encoder-only hard-argmax grayscale-LUT bolt-on does NOT preserve "
            "quality on 3ch-trained renderer; sister Lane AL (training+inflate "
            "integration) IS correct compounding path"
        ),
        "latex_form": (
            r"\text{quality}_{grayscale\_lut\_bolt\_on, 3ch\_renderer} \ll "
            r"\text{quality}_{grayscale\_lut\_integrated, grayscale\_trained\_renderer}"
        ),
        "python_callable_module_path": (
            "tac.canonical_equations.hard_argmax_grayscale_lut_bolt_on_does_not_preserve_quality_3ch_renderer"
        ),
        "domain_of_validity": {
            "codec_family": ["grayscale_lut_bolt_on", "hard_argmax"],
            "substrate_class": ["3ch_trained_renderer"],
            "verdict_class": ["bolt_on_vs_integrated_path"],
        },
        "units_in": {"renderer_channel_count": "int (3 for RGB)", "bolt_on_vs_integrated": "enum"},
        "units_out": {"quality_preserved": "bool"},
        "memo_anchor": "project_lane_mm_v2_landed_2_63_falsified_20260429.md",
    },
]


def main() -> int:
    """Register the 11 canonical equations + emit 1 anchor each (residual=0.0)."""
    now_utc = _utc_now_iso()
    registered_count = 0
    skipped_count = 0
    errors: list[str] = []

    from tac.canonical_equations.registry import load_registry_events_lenient

    existing_events = load_registry_events_lenient()
    existing_ids = {r.get("equation_id", "") for r in existing_events}

    for eq_spec in EQUATIONS:
        eq_id = eq_spec["equation_id"]
        if eq_id in existing_ids:
            print(f"[skip] equation_id={eq_id} already registered; skipping per APPEND-ONLY")
            skipped_count += 1
            continue
        try:
            provenance = build_provenance_for_research_sidecar(
                sidecar_path=f"~/.claude/projects/-Users-adpena-Projects-pact/memory/{eq_spec['memo_anchor']}",
                reactivation_criteria=REACTIVATION_CRITERIA,
                measurement_axis="[research-signal]",
                hardware_substrate="unknown",
                captured_at_utc=now_utc,
            )
            equation = CanonicalEquation(
                equation_id=eq_spec["equation_id"],
                name=eq_spec["name"],
                one_line_summary=eq_spec["one_line_summary"],
                latex_form=eq_spec["latex_form"],
                python_callable_module_path=eq_spec["python_callable_module_path"],
                domain_of_validity=eq_spec["domain_of_validity"],
                units_in=eq_spec["units_in"],
                units_out=eq_spec["units_out"],
                empirical_anchors=(),
                predicted_vs_empirical_residual={},
                last_calibration_utc=now_utc,
                next_recalibration_trigger=RECALIBRATE_ON_NEW_ANCHORS,
                canonical_consumers=(
                    "tac.cathedral_consumers.canonical_equation_lookup_consumer",
                    "tools/list_canonical_equations.py",
                ),
                canonical_producers=(
                    "tools/audit_negative_results_v3_canonical_apparatus_mutation_gaps.py (planned)",
                    f"memory:{eq_spec['memo_anchor']}",
                ),
                provenance=provenance,
            )
            register_canonical_equation(
                equation,
                agent="claude",
                subagent_id=SUBAGENT_ID,
                notes=REGISTRATION_NOTES,
            )
            anchor = EmpiricalAnchor(
                anchor_id=f"{eq_spec['equation_id']}_initial_memo_anchor_{now_utc.replace(':', '').replace('-', '')}",
                measurement_utc=now_utc,
                inputs={"memo_anchor": eq_spec["memo_anchor"]},
                predicted_output="see memo",
                empirical_output="see memo",
                residual=0.0,
                source_artifact=f"memory:{eq_spec['memo_anchor']}",
                measurement_method="memory_corpus_audit_axis_5_minor",
                provenance=provenance,
            )
            update_equation_with_empirical_anchor(
                eq_spec["equation_id"],
                anchor,
                agent="claude",
                subagent_id=SUBAGENT_ID,
                notes=REGISTRATION_NOTES,
            )
            print(f"[ok] registered + anchored {eq_id}")
            registered_count += 1
        except Exception as exc:  # noqa: BLE001 — exhaustive error capture
            errmsg = f"{eq_id}: {type(exc).__name__}: {exc}"
            print(f"[FAIL] {errmsg}", file=sys.stderr)
            errors.append(errmsg)

    print()
    print("=" * 60)
    print(f"Registered: {registered_count}/{len(EQUATIONS)}")
    print(f"Skipped (already present): {skipped_count}")
    print(f"Errors: {len(errors)}")
    for e in errors:
        print(f"  - {e}")
    print("=" * 60)
    if errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
