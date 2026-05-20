from __future__ import annotations

import math

from tac.optimization.percepta_microprogram_plan import (
    DEFAULT_BYTE_POLISH_FLOOR_BYTES,
    ExactEvalCustody,
    MicroprogramPrototypeSpec,
    build_microprogram_plan,
    default_weight_embedded_probe_spec,
    rate_delta_for_bytes,
)


def test_rate_delta_matches_contest_rate_formula() -> None:
    assert math.isclose(
        rate_delta_for_bytes(78),
        25.0 * 78.0 / 37_545_489.0,
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def test_default_weight_embedded_probe_is_prototype_only_until_measurement() -> None:
    plan = build_microprogram_plan(default_weight_embedded_probe_spec())

    assert plan.verdict == "PROTOTYPE_GO_PROMOTION_BLOCKED"
    assert plan.prototype_blockers == ()
    assert "exact_eval_custody_missing" in plan.promotion_blockers
    assert "does_not_beat_simple_q_or_byte_edit_hurdle" in plan.promotion_blockers
    assert plan.simple_edit_hurdle_delta_score == -rate_delta_for_bytes(DEFAULT_BYTE_POLISH_FLOOR_BYTES)
    assert all("tools/" in command for command in plan.cheapest_empirical_smoke)
    assert "baseline/data_dir/x" in plan.cheapest_empirical_smoke[0]
    assert "--archive-bin" in plan.cheapest_empirical_smoke[0]
    assert "archive.zip" not in plan.cheapest_empirical_smoke[0]
    assert "--baseline-raw" in plan.cheapest_empirical_smoke[2]


def test_full_wasm_interpreter_is_no_go_even_if_predicted_better() -> None:
    spec = MicroprogramPrototypeSpec(
        prototype_id="bad_full_wasm",
        surface="general_wasm_interpreter",
        opcodes=("i32.load", "i32.store"),
        expected_component_delta_score=-0.01,
        encoded_program_bytes=1,
    )

    plan = build_microprogram_plan(spec)

    assert plan.verdict == "NO_GO"
    assert "full_wasm_interpreter_not_byte_faithful" in plan.prototype_blockers
    assert any(row.startswith("forbidden_or_unbounded_opcode") for row in plan.prototype_blockers)


def test_scorer_or_network_runtime_blocks_prototype() -> None:
    spec = MicroprogramPrototypeSpec(
        prototype_id="bad_runtime",
        surface="decoder_side_microprogram",
        opcodes=("lookup_const4",),
        expected_component_delta_score=-0.001,
        scorer_free_inflate=False,
        network_free_inflate=False,
    )

    plan = build_microprogram_plan(spec)

    assert plan.verdict == "NO_GO"
    assert "inflate_loads_scorer" in plan.prototype_blockers
    assert "inflate_uses_network_or_external_io" in plan.prototype_blockers


def test_promotion_requires_beating_best_simple_edit_and_custody() -> None:
    custody = ExactEvalCustody(
        candidate_archive_sha256="a" * 64,
        runtime_tree_sha256="b" * 64,
        inflated_outputs_manifest_sha256="c" * 64,
        terminal_dispatch_claim=True,
        axis_tag="[contest-CUDA]",
    )
    spec = MicroprogramPrototypeSpec(
        prototype_id="tiny_gate",
        surface="weight_embedded_circuit",
        opcodes=("select_masked", "add_i8_saturating", "clamp_u8"),
        expected_component_delta_score=-0.0012,
        best_simple_edit_delta_score=-0.0010,
        custody=custody,
    )

    plan = build_microprogram_plan(spec)

    assert plan.verdict == "PROMOTION_GO"
    assert plan.projected_total_delta_score < plan.simple_edit_hurdle_delta_score
    assert plan.promotion_blockers == ()


def test_candidate_that_only_beats_byte_floor_but_not_q_edit_stays_blocked() -> None:
    custody = ExactEvalCustody(
        candidate_archive_sha256="a" * 64,
        runtime_tree_sha256="b" * 64,
        inflated_outputs_manifest_sha256="c" * 64,
        terminal_dispatch_claim=True,
        axis_tag="[contest-CPU]",
    )
    spec = MicroprogramPrototypeSpec(
        prototype_id="not_better_than_q",
        surface="weight_embedded_circuit",
        opcodes=("select_masked", "add_i8_saturating"),
        expected_component_delta_score=-0.0002,
        best_simple_edit_delta_score=-0.0005,
        custody=custody,
    )

    plan = build_microprogram_plan(spec)

    assert plan.verdict == "PROTOTYPE_GO_PROMOTION_BLOCKED"
    assert "does_not_beat_simple_q_or_byte_edit_hurdle" in plan.promotion_blockers


def test_runtime_patch_bytes_are_gated_but_not_charged_as_archive_rate() -> None:
    spec = MicroprogramPrototypeSpec(
        prototype_id="runtime_review_cost",
        surface="decoder_side_microprogram",
        opcodes=("lookup_const4",),
        expected_component_delta_score=0.0,
        runtime_patch_bytes=511,
    )

    plan = build_microprogram_plan(spec)

    assert plan.charged_rate_delta_score == 0.0
    assert plan.spec.charged_byte_delta == 0
    assert "runtime_patch_too_large" not in plan.prototype_blockers


def test_excessive_runtime_patch_bytes_block_prototype() -> None:
    spec = MicroprogramPrototypeSpec(
        prototype_id="runtime_too_large",
        surface="decoder_side_microprogram",
        opcodes=("lookup_const4",),
        expected_component_delta_score=-0.01,
        runtime_patch_bytes=513,
    )

    plan = build_microprogram_plan(spec)

    assert plan.verdict == "NO_GO"
    assert "runtime_patch_too_large" in plan.prototype_blockers


def test_pr110_live_file_touch_blocks_even_tiny_weight_circuit() -> None:
    spec = MicroprogramPrototypeSpec(
        prototype_id="live_pr110_touch",
        surface="weight_embedded_circuit",
        opcodes=("select_masked",),
        expected_component_delta_score=-0.01,
        touches_pr110_live_files=True,
    )

    plan = build_microprogram_plan(spec)

    assert plan.verdict == "NO_GO"
    assert "touches_pr110_live_submission_files" in plan.prototype_blockers
