from __future__ import annotations

from tac.witness_dsl.spec_next_launch_all_levers_20260713 import (
    FULL_VARIANT,
    TRIMMED_COMPLIANT_VARIANT,
    compile_next_launch_all_levers_ticket,
)


def _flags(cfg) -> set[str]:
    return set(cfg.typed.to_program().compile_trainer_argv())


def test_full_and_trimmed_memory_variants_are_typed_and_distinct() -> None:
    full = compile_next_launch_all_levers_ticket(variant=FULL_VARIANT)
    trimmed = compile_next_launch_all_levers_ticket(variant=TRIMMED_COMPLIANT_VARIANT)

    assert full.typed.validate_program() == []
    assert trimmed.typed.validate_program() == []
    assert "fresh_frequency_shift_init" in full.dsl_levers
    assert "fresh_frequency_shift_init" not in trimmed.dsl_levers
    assert "--self-orient" in _flags(full)
    assert "--self-orient" not in _flags(trimmed)
    assert full.dsl_program_manifest["memory_variant"] == FULL_VARIANT
    assert trimmed.dsl_program_manifest["memory_variant"] == TRIMMED_COMPLIANT_VARIANT
    assert trimmed.dsl_program_manifest["memory_trim"]["score_impact"].startswith("UNKNOWN")


def test_both_variants_remain_held_on_real_dependencies() -> None:
    for variant in (FULL_VARIANT, TRIMMED_COMPLIANT_VARIANT):
        cfg = compile_next_launch_all_levers_ticket(variant=variant)
        blockers = {row["id"] for row in cfg.dsl_program_manifest["launch_blockers"]}
        assert "D_A_EXACT_COMPONENT_TIMERS_MISSING" in blockers
        assert "D_B_EXACT_ENGAGEMENT_HOOK_MISSING" in blockers
        assert "MEMORY_WATERFILL_B2_UNMEASURED_N600" in blockers


def test_governed_launcher_resolves_the_trimmed_named_config() -> None:
    from tools import launch_witness_run as launcher

    cfg = launcher.derive_named_config(
        "next_launch_all_levers_trimmed_20260713",
        "experiments/results/mlx_fleet_gt_cache/gt_n600.npz",
        num_pairs=600,
        epochs=3000,
        overfit=True,
    )

    assert cfg.name == "next_launch_all_levers_trimmed_20260713"
    assert cfg.dsl_program_manifest["memory_variant"] == TRIMMED_COMPLIANT_VARIANT
    assert "--self-orient" not in _flags(cfg)
