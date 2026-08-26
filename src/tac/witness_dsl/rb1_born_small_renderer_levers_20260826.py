"""DSL custody for the four launch-disabled RB1 born-small renderer configs."""

from __future__ import annotations

from tac.witness_dsl.curriculum_dsl import Lever

TRAINER_RELPATH = "experiments/ddm_wd3_scorer_aware_width_distillation.py"
VEHICLE = "ddm_rb1_born_small_renderer"
CONFIG_ROOT = "/Volumes/APDataStore/pact/ddm_or1_orthogonal_sweep/next_renderer_born_small/configs"


def _lever(arm: str, seed: int) -> Lever:
    config = f"{CONFIG_ROOT}/{arm.lower()}_seed_{seed}.train.json"
    return Lever(
        name=f"rb1_{arm.lower()}_aligned_seed_{seed}",
        overrides={"--compiled-config": config},
        notes=(
            "RB1 exact BS3 archive-parseback field, byte-gated WD3 renderer, and W96B CE1 "
            "target-vs-best-other expected-flip law from step zero. The selected config is "
            "launch-disabled until the typed SR3/#1304/MAIN fire trigger is satisfied."
        ),
        constant_manifest={
            "--compiled-config": {
                "value": config,
                "owner": "ddm_rb1",
                "vehicle": VEHICLE,
                "arm": arm,
                "seed": seed,
                "target_object": "bs3_exact_archive_parseback",
                "law": "expected_flip_target_vs_best_other_margin_v1",
            }
        },
        runtime_receipt_schemas={
            "target_object": "ddm_rb1_born_small_target_object.v1",
            "candidate_archive": "RB1A nested exact BS3 body plus Brotli-q11 WD3 renderer",
            "evaluation_payload_custody": "ddm_w96b_evaluation_retention.v1 content-addressed chunks",
        },
        policy_contracts={
            "score_claim": False,
            "launch_now": False,
            "pose_active_from_step_zero": True,
            "all_payloads_retained": True,
            "resumable_from_disk": True,
            "vehicle": VEHICLE,
        },
    )


def lever_rb1_d56_seed_20260826() -> Lever:
    return _lever("D56", 20260826)


def lever_rb1_d56_seed_20260827() -> Lever:
    return _lever("D56", 20260827)


def lever_rb1_f64_seed_20260826() -> Lever:
    return _lever("F64", 20260826)


def lever_rb1_f64_seed_20260827() -> Lever:
    return _lever("F64", 20260827)
