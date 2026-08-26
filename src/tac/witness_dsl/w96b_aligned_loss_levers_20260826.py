"""DSL custody for the two sealed W96B aligned-objective launch configs.

WD3 intentionally exposes one launch flag: ``--compiled-config``.  The loss
law, tau schedule, STEP-ZERO pose supervision, optimizer floor, retention mode,
and resume identity live inside that validated config.  These levers therefore
select the complete reviewed payload rather than inventing shadow CLI flags.

The treatment is default-OFF in the trainer.  Only these two configs select the
exact CE1 expected-flip target-vs-best-other margin law.
"""

from __future__ import annotations

from tac.witness_dsl.curriculum_dsl import Lever

TRAINER_RELPATH = "experiments/ddm_wd3_scorer_aware_width_distillation.py"
VEHICLE = "ddm_s1a_w96_flattened_renderer"
CONFIG_ROOT = "/Volumes/APDataStore/pact/ddm_w96a_aligned_window/launch_requests"


def _lever_arguments(seed: int) -> dict[str, object]:
    config = f"{CONFIG_ROOT}/aligned_seed_{seed}.json"
    return {
        "name": f"w96b_exact_expected_flip_seed_{seed}",
        "notes": (
            "W96B exact CE1 law on the S1A W96 renderer: 100*mean(sigmoid(-(z_target-"
            "max_other)/tau)) over WD3's measured selected cells; linear tau 0.15->0.05 "
            "over the full 65-epoch window; CosineAnnealingLR eta_min=0.01*lr; exact pose "
            "score active at step zero. The compiled config also requires lossless "
            "content-addressed evaluation retention. Build-only registration: no scorer, "
            "Metal, training, or score claim."
        ),
        "constant_manifest": {
            "--compiled-config": {
                "value": config,
                "owner": "ddm_w96b",
                "vehicle": VEHICLE,
                "law": "expected_flip_target_vs_best_other_margin_v1",
                "seed": seed,
            }
        },
        "runtime_receipt_schemas": {
            "aligned_loss_trace": (
                "checkpoint history records seg_axis_expected_flip_margin_score and "
                "seg_axis_expected_flip_tau at every trained epoch"
            ),
            "evaluation_payload_custody": (
                "CAS_RETENTION_MANIFEST.json maps every logical payload to ordered immutable "
                "SHA-256 objects and restores byte-identically without symlinks"
            ),
        },
        "policy_contracts": {
            "score_claim": False,
            "default_off_byte_identity": True,
            "pose_active_from_step_zero": True,
            "all_payloads_retained": True,
            "vehicle": VEHICLE,
        },
    }


def lever_w96b_expected_flip_seed_20260815() -> Lever:
    """Select the sealed first-seed aligned config."""

    return Lever(
        overrides={"--compiled-config": f"{CONFIG_ROOT}/aligned_seed_20260815.json"},
        **_lever_arguments(20260815),
    )


def lever_w96b_expected_flip_seed_20260816() -> Lever:
    """Select the sealed second-seed aligned config."""

    return Lever(
        overrides={"--compiled-config": f"{CONFIG_ROOT}/aligned_seed_20260816.json"},
        **_lever_arguments(20260816),
    )
