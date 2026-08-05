# SPDX-License-Identifier: MIT
"""TK1 TR1 consumers: PE3 conditioning slot and cheapdct4 pose accounting.

These factories make the two TP1 owed consumers visible to the witness DSL.
Both are default-OFF in the trainer and carry ``score_claim=False``.  PE3 is a
conditioning prior only, never a target-label replacement.  cheapdct4 is an
accounting hook that decodes OD9's stage2 qcoeff carriage and reports OD9's
measured subset pose term; it is not a full joint-descent pose consumer.
"""

from __future__ import annotations

from tac.witness_dsl.curriculum_dsl import Lever

TRAINER_RELPATH = "experiments/train_tr1_partition_renderer_mlx.py"

PE3_EXPECTED_SECTION_SHA256 = (
    "5cc024ad32df7fedb18afb75dbed6be9c1af948dac826a1736cb1084949855c2"
)


def _nonempty_path(value: str, name: str) -> str:
    path = str(value).strip()
    if not path:
        raise ValueError(f"{name} must be a non-empty path")
    return path


def lever_tk1_pe3_conditioning(cache_path: str) -> Lever:
    """PE3 conditioning-only input prior with per-mode trust gates.

    Falsifier: any downstream run or receipt treats PE3 labels as replacement
    targets, or cannot suppress generator_pair_bisector independently of
    depth_conditioned_curve.
    """
    cache = _nonempty_path(cache_path, "cache_path")
    return Lever(
        name="tk1_pe3_conditioning_only",
        overrides={
            "--pe3-conditioning-cache": cache,
            "--pe3-conditioning-mode": "conditioning_only",
        },
        notes=(
            "TK1 PE3 consumer for TR1: parse the receiver-closed PE3EDGE1 section "
            "into boundary-distance/component-mode/transition conditioning channels. "
            "No label replacement loss exists; learned per-mode gates can drive the "
            "grammar prior to zero. score_claim=False."
        ),
        constant_manifest={
            "--pe3-conditioning-mode": {
                "value": "conditioning_only",
                "rung": "LC1-FORMULATION-NEGATIVE-COMPOSITION",
                "provenance": (
                    "LC1 measured PE3 ideal labels net -12884 at n32, so PE3 may re-enter "
                    "only as a conditioning prior consumed by learned TR1 descent."
                ),
            },
            "--pe3-conditioning-cache": {
                "value": cache,
                "rung": "RECEIVER-CLOSED-PE3-CUSTODY",
                "provenance": (
                    "Trainer rechecks the LC1/PK1 PE3 section sha256 "
                    f"{PE3_EXPECTED_SECTION_SHA256} before attaching conditioning features."
                ),
            },
        },
        runtime_receipt_schemas={
            "pe3_conditioning_init": "ON-only trainer telemetry and tr1_window_receipt field",
        },
        policy_contracts={
            "score_claim": False,
            "default_off_byte_identity": True,
            "label_replacement": False,
        },
    )


def lever_tk1_cheapdct4_pose_accounting(cache_path: str) -> Lever:
    """OD9 cheapdct4 qcoeff carriage accounting hook.

    Falsifier: a launch claims a pose-carrying base while the OD9 coefficients
    are stored outside the trainer/accounting surface, or promotes the n32 pose
    term as an n600/contest score.
    """
    cache = _nonempty_path(cache_path, "cache_path")
    return Lever(
        name="tk1_cheapdct4_pose_accounting",
        overrides={
            "--cheapdct4-pose-cache": cache,
            "--cheapdct4-pose-mode": "accounting",
        },
        notes=(
            "TK1 cheapdct4 consumer for TR1: decode OD9 stage2_qcoeffs and report "
            "OD9's measured n32 pose term in the trainer/composed-S receipt. This "
            "is accounting, not full in-loop joint-descent consumption. score_claim=False."
        ),
        constant_manifest={
            "--cheapdct4-pose-mode": {
                "value": "accounting",
                "rung": "TP1-FIRE-ORDER-ACCOUNTING-FIRST",
                "provenance": (
                    "TP1 refused pose-carrying-base claims until OD9 stage2_qcoeffs are "
                    "consumed by a trainer or admission/accounting hook."
                ),
            },
            "--cheapdct4-pose-cache": {
                "value": cache,
                "rung": "OD9-RECEIPT-CUSTODY",
                "provenance": (
                    "Trainer requires OD9_RECEIPT.json so packet sha, decoded qcoeffs, "
                    "projected bytes, and measured n32 d_pose term travel together."
                ),
            },
        },
        runtime_receipt_schemas={
            "cheapdct4_pose_accounting_init": (
                "ON-only trainer telemetry and tr1_window_receipt/composed-S receipt field"
            ),
        },
        policy_contracts={
            "score_claim": False,
            "default_off_byte_identity": True,
            "full_in_loop_consumption": False,
        },
    )
