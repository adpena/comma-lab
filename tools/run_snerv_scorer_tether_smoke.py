#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run a tiny SNeRV PR95 scorer-tether smoke and write a queue-consumable report."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    from tools.tool_bootstrap import ensure_repo_imports, repo_root_from_tool
except ModuleNotFoundError:  # pragma: no cover
    from tool_bootstrap import ensure_repo_imports, repo_root_from_tool

REPO_ROOT = repo_root_from_tool(__file__)
ensure_repo_imports(REPO_ROOT)

from tac.repo_io import write_json_artifact  # noqa: E402

SCHEMA = "snerv_scorer_tether_smoke.v1"
FALSE_AUTHORITY = {
    "score_claim": False,
    "score_claim_valid": False,
    "frontier_score_claim": False,
    "promotion_eligible": False,
    "promotable": False,
    "rank_or_kill_eligible": False,
    "ready_for_exact_eval_dispatch": False,
    "dispatch_attempted": False,
}
SEG_TETHER = "snerv_segnet_last_frame_distill"
POSE_TETHER = "snerv_posenet_yuv6_pair_distill"
GUARD_TETHER = "snerv_scorer_input_distribution_guard"


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    report = run_snerv_scorer_tether_smoke(steps=int(args.steps))
    write_json_artifact(args.output_json, report)
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "output_json": args.output_json.as_posix(),
                "passed": report["passed"],
                "blockers": report["blockers"],
                **FALSE_AUTHORITY,
            },
            sort_keys=True,
        )
    )
    return 0 if report["passed"] else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=2)
    return parser


def run_snerv_scorer_tether_smoke(*, steps: int = 2) -> dict[str, Any]:
    """Exercise the real shared MLX adapter path that SNeRV long training uses."""

    if int(steps) < 2:
        raise ValueError("steps must be >= 2 so lambda activation can be observed")
    try:
        import mlx.core as mx
    except Exception as exc:  # pragma: no cover - host dependent
        return {
            "schema": SCHEMA,
            "created_utc": datetime.now(UTC).isoformat(),
            "passed": False,
            "steps": int(steps),
            "blockers": [
                "snerv_scorer_tether_smoke_mlx_unavailable",
                f"snerv_scorer_tether_smoke_mlx_import_error:{type(exc).__name__}",
            ],
            **FALSE_AUTHORITY,
        }

    from tac.substrates._shared.mlx_score_aware.adapter import MlxScoreAwareAdapter
    from tac.substrates._shared.mlx_score_aware.dual_ascent import (
        build_default_nerv_train_time_dual_ascent_config,
    )

    bundle = _make_minimal_pr95_score_bundle()
    dual_config = build_default_nerv_train_time_dual_ascent_config(
        family="snerv",
        segnet_distillation_weight=1.0,
        pose_distillation_weight=1.0,
        scorer_input_distribution_guard_weight=1.0,
    )
    for constraint in dual_config["constraints"]:
        if constraint["constraint_id"] in {
            f"{SEG_TETHER}",
            f"{POSE_TETHER}",
            f"{GUARD_TETHER}",
        } or constraint["constraint_id"].endswith(
            (
                "segnet_last_frame_distill",
                "posenet_yuv6_pair_distill",
                "scorer_input_distribution_guard",
            )
        ):
            constraint["target"] = 0.0
            constraint.pop("target_fraction_of_initial", None)
    adapter = MlxScoreAwareAdapter(
        bundle,
        substrate_id="snerv_scorer_tether_smoke",
        pr95_faithful_curriculum_enabled=True,
        pr95_curriculum_total_epochs=8,
        train_time_dual_ascent_config=dual_config,
    )
    adapter.notify_global_epoch(0)
    batch = adapter.sample_batch(batch_size=2, seed=0)
    metrics_by_step: list[dict[str, float]] = []
    for _step in range(int(steps)):
        metrics = adapter.train_step(
            batch=batch,
            learning_rate=1.0e-3,
            loss_weights={
                "recon": 1.0,
                "distill": 1.0,
                "pose_distill": 1.0,
                "scorer_input_guard": 1.0,
            },
        )
        metrics_by_step.append({str(key): float(value) for key, value in metrics.items()})
    mx.eval(adapter.model.parameters())
    blockers = _smoke_blockers(metrics_by_step)
    telemetry_contract = _telemetry_contract(metrics_by_step, blockers)
    required_control_contract = _required_control_contract(telemetry_contract)
    return {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).isoformat(),
        "operation": "snerv_pr95_scorer_tether_dual_ascent_smoke",
        "steps": int(steps),
        "passed": not blockers,
        "blockers": blockers,
        "metric_summary": _metric_summary(metrics_by_step),
        "score_aware_long_training_telemetry_contract": telemetry_contract,
        "score_aware_long_training_required_control_contract": (
            required_control_contract
        ),
        "score_aware_long_training_scorer_input_distribution_guard_bound": True,
        "telemetry_contract": {
            "snerv_posenet_yuv6_pair_distill_missing_metric_must_be_zero": True,
            "snerv_segnet_last_frame_distill_missing_metric_must_be_zero": True,
            "snerv_scorer_input_distribution_guard_missing_metric_must_be_zero": (
                True
            ),
            "scorer_tether_lambdas_must_activate_by_final_step": True,
            "scorer_input_distribution_guard_lambda_must_activate_by_final_step": (
                True
            ),
        },
        "launch_gate": {
            "local_mlx_long_training_allowed_if_passed": not blockers,
            "blocks_snerv_long_training_if_failed": True,
        },
        **FALSE_AUTHORITY,
    }


def _smoke_blockers(metrics_by_step: list[dict[str, float]]) -> list[str]:
    if len(metrics_by_step) < 2:
        return ["snerv_scorer_tether_smoke_insufficient_steps"]
    first = metrics_by_step[0]
    final = metrics_by_step[-1]
    blockers: list[str] = []
    checks = (
        (SEG_TETHER, "loss_part_distill", "loss_part_pr95_stage_seg_surrogate"),
        (
            POSE_TETHER,
            "loss_part_pose_score_term",
            "loss_part_pr95_stage_pose_surrogate",
        ),
    )
    for tether, alias_key, source_key in checks:
        missing_key = f"dual_ascent_missing_metric__{tether}"
        metric_key = f"dual_ascent_metric__{tether}"
        lambda_key = f"dual_ascent_lambda__{tether}"
        if float(first.get(missing_key, 1.0)) != 0.0:
            blockers.append(f"{tether}_missing_on_first_smoke_step")
        if float(final.get(missing_key, 1.0)) != 0.0:
            blockers.append(f"{tether}_missing_on_final_smoke_step")
        if alias_key not in final:
            blockers.append(f"{tether}_canonical_alias_missing")
        if source_key not in final:
            blockers.append(f"{tether}_pr95_stage_surrogate_missing")
        if (
            alias_key in final
            and source_key in final
            and abs(float(final[alias_key]) - float(final[source_key])) > 1.0e-6
        ):
            blockers.append(f"{tether}_canonical_alias_mismatch")
        if metric_key not in final:
            blockers.append(f"{tether}_dual_metric_missing")
        if float(final.get(lambda_key, 0.0)) <= 0.0:
            blockers.append(f"{tether}_dual_lambda_inactive")
    guard_missing_key = f"dual_ascent_missing_metric__{GUARD_TETHER}"
    guard_metric_key = f"dual_ascent_metric__{GUARD_TETHER}"
    guard_lambda_key = f"dual_ascent_lambda__{GUARD_TETHER}"
    if float(first.get(guard_missing_key, 1.0)) != 0.0:
        blockers.append(f"{GUARD_TETHER}_missing_on_first_smoke_step")
    if float(final.get(guard_missing_key, 1.0)) != 0.0:
        blockers.append(f"{GUARD_TETHER}_missing_on_final_smoke_step")
    if "loss_part_scorer_input_distribution_guard" not in final:
        blockers.append(f"{GUARD_TETHER}_loss_part_missing")
    if guard_metric_key not in final:
        blockers.append(f"{GUARD_TETHER}_dual_metric_missing")
    if float(final.get(guard_lambda_key, 0.0)) <= 0.0:
        blockers.append(f"{GUARD_TETHER}_dual_lambda_inactive")
    return _ordered_unique(blockers)


def _metric_summary(metrics_by_step: list[dict[str, float]]) -> dict[str, Any]:
    final = metrics_by_step[-1] if metrics_by_step else {}
    keys = [
        "loss_part_distill",
        "loss_part_pr95_stage_seg_surrogate",
        "loss_part_pose_distill",
        "loss_part_pose_score_term",
        "loss_part_pr95_stage_pose_surrogate",
        f"dual_ascent_missing_metric__{SEG_TETHER}",
        f"dual_ascent_lambda__{SEG_TETHER}",
        f"dual_ascent_metric__{SEG_TETHER}",
        f"dual_ascent_missing_metric__{POSE_TETHER}",
        f"dual_ascent_lambda__{POSE_TETHER}",
        f"dual_ascent_metric__{POSE_TETHER}",
        "loss_part_scorer_input_distribution_guard",
        "loss_part_weighted_scorer_input_distribution_guard",
        "loss_part_stage_weight_scorer_input_distribution_guard",
        "loss_part_config_weight_scorer_input_distribution_guard",
        f"dual_ascent_missing_metric__{GUARD_TETHER}",
        f"dual_ascent_lambda__{GUARD_TETHER}",
        f"dual_ascent_metric__{GUARD_TETHER}",
    ]
    return {
        "step_count": len(metrics_by_step),
        "final": {key: final.get(key) for key in keys},
    }


def _telemetry_contract(
    metrics_by_step: list[dict[str, float]],
    blockers: list[str],
) -> dict[str, Any]:
    final = metrics_by_step[-1] if metrics_by_step else {}
    seg_lambda = float(final.get(f"dual_ascent_lambda__{SEG_TETHER}", 0.0))
    pose_lambda = float(final.get(f"dual_ascent_lambda__{POSE_TETHER}", 0.0))
    guard_lambda = float(final.get(f"dual_ascent_lambda__{GUARD_TETHER}", 0.0))
    guard_loss_observed = "loss_part_scorer_input_distribution_guard" in final
    guard_dual_observed = f"dual_ascent_metric__{GUARD_TETHER}" in final
    return {
        "schema": "compact_score_aware_training_telemetry_contract.v1",
        "family": "snerv",
        "source": SCHEMA,
        "telemetry_exists": True,
        "row_count": len(metrics_by_step),
        "passed": not blockers,
        "blockers": list(blockers),
        "expected_segnet_dual": True,
        "expected_posenet_dual": True,
        "expected_segnet_dual_lambda": True,
        "expected_posenet_dual_lambda": True,
        "expected_scorer_input_guard_metric": True,
        "segnet_loss_metric_observed": "loss_part_distill" in final,
        "posenet_loss_metric_observed": "loss_part_pose_score_term" in final,
        "posenet_score_term_metric_observed": "loss_part_pose_score_term" in final,
        "segnet_dual_metric_observed": f"dual_ascent_metric__{SEG_TETHER}" in final,
        "posenet_dual_metric_observed": f"dual_ascent_metric__{POSE_TETHER}" in final,
        "segnet_dual_lambda_active_observed": seg_lambda > 0.0,
        "posenet_dual_lambda_active_observed": pose_lambda > 0.0,
        "scorer_input_guard_metric_observed": guard_loss_observed,
        "scorer_input_guard_dual_metric_observed": guard_dual_observed,
        "scorer_input_guard_dual_lambda_active_observed": guard_lambda > 0.0,
        "authority": "macos_mlx_research_signal_false_authority",
        **FALSE_AUTHORITY,
    }


def _required_control_contract(
    telemetry_contract: Mapping[str, Any],
) -> dict[str, Any]:
    guard_passed = bool(
        telemetry_contract.get("passed") is True
        and telemetry_contract.get("expected_scorer_input_guard_metric") is True
        and telemetry_contract.get("scorer_input_guard_metric_observed") is True
        and telemetry_contract.get("scorer_input_guard_dual_metric_observed") is True
    )
    blockers: list[str] = []
    if not guard_passed:
        blockers.append("snerv_scorer_input_distribution_guard_smoke_not_passed")
    return {
        "schema": "snerv_score_aware_long_training_required_control_contract.v1",
        "source": SCHEMA,
        "passed": guard_passed,
        "controls": {
            "scorer_input_distribution_guard": {
                "control_id": "scorer_input_distribution_guard",
                "required": True,
                "bound": True,
                "telemetry_observed": bool(
                    telemetry_contract.get("scorer_input_guard_metric_observed")
                    is True
                    and telemetry_contract.get(
                        "scorer_input_guard_dual_metric_observed"
                    )
                    is True
                ),
                "passed": guard_passed,
                "blockers": list(blockers),
            }
        },
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }


def _make_minimal_pr95_score_bundle() -> object:
    import mlx.core as mx
    import mlx.nn as nn
    import numpy as np

    from tac.substrates._shared.mlx_score_aware.bundle import RendererBundle
    from tac.substrates.hinton_distilled_scorer_surrogate import (
        RealPoseNetTeacherCache,
        RealSegNetTeacherLogitsCache,
        build_learnable_pose_student_head,
        build_learnable_student_head,
    )

    class TinyRenderer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder_weight = mx.zeros((4, 4))
            self.decoder_bias = mx.zeros((4,))

        def reconstruct_pair(self, indices: Any) -> tuple[Any, Any]:
            bs = int(indices.shape[0])
            scale = mx.sum(self.decoder_weight) + mx.sum(self.decoder_bias)
            base = mx.ones((bs, 3, 4, 4)) * 0.5 * (scale * 0.0 + 1.0)
            mod = mx.broadcast_to(
                mx.reshape(self.decoder_weight[:1, :1] * 0.01, (1, 1, 1, 1)),
                (bs, 3, 4, 4),
            )
            return base + mod, base + mod * 2.0

    num_pairs = 8
    num_classes = 5
    base_labels = np.asarray(
        [
            [[0, 1], [2, 3]],
            [[4, 3], [2, 1]],
            [[1, 2], [3, 4]],
            [[0, 2], [4, 1]],
            [[3, 1], [0, 2]],
            [[2, 4], [1, 0]],
            [[4, 0], [3, 2]],
            [[1, 3], [2, 4]],
        ],
        dtype=np.int32,
    )
    labels = np.tile(base_labels, (1, 2, 2))
    logits = np.full((num_pairs, 4, 4, num_classes), -1.5, dtype=np.float32)
    for pair_index in range(num_pairs):
        for row in range(2):
            for col in range(2):
                logits[pair_index, row, col, labels[pair_index, row, col]] = 2.5
    pose_np = np.stack(
        [
            np.linspace(0.1 * i, 0.1 * i + 0.5, 6, dtype=np.float32)
            for i in range(num_pairs)
        ],
        axis=0,
    )
    seg_head = build_learnable_student_head(
        num_classes=num_classes,
        in_channels=3,
        seed=23,
        init_scale=0.2,
    )
    seg_head.weight = mx.zeros((3, num_classes))
    seg_head.bias = mx.array([4.0, 1.0, 0.0, -1.0, -2.0], dtype=mx.float32)
    return RendererBundle(
        model=TinyRenderer(),
        target_rgb_0=mx.zeros((num_pairs, 4, 4, 3)),
        target_rgb_1=mx.zeros((num_pairs, 4, 4, 3)),
        num_pairs=num_pairs,
        forward_convention="reconstruct_pair_nchw01",
        distillation_weight=1.0,
        scorer_teacher=RealSegNetTeacherLogitsCache(
            teacher_logits_thwk=mx.array(logits),
            frame_count=num_pairs,
            height=2,
            width=2,
            num_classes=num_classes,
        ),
        learnable_student_head=seg_head,
        pose_distillation_weight=1.0,
        pose_scorer_teacher=RealPoseNetTeacherCache(
            teacher_pose_np=mx.array(pose_np),
            num_pairs=num_pairs,
            pose_dims=6,
            per_dim_scale=mx.ones((6,)),
        ),
        learnable_pose_student_head=build_learnable_pose_student_head(
            pose_dims=6,
            pool_grid=1,
            input_channels=3,
            seed=29,
            init_scale=0.1,
        ),
        scorer_input_distribution_guard_weight=1.0,
    )


def _ordered_unique(values: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


if __name__ == "__main__":
    raise SystemExit(main())
