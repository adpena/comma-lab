#!/usr/bin/env python3
"""Plan a CPU-only sub-0.17 HNeRV weight-cutting candidate.

This tool emits a construction plan, not a candidate archive. It focuses on
byte-cutting the HNeRV decoder weights with low-rank SVD on ``stem.weight`` and
early decoder ``Conv2d`` tensors, then applying continuous-K allocation and
analytical lossy coarsening in a safe order. It never loads scorers, launches
remote work, or creates a score claim.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

CONTEST_ARCHIVE_DENOMINATOR = 37_545_489
SCHEMA = "sub017_cpu_frontier_plan.v1"
TOOL = "tools/plan_sub017_cpu_frontier.py"

DEFAULT_TARGET_SCORE = 0.17
DEFAULT_ANCHOR_ARCHIVE_BYTES = 185_578
DEFAULT_ANCHOR_SCORE = 0.20898105277982337
DEFAULT_ANCHOR_AVG_POSE_DIST = 0.0000336
DEFAULT_ANCHOR_AVG_SEG_DIST = 0.00067082
DEFAULT_ANCHOR_SHA256 = "ec0890c2d2317dcad903ed37ffddb2794cd19c1df9effa057cb7f05af205e1ce"
DEFAULT_PROTECTED_NONWEIGHT_BYTES = 15_957


@dataclass(frozen=True)
class LayerProxy:
    name: str
    shape: tuple[int, ...]
    compressed_bytes: int
    role: str

    @property
    def numel(self) -> int:
        total = 1
        for dim in self.shape:
            total *= dim
        return total

    @property
    def matrix_shape(self) -> tuple[int, int] | None:
        if len(self.shape) == 2:
            return self.shape
        if len(self.shape) == 4:
            out_ch, in_ch, kh, kw = self.shape
            return out_ch, in_ch * kh * kw
        return None


LAYER_PROXIES: tuple[LayerProxy, ...] = (
    LayerProxy("stem.weight", (1728, 28), 31_872, "stem_projection"),
    LayerProxy("stem.bias", (1728,), 1_608, "bias_protected"),
    LayerProxy("blocks.0.weight", (144, 36, 3, 3), 32_544, "early_decoder_conv2d"),
    LayerProxy("blocks.0.bias", (144,), 148, "bias_protected"),
    LayerProxy("blocks.1.weight", (144, 36, 3, 3), 32_722, "early_decoder_conv2d"),
    LayerProxy("blocks.1.bias", (144,), 148, "bias_protected"),
    LayerProxy("blocks.2.weight", (108, 36, 3, 3), 25_643, "early_decoder_conv2d"),
    LayerProxy("blocks.2.bias", (108,), 112, "bias_protected"),
    LayerProxy("blocks.3.weight", (80, 27, 3, 3), 14_187, "mid_decoder_conv2d"),
    LayerProxy("blocks.3.bias", (80,), 84, "bias_protected"),
    LayerProxy("blocks.4.weight", (72, 20, 3, 3), 10_335, "mid_decoder_conv2d"),
    LayerProxy("blocks.4.bias", (72,), 76, "bias_protected"),
    LayerProxy("blocks.5.weight", (72, 18, 3, 3), 7_810, "mid_decoder_conv2d"),
    LayerProxy("blocks.5.bias", (72,), 76, "bias_protected"),
    LayerProxy("skips.2.weight", (27, 36, 1, 1), 783, "skip_projection_protected"),
    LayerProxy("skips.2.bias", (27,), 31, "bias_protected"),
    LayerProxy("skips.3.weight", (20, 27, 1, 1), 495, "skip_projection_protected"),
    LayerProxy("skips.3.bias", (20,), 24, "bias_protected"),
    LayerProxy("skips.4.weight", (18, 20, 1, 1), 345, "skip_projection_protected"),
    LayerProxy("skips.4.bias", (18,), 22, "bias_protected"),
    LayerProxy("refine.0.weight", (9, 18, 3, 3), 1_083, "late_refine_protected"),
    LayerProxy("refine.0.bias", (9,), 13, "bias_protected"),
    LayerProxy("refine.1.weight", (18, 9, 3, 3), 1_183, "late_refine_protected"),
    LayerProxy("refine.1.bias", (18,), 22, "bias_protected"),
    LayerProxy("rgb_0.weight", (3, 18, 3, 3), 415, "rgb_head_protected"),
    LayerProxy("rgb_0.bias", (3,), 7, "bias_protected"),
    LayerProxy("rgb_1.weight", (3, 18, 3, 3), 432, "rgb_head_protected"),
    LayerProxy("rgb_1.bias", (3,), 7, "bias_protected"),
)

LOW_RANK_OPTIONS: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "svd_stem_blocks01_conservative",
        "risk": "medium",
        "residual_fraction": 0.08,
        "continuous_k_extra_savings_bytes": 6_000,
        "ranks": {
            "stem.weight": 20,
            "blocks.0.weight": 48,
            "blocks.1.weight": 48,
        },
    },
    {
        "candidate_id": "svd_stem_blocks012_balanced",
        "risk": "medium_high",
        "residual_fraction": 0.06,
        "continuous_k_extra_savings_bytes": 8_500,
        "ranks": {
            "stem.weight": 16,
            "blocks.0.weight": 32,
            "blocks.1.weight": 32,
            "blocks.2.weight": 24,
        },
    },
    {
        "candidate_id": "svd_stem_blocks0123_aggressive",
        "risk": "high",
        "residual_fraction": 0.04,
        "continuous_k_extra_savings_bytes": 12_000,
        "ranks": {
            "stem.weight": 12,
            "blocks.0.weight": 24,
            "blocks.1.weight": 24,
            "blocks.2.weight": 18,
            "blocks.3.weight": 18,
        },
    },
)


def rate_score(archive_bytes: int) -> float:
    return 25.0 * float(archive_bytes) / float(CONTEST_ARCHIVE_DENOMINATOR)


def quality_score(avg_seg_dist: float, avg_pose_dist: float) -> float:
    return 100.0 * avg_seg_dist + math.sqrt(10.0 * avg_pose_dist)


def score_from_components(
    *, archive_bytes: int, avg_seg_dist: float, avg_pose_dist: float
) -> float:
    return quality_score(avg_seg_dist, avg_pose_dist) + rate_score(archive_bytes)


def max_archive_bytes_for_target(
    *, target_score: float, avg_seg_dist: float, avg_pose_dist: float
) -> int:
    quality = quality_score(avg_seg_dist, avg_pose_dist)
    rate_budget = target_score - quality
    if rate_budget <= 0:
        return -1
    return math.floor(rate_budget * CONTEST_ARCHIVE_DENOMINATOR / 25.0)


def _layer_by_name() -> dict[str, LayerProxy]:
    return {layer.name: layer for layer in LAYER_PROXIES}


def estimate_low_rank_layer(
    layer: LayerProxy,
    *,
    rank: int,
    residual_fraction: float,
    factor_metadata_bytes: int = 96,
) -> dict[str, Any]:
    matrix_shape = layer.matrix_shape
    if matrix_shape is None:
        raise ValueError(f"{layer.name} is not matrixizable")
    rows, cols = matrix_shape
    if rank <= 0 or rank > min(rows, cols):
        raise ValueError(f"rank {rank} invalid for {layer.name} matrix {matrix_shape}")

    factor_values = rank * (rows + cols)
    factorized_bytes = math.ceil(layer.compressed_bytes * factor_values / layer.numel)
    residual_bytes = math.ceil(layer.compressed_bytes * residual_fraction)
    estimated_bytes = factorized_bytes + residual_bytes + factor_metadata_bytes
    saved_bytes = layer.compressed_bytes - estimated_bytes
    return {
        "name": layer.name,
        "role": layer.role,
        "shape": list(layer.shape),
        "matrix_shape": [rows, cols],
        "rank": rank,
        "full_numel": layer.numel,
        "factor_values": factor_values,
        "current_compressed_bytes_proxy": layer.compressed_bytes,
        "estimated_low_rank_bytes": estimated_bytes,
        "estimated_saved_bytes": saved_bytes,
        "residual_fraction": residual_fraction,
        "construction": [
            "matrixize tensor",
            "CPU SVD on dequantized fp32 weights",
            "emit charged int8/fp16 U/V factor streams",
            "emit optional charged residual stream",
            "fork inflate runtime to multiply factors before Conv2d use",
        ],
    }


def evaluate_option(
    option: dict[str, Any],
    *,
    anchor_archive_bytes: int,
    target_max_archive_bytes: int,
    avg_seg_dist: float,
    avg_pose_dist: float,
) -> dict[str, Any]:
    layers = _layer_by_name()
    rows = [
        estimate_low_rank_layer(
            layers[name],
            rank=int(rank),
            residual_fraction=float(option["residual_fraction"]),
        )
        for name, rank in option["ranks"].items()
    ]
    svd_saved = sum(max(0, int(row["estimated_saved_bytes"])) for row in rows)
    continuous_k_savings = int(option["continuous_k_extra_savings_bytes"])
    runtime_overhead = 2_048 + 48 * len(rows)
    net_savings = max(0, svd_saved + continuous_k_savings - runtime_overhead)
    estimated_archive_bytes = max(0, anchor_archive_bytes - net_savings)
    projected_score_if_components_hold = score_from_components(
        archive_bytes=estimated_archive_bytes,
        avg_seg_dist=avg_seg_dist,
        avg_pose_dist=avg_pose_dist,
    )
    meets_byte_budget = (
        target_max_archive_bytes >= 0 and estimated_archive_bytes <= target_max_archive_bytes
    )
    return {
        "candidate_id": option["candidate_id"],
        "risk": option["risk"],
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "target_layers": rows,
        "runtime_overhead_bytes_proxy": runtime_overhead,
        "svd_saved_bytes_proxy": svd_saved,
        "continuous_k_extra_savings_bytes_proxy": continuous_k_savings,
        "net_saved_bytes_proxy": net_savings,
        "estimated_archive_bytes": estimated_archive_bytes,
        "target_max_archive_bytes": target_max_archive_bytes,
        "meets_sub017_byte_budget_if_components_hold": meets_byte_budget,
        "projected_score_if_components_hold": projected_score_if_components_hold,
        "dispatch_blockers": [
            "cpu_design_only_no_archive_built",
            "component_preservation_unproven",
            "factorized_inflate_runtime_not_built",
            "exact_cuda_auth_eval_required_before_score_claim",
        ],
    }


def build_plan(
    *,
    target_score: float = DEFAULT_TARGET_SCORE,
    anchor_archive_bytes: int = DEFAULT_ANCHOR_ARCHIVE_BYTES,
    anchor_avg_pose_dist: float = DEFAULT_ANCHOR_AVG_POSE_DIST,
    anchor_avg_seg_dist: float = DEFAULT_ANCHOR_AVG_SEG_DIST,
    protected_nonweight_bytes: int = DEFAULT_PROTECTED_NONWEIGHT_BYTES,
) -> dict[str, Any]:
    target_max_archive_bytes = max_archive_bytes_for_target(
        target_score=target_score,
        avg_seg_dist=anchor_avg_seg_dist,
        avg_pose_dist=anchor_avg_pose_dist,
    )
    anchor_quality = quality_score(anchor_avg_seg_dist, anchor_avg_pose_dist)
    weight_payload_target = target_max_archive_bytes - protected_nonweight_bytes
    current_score = score_from_components(
        archive_bytes=anchor_archive_bytes,
        avg_seg_dist=anchor_avg_seg_dist,
        avg_pose_dist=anchor_avg_pose_dist,
    )
    candidates = [
        evaluate_option(
            option,
            anchor_archive_bytes=anchor_archive_bytes,
            target_max_archive_bytes=target_max_archive_bytes,
            avg_seg_dist=anchor_avg_seg_dist,
            avg_pose_dist=anchor_avg_pose_dist,
        )
        for option in LOW_RANK_OPTIONS
    ]
    recommended = next(
        (row for row in candidates if row["meets_sub017_byte_budget_if_components_hold"]),
        candidates[-1],
    )
    return {
        "schema": SCHEMA,
        "tool": TOOL,
        "created_at_utc": datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "evidence_grade": "[CPU-design]",
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "dispatch_attempted": False,
        "target_score": target_score,
        "anchor": {
            "label": "PR103-on-PR106 AC repack active local HNeRV rate anchor",
            "archive_bytes": anchor_archive_bytes,
            "archive_sha256": DEFAULT_ANCHOR_SHA256,
            "avg_pose_dist": anchor_avg_pose_dist,
            "avg_seg_dist": anchor_avg_seg_dist,
            "score_recomputed_from_components": DEFAULT_ANCHOR_SCORE,
            "score_from_supplied_components": current_score,
            "score_basis": "contest_formula_components_for_byte_budget_only",
        },
        "byte_budget": {
            "anchor_quality_score_without_rate": anchor_quality,
            "target_max_archive_bytes_if_components_hold": target_max_archive_bytes,
            "required_archive_savings_bytes": anchor_archive_bytes - target_max_archive_bytes,
            "protected_nonweight_bytes": protected_nonweight_bytes,
            "target_weight_payload_bytes": weight_payload_target,
            "rate_only_assumption": "avg_pose_dist and avg_seg_dist held fixed for budget math only",
        },
        "safe_stacking_order": [
            {
                "step": 1,
                "name": "svd_low_rank_stem_and_early_conv2d",
                "reason": "remove large decoder weight mass before quantization noise is introduced",
                "must_prove": [
                    "CPU reconstruction rel_err per tensor",
                    "charged factor and residual bytes",
                    "runtime consumes factorized tensors",
                ],
            },
            {
                "step": 2,
                "name": "continuous_k_allocation",
                "reason": "choose per-stream K under a global distortion budget after SVD changes tensor spectra",
                "must_prove": [
                    "no dead K side-info bytes",
                    "ordered schema matches runtime decode order",
                    "guarded selected-K caps for scorer-risky layers",
                ],
            },
            {
                "step": 3,
                "name": "analytical_lossy_coarsening",
                "reason": "coarsen factor and residual streams only after rank selection",
                "must_prove": [
                    "aggregate fp32 smoke rel_err below guard",
                    "per-layer max rel_err below guard",
                    "lossy stream changes charged bytes",
                ],
            },
            {
                "step": 4,
                "name": "entropy_pack_and_noop_guards",
                "reason": "pack final charged streams and reject no-op or metadata-only wins",
                "must_prove": [
                    "old/new archive SHA boundary",
                    "single scored archive closure",
                    "strict compliance before any exact eval claim",
                ],
            },
        ],
        "candidate_construction_plans": candidates,
        "recommended_cpu_candidate_id": recommended["candidate_id"],
        "recommended_cpu_candidate": recommended,
        "global_dispatch_blockers": [
            "no_archive_built_by_this_tool",
            "cpu_projection_not_score_evidence",
            "factorized_hnerv_runtime_not_implemented",
            "exact_cuda_auth_eval_required_before_score_claim",
            "remote_dispatch_forbidden_for_this_task",
        ],
        "next_local_artifacts": [
            "layerwise SVD reconstruction manifest for stem.weight and blocks.0-2.weight",
            "continuous-K allocation manifest over factor/residual streams",
            "byte-closed forked inflate runtime consuming low-rank factors",
            "strict no-score compliance report",
        ],
    }


def render_markdown(plan: dict[str, Any]) -> str:
    lines = [
        "# Sub-0.17 CPU Frontier Plan",
        "",
        f"score_claim: `{str(plan['score_claim']).lower()}`",
        f"ready_for_exact_eval_dispatch: `{str(plan['ready_for_exact_eval_dispatch']).lower()}`",
        f"target_score: `{plan['target_score']}`",
        "",
        "## Byte Budget",
        "",
        f"- anchor archive bytes: `{plan['anchor']['archive_bytes']}`",
        f"- max archive bytes if components hold: `{plan['byte_budget']['target_max_archive_bytes_if_components_hold']}`",
        f"- required archive savings: `{plan['byte_budget']['required_archive_savings_bytes']}`",
        f"- target weight payload bytes: `{plan['byte_budget']['target_weight_payload_bytes']}`",
        "",
        "## Recommended Candidate",
        "",
        f"- id: `{plan['recommended_cpu_candidate_id']}`",
        f"- estimated archive bytes: `{plan['recommended_cpu_candidate']['estimated_archive_bytes']}`",
        f"- projected score if components hold: `{plan['recommended_cpu_candidate']['projected_score_if_components_hold']:.12f}`",
        f"- risk: `{plan['recommended_cpu_candidate']['risk']}`",
        "",
        "## Safe Stacking Order",
        "",
    ]
    for step in plan["safe_stacking_order"]:
        lines.append(f"{step['step']}. `{step['name']}` - {step['reason']}")
    lines.extend(["", "## Dispatch Blockers", ""])
    for blocker in plan["global_dispatch_blockers"]:
        lines.append(f"- `{blocker}`")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-score", type=float, default=DEFAULT_TARGET_SCORE)
    parser.add_argument("--anchor-archive-bytes", type=int, default=DEFAULT_ANCHOR_ARCHIVE_BYTES)
    parser.add_argument("--anchor-avg-pose-dist", type=float, default=DEFAULT_ANCHOR_AVG_POSE_DIST)
    parser.add_argument("--anchor-avg-seg-dist", type=float, default=DEFAULT_ANCHOR_AVG_SEG_DIST)
    parser.add_argument("--protected-nonweight-bytes", type=int, default=DEFAULT_PROTECTED_NONWEIGHT_BYTES)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--print-markdown", action="store_true")
    args = parser.parse_args(argv)

    plan = build_plan(
        target_score=args.target_score,
        anchor_archive_bytes=args.anchor_archive_bytes,
        anchor_avg_pose_dist=args.anchor_avg_pose_dist,
        anchor_avg_seg_dist=args.anchor_avg_seg_dist,
        protected_nonweight_bytes=args.protected_nonweight_bytes,
    )
    text = json.dumps(plan, indent=2, sort_keys=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text + "\n", encoding="utf-8")
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(render_markdown(plan), encoding="utf-8")
    print(render_markdown(plan) if args.print_markdown else text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
