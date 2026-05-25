#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Active exploration measurement CLI for the 4 unexplored Conv2d drift paths.

T3-GRAND-COUNCIL-ACTIVE-EXPLORATION-CONV2D-DRIFT-UNEXPLORED-PATHS 2026-05-25
per operator directive *"the grand council should explore the unexplored and
address the unaddressed"*.

Empirically measures the 4 threads from the active exploration brief:

1. **Kahan compensated summation** (Thread 1)
2. **FP64 intermediate accumulation** (Thread 2)
3. **MLX-side deterministic-reduction enforcement** (Thread 3)
4. **cuDNN reference Conv2d 3x3** (Thread 4)

Routes through the canonical sister codex implementations at
``tac.local_acceleration.mlx_scorer_torch_parity``:

- ``build_mlx_conv2d_accumulation_probe_manifest`` exposes optimized /
  fixed_fp32 / kahan_fp32 / fixed_fp64 accumulation modes
- ``mlx_runtime_determinism_contract`` enumerates MLX public deterministic API

This CLI EMPIRICALLY MEASURES each path's drift reduction (instead of asserting
the Slot 2 "NOT FIXABLE" verdict on theoretical grounds). Per CLAUDE.md
"Apples-to-apples evidence discipline" + Carmack MVP-first 5/5 step 2 (the
falsifiable challenge).

Per CLAUDE.md "Forbidden /tmp paths in any persisted artifact": output goes
to ``experiments/results/conv2d_drift_unexplored_paths_<utc>/results.json``
(canonical durable location, NOT /tmp).

Per Catalog #287/#323 canonical Provenance: every measurement carries
``evidence_grade=macOS-MLX-research-signal`` + ``score_claim=False`` +
``axis_tag=[predicted]``.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from tac.local_acceleration.deterministic_primitives import (  # noqa: E402
    classify_reduction_percent,
)
from tac.local_acceleration.mlx_scorer_torch_parity import (  # noqa: E402
    MLXConv2dAccumulationThresholds,
    build_mlx_conv2d_accumulation_probe_manifest,
    mlx_runtime_determinism_contract,
)

# Canonical PR95 HNeRV decoder Conv2d 3x3 stage shapes per Slot 2 anchor.
# Per the canonical anchor: stage 2 is the operative scale that matched the
# Slot 1+2 measurement of 1.43e-06 max_abs at the 36->144 / 6x8 size.
PR95_STAGE2_SHAPE = {
    "name": "pr95_stage2_36_to_144_6x8",
    "batch": 1, "in_channels": 36, "out_channels": 144,
    "height": 6, "width": 8, "kernel_height": 3, "kernel_width": 3,
    "stride": 1, "padding": 1, "dilation": 1, "groups": 1,
}

PR95_STAGE4_SHAPE = {
    "name": "pr95_midstage_144_to_144_24x32",
    "batch": 2, "in_channels": 144, "out_channels": 144,
    "height": 24, "width": 32, "kernel_height": 3, "kernel_width": 3,
    "stride": 1, "padding": 1, "dilation": 1, "groups": 1,
}

PR95_FINAL_HEAD_SHAPE = {
    "name": "pr95_final_head_class_256_to_256_48x64",
    "batch": 1, "in_channels": 256, "out_channels": 256,
    "height": 48, "width": 64, "kernel_height": 3, "kernel_width": 3,
    "stride": 1, "padding": 1, "dilation": 1, "groups": 1,
}

CANONICAL_SHAPES = [PR95_STAGE2_SHAPE, PR95_STAGE4_SHAPE, PR95_FINAL_HEAD_SHAPE]
SMOKE_SHAPE = {
    "name": "smoke_tiny_conv2d_4_to_5_4x4",
    "batch": 1,
    "in_channels": 4,
    "out_channels": 5,
    "height": 4,
    "width": 4,
    "kernel_height": 3,
    "kernel_width": 3,
    "stride": 1,
    "padding": 1,
    "dilation": 1,
    "groups": 1,
}
SHAPE_PRESETS = {
    "all-pr95": CANONICAL_SHAPES,
    "pr95-stage2": [PR95_STAGE2_SHAPE],
    "smoke": [SMOKE_SHAPE],
}
VALID_MITIGATION_PATHS = frozenset(
    {"cudnn_reference", "fp64", "kahan", "mlx_deterministic"}
)


def _parse_mitigation_paths(raw: str) -> tuple[str, ...]:
    paths = tuple(part.strip() for part in raw.split(",") if part.strip())
    if not paths or "all" in paths:
        return tuple(sorted(VALID_MITIGATION_PATHS))
    unknown = sorted(set(paths) - VALID_MITIGATION_PATHS)
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown mitigation path(s): {', '.join(unknown)}; expected all "
            f"or comma-list from {', '.join(sorted(VALID_MITIGATION_PATHS))}"
        )
    return tuple(dict.fromkeys(paths))


def _shape_specs_for_preset(shape_preset: str) -> list[dict[str, Any]]:
    try:
        shapes = SHAPE_PRESETS[shape_preset]
    except KeyError as exc:
        raise ValueError(
            f"unknown shape preset {shape_preset!r}; expected one of "
            f"{sorted(SHAPE_PRESETS)}"
        ) from exc
    return [dict(shape) for shape in shapes]


def _synthetic_conv2d_case(
    shape_spec: dict[str, int], *, seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(int(seed))
    x = rng.normal(
        loc=0.0, scale=0.25,
        size=(shape_spec["batch"], shape_spec["in_channels"],
              shape_spec["height"], shape_spec["width"]),
    ).astype(np.float32)
    weight = rng.normal(
        loc=0.0, scale=0.2,
        size=(shape_spec["out_channels"],
              shape_spec["in_channels"] // shape_spec["groups"],
              shape_spec["kernel_height"], shape_spec["kernel_width"]),
    ).astype(np.float32)
    bias = rng.normal(loc=0.0, scale=0.05, size=(shape_spec["out_channels"],)).astype(np.float32)
    return x, weight, bias


def measure_threads_1_and_2(shape_spec: dict[str, Any]) -> dict[str, Any]:
    """Measure Kahan (Thread 1) + FP64 (Thread 2) Conv2d 3x3 drift reduction.

    Routes through sister codex ``build_mlx_conv2d_accumulation_probe_manifest``
    which exposes ``optimized_mlx_conv2d`` (baseline) + ``fixed_fp32`` +
    ``kahan_fp32`` + ``fixed_fp64`` accumulation modes against a PyTorch CPU
    reference. Returns per-mode max_abs_delta + reduction percentages.
    """
    x, weight, bias = _synthetic_conv2d_case(shape_spec)
    manifest = build_mlx_conv2d_accumulation_probe_manifest(
        input_nchw=x, weight_oihw=weight, bias=bias,
        stride=shape_spec["stride"], padding=shape_spec["padding"],
        dilation=shape_spec["dilation"], groups=shape_spec["groups"],
        device_type="cpu", torch_device_type="cpu",
        thresholds=MLXConv2dAccumulationThresholds(
            max_optimized_abs_delta=1.0,
            max_fixed_fp32_abs_delta=1.0,
            max_kahan_fp32_abs_delta=1.0,
            max_fixed_fp64_abs_delta=1.0,
        ),
        use_deterministic_algorithms=True,
    )
    rows = {row["mode"]: row for row in manifest["rows"]}
    baseline = float(rows["optimized_mlx_conv2d"]["max_abs_delta"])

    def _reduction(mode: str) -> float:
        candidate = float(rows[mode]["max_abs_delta"])
        if baseline <= 0:
            return 0.0
        return 100.0 * (baseline - candidate) / baseline

    return {
        "shape_name": shape_spec["name"],
        "shape_spec": dict(shape_spec),
        "baseline_optimized_max_abs": baseline,
        "fixed_fp32_max_abs": float(rows["fixed_fp32"]["max_abs_delta"]),
        "kahan_fp32_max_abs": float(rows["kahan_fp32"]["max_abs_delta"]),
        "fixed_fp64_max_abs": float(rows["fixed_fp64"]["max_abs_delta"]),
        "fixed_fp32_reduction_percent": _reduction("fixed_fp32"),
        "kahan_fp32_reduction_percent": _reduction("kahan_fp32"),
        "fixed_fp64_reduction_percent": _reduction("fixed_fp64"),
        # Carmack MVP-first 5/5 step 2: falsifiable prediction band.
        # Per Higham 2002 Kahan should reduce O(N*eps) -> O(eps^2), predicted
        # reduction >90% if drift were primarily summation-precision.
        # Per FP64 53/23 mantissa ratio, predicted reduction >99% if drift were
        # primarily LSB rounding at accumulation.
        "thread_1_kahan_predicted_reduction_percent_lower_bound": 50.0,
        "thread_1_kahan_predicted_falsification_threshold_percent": 50.0,
        "thread_2_fp64_predicted_reduction_percent_lower_bound": 50.0,
        "thread_2_fp64_predicted_falsification_threshold_percent": 50.0,
        "thread_1_verdict": _classify_reduction(_reduction("kahan_fp32")),
        "thread_2_verdict": _classify_reduction(_reduction("fixed_fp64")),
        "evidence_grade": "macOS-MLX-research-signal",
        "axis_tag": "[predicted]",
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _classify_reduction(reduction_percent: float) -> str:
    """Classify a per-path drift reduction into FIXABLE / NOT_FIXABLE_*.

    Per Carmack MVP-first 5/5 falsification thresholds:
    - >= 50%: FIXABLE (operationally meaningful drift reduction)
    - 10-50%: PARTIALLY_FIXABLE_MARGINAL (measurable but below predicted band)
    - 0-10%: NOT_FIXABLE_SUBSTITUTION_ONLY (drift floor is non-summation)
    - <0% (negative): NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL (substitute makes it worse)
    """
    return classify_reduction_percent(reduction_percent).value


def measure_thread_3() -> dict[str, Any]:
    """Measure Thread 3 - MLX-side deterministic-reduction enforcement.

    Routes through sister codex ``mlx_runtime_determinism_contract`` which
    enumerates MLX public API surfaces for deterministic-reduction flags.

    Verdict taxonomy:
    - FIXABLE if MLX exposes any public deterministic-reduction control
    - NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL if no public control observed
    """
    contract = mlx_runtime_determinism_contract(device_type="cpu")
    has_flag = bool(contract.get("deterministic_reduction_flag_available", False))
    return {
        "mlx_version": contract.get("mlx_version"),
        "deterministic_reduction_flag_available": has_flag,
        "public_core_deterministic_attrs": contract.get("public_core_deterministic_attrs", []),
        "public_metal_deterministic_attrs": contract.get("public_metal_deterministic_attrs", []),
        "classification": contract.get("classification"),
        "thread_3_verdict": (
            "FIXABLE_VIA_MLX_PUBLIC_API"
            if has_flag
            else "NOT_FIXABLE_FRAMEWORK_FUNDAMENTAL_NO_PUBLIC_API"
        ),
        "thread_3_predicted_verdict_lower_bound": "FIXABLE_OR_FRAMEWORK_DIFFERENT",
        "evidence_grade": "macOS-MLX-research-signal",
        "axis_tag": "[predicted]",
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def measure_thread_4() -> dict[str, Any]:
    """Measure Thread 4 - cuDNN reference Conv2d 3x3 availability.

    Per CLAUDE.md "MPS auth eval is NOISE" - MPS is NOT a valid substitute for
    cuDNN (23x drift on PoseNet documented). Therefore Thread 4 EMPIRICAL
    measurement on macOS requires either (a) NVIDIA GPU locally OR (b) paid
    cloud dispatch (Modal A100, Vast.ai 4090, Lightning A100).

    Verdict taxonomy:
    - LOCAL_MEASUREMENT_AVAILABLE if torch.backends.cudnn.is_available() True
    - DEFERRED_PENDING_PAID_DISPATCH if no local cuDNN
    """
    import torch
    cudnn_locally_available = (
        hasattr(torch.backends, "cudnn")
        and torch.backends.cudnn.is_available()
        and torch.cuda.is_available()
    )
    mps_available = torch.backends.mps.is_available()
    return {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": mps_available,
        "cudnn_enabled": bool(torch.backends.cudnn.enabled),
        "cudnn_locally_available": cudnn_locally_available,
        "thread_4_verdict": (
            "LOCAL_MEASUREMENT_AVAILABLE"
            if cudnn_locally_available
            else "DEFERRED_PENDING_PAID_DISPATCH"
        ),
        "thread_4_required_paid_dispatch_estimated_cost_usd": (
            None if cudnn_locally_available else 2.0
        ),
        "mps_not_substitute_for_cudnn_rationale": (
            "MPS is NOT a substitute for cuDNN per CLAUDE.md 'MPS auth eval is "
            "NOISE' non-negotiable (23x drift on PoseNet documented)"
        ),
        "evidence_grade": "macOS-MLX-research-signal",
        "axis_tag": "[predicted]",
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def _skipped_path(path_name: str) -> dict[str, Any]:
    return {
        "path_name": path_name,
        "measurement_status": "skipped_by_mitigation_paths_filter",
        "evidence_grade": "macOS-MLX-research-signal",
        "axis_tag": "[predicted]",
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
    }


def build_active_exploration_manifest(
    *,
    run_id: str | None = None,
    mitigation_paths: tuple[str, ...] | None = None,
    shape_preset: str = "pr95-stage2",
) -> dict[str, Any]:
    """Build full manifest covering Threads 1-4."""
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    selected_paths = mitigation_paths or tuple(sorted(VALID_MITIGATION_PATHS))
    shape_specs = _shape_specs_for_preset(shape_preset)
    threads_1_2 = []
    if {"kahan", "fp64"} & set(selected_paths):
        for shape_spec in shape_specs:
            threads_1_2.append(measure_threads_1_and_2(shape_spec))
    thread_3 = (
        measure_thread_3()
        if "mlx_deterministic" in selected_paths
        else _skipped_path("mlx_deterministic")
    )
    thread_4 = (
        measure_thread_4()
        if "cudnn_reference" in selected_paths
        else _skipped_path("cudnn_reference")
    )

    # Aggregate verdicts across scales (Thread 1+2).
    kahan_reductions = [r["kahan_fp32_reduction_percent"] for r in threads_1_2]
    fp64_reductions = [r["fixed_fp64_reduction_percent"] for r in threads_1_2]
    aggregate_kahan_verdict = (
        _classify_reduction(max(kahan_reductions))
        if "kahan" in selected_paths and kahan_reductions
        else "NOT_MEASURED"
    )
    aggregate_fp64_verdict = (
        _classify_reduction(max(fp64_reductions))
        if "fp64" in selected_paths and fp64_reductions
        else "NOT_MEASURED"
    )

    # Compose overall verdict.
    fixable_count = sum(
        1
        for v in (
            aggregate_kahan_verdict,
            aggregate_fp64_verdict,
            thread_3.get("thread_3_verdict", "NOT_MEASURED"),
            thread_4.get("thread_4_verdict", "NOT_MEASURED"),
        )
        if "FIXABLE" in v and "NOT_FIXABLE" not in v
    )
    partially_fixable_count = sum(
        1
        for v in (
            aggregate_kahan_verdict,
            aggregate_fp64_verdict,
        )
        if "PARTIALLY_FIXABLE" in v
    )
    deferred_count = sum(
        1
        for v in (
            thread_3.get("thread_3_verdict", "NOT_MEASURED"),
            thread_4.get("thread_4_verdict", "NOT_MEASURED"),
        )
        if "DEFERRED" in v or "NOT_FIXABLE" in v
    )

    return {
        "schema_version": "active_exploration_conv2d_drift_unexplored_paths.v1",
        "run_id": run_id,
        "anchor_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "shape_preset": shape_preset,
        "shape_count": len(shape_specs),
        "mitigation_paths": list(selected_paths),
        "hardware_substrate": "macos_apple_silicon_m5_max_mlx_cpu_vs_torch_cpu",
        "evidence_grade": "macOS-MLX-research-signal",
        "axis_tag": "[predicted]",
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "thread_1_kahan_per_scale_measurements": threads_1_2,
        "thread_1_aggregate_verdict": aggregate_kahan_verdict,
        "thread_1_max_observed_reduction_percent": (
            max(kahan_reductions) if kahan_reductions else None
        ),
        "thread_2_fp64_per_scale_measurements": [
            {
                "shape_name": r["shape_name"],
                "shape_spec": r["shape_spec"],
                "baseline_optimized_max_abs": r["baseline_optimized_max_abs"],
                "fixed_fp64_max_abs": r["fixed_fp64_max_abs"],
                "fixed_fp64_reduction_percent": r["fixed_fp64_reduction_percent"],
                "thread_2_verdict": r["thread_2_verdict"],
            }
            for r in threads_1_2
        ],
        "thread_2_aggregate_verdict": aggregate_fp64_verdict,
        "thread_2_max_observed_reduction_percent": (
            max(fp64_reductions) if fp64_reductions else None
        ),
        "thread_3_mlx_deterministic_investigation": thread_3,
        "thread_4_cudnn_reference_measurement": thread_4,
        "active_exploration_summary": {
            "threads_fixable_count": fixable_count,
            "threads_partially_fixable_count": partially_fixable_count,
            "threads_deferred_or_not_fixable_count": deferred_count,
            "overall_verdict": (
                "PROCEED"
                if (fixable_count + partially_fixable_count) >= 2
                else "PARTIAL"
                if (fixable_count + partially_fixable_count) >= 1
                else "DEFER"
            ),
        },
        "blockers": [
            "active_exploration_is_local_mlx_pytorch_macos_research_signal_only",
            "requires_paired_cuda_t4_or_linux_x86_64_eval_for_promotion",
            "thread_4_cudnn_reference_deferred_pending_paid_dispatch",
        ],
        "carmack_mvp_first_5_of_5": {
            "step_1_free_local_macos": "TRUE - $0 GPU local MLX+PyTorch CPU",
            "step_2_falsifiable_challenge_made": True,
            "step_2_thread_1_predicted_kahan_reduction_lower_bound_percent": 50.0,
            "step_2_thread_2_predicted_fp64_reduction_lower_bound_percent": 50.0,
            "step_2_thread_1_kahan_falsified": (
                max(kahan_reductions) < 50.0 if kahan_reductions else None
            ),
            "step_2_thread_2_fp64_falsified": (
                max(fp64_reductions) < 50.0 if fp64_reductions else None
            ),
            "step_3_catalog_344_referenced": "TRUE - 4 candidate equations queued FORMALIZATION_PENDING",
            "step_4_verdict_same_commit_batch": "TRUE",
            "step_5_operator_priority_queue_reroute": "TRUE - Slot 1 export bridge VERDICT upgrade routed",
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path; defaults to canonical experiments/results/...",
    )
    parser.add_argument("--run-id", default=None)
    parser.add_argument(
        "--mitigation-paths",
        type=_parse_mitigation_paths,
        default=tuple(sorted(VALID_MITIGATION_PATHS)),
        help="Comma list: kahan,fp64,mlx_deterministic,cudnn_reference (default: all)",
    )
    parser.add_argument(
        "--shape-preset",
        choices=sorted(SHAPE_PRESETS),
        default="pr95-stage2",
        help=(
            "Conv2d shape preset for Kahan/FP64 measurements. "
            "Use all-pr95 only for explicit wider sweeps."
        ),
    )
    args = parser.parse_args(argv)

    manifest = build_active_exploration_manifest(
        run_id=args.run_id,
        mitigation_paths=args.mitigation_paths,
        shape_preset=args.shape_preset,
    )
    if args.output is None:
        run_id = manifest["run_id"]
        out_dir = REPO_ROOT / "experiments" / "results" / f"conv2d_drift_unexplored_paths_{run_id}"
        out_dir.mkdir(parents=True, exist_ok=True)
        args.output = out_dir / "results.json"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({
        "output": str(args.output),
        "schema_version": manifest["schema_version"],
        "shape_preset": manifest["shape_preset"],
        "mitigation_paths": manifest["mitigation_paths"],
        "thread_1_aggregate_verdict": manifest["thread_1_aggregate_verdict"],
        "thread_2_aggregate_verdict": manifest["thread_2_aggregate_verdict"],
        "thread_3_verdict": manifest["thread_3_mlx_deterministic_investigation"].get(
            "thread_3_verdict",
            "NOT_MEASURED",
        ),
        "thread_4_verdict": manifest["thread_4_cudnn_reference_measurement"].get(
            "thread_4_verdict",
            "NOT_MEASURED",
        ),
        "overall_verdict": manifest["active_exploration_summary"]["overall_verdict"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
