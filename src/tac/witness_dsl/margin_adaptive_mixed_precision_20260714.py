# SPDX-License-Identifier: MIT
"""Typed, launch-inert policy for the real-n600 margin-adaptive probe.

The policy intentionally separates the native executable treatment (one
frame-independent precision cap per frozen Conv2d) from the per-pixel margin
waterfill diagnostic.  SegNet contains global squeeze/excite dependencies, so
the latter is a lower bound rather than a sparse-kernel speed claim.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_PROFILE_CAPS = (8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 27, 28, 29, 30, 31)


@dataclass(frozen=True)
class MarginAdaptiveMixedPrecisionPolicy:
    pair_start: int = 0
    pair_stop: int = 600
    design_stop: int = 264
    profile_caps: tuple[int, ...] = DEFAULT_PROFILE_CAPS
    n_processes: int = 10
    gt_cache: str = "experiments/results/mlx_fleet_gt_cache/gt_n600.npz"
    uniform_nogo_receipt: str = (
        "experiments/results/throughput_authority_ladder_20260714/"
        "fixedpoint_scorer_forward_n600_fresh_89b970ff60.json"
    )
    calibration_receipt: str = (
        "experiments/results/throughput_authority_ladder_20260714/"
        "dynamic_fixedpoint_scorer_forward_int64_ceiling_corrected_n600.json"
    )
    integer_precursor_receipt: str = (
        "experiments/results/throughput_authority_ladder_20260714/"
        "weight_l1_class_pair_tie_snap_scorer_forward_n600.json"
    )
    output: str = (
        "experiments/results/margin_adaptive_mixed_precision_20260714/"
        "margin_adaptive_mixed_precision_n600.json"
    )
    accumulation: str = "exact_signed_int64"
    weight_scale_granularity: str = "per_output_channel"
    activation_scale_granularity: str = "per_layer_dynamic_exact_absmax"
    integer_operand_storage_buckets: tuple[int, ...] = (8, 16, 32)
    native_allocation_granularity: str = "per_layer_frame_independent"
    spatial_waterfill_native_execution: bool = False
    checkpoint_every_pairs: int = 1
    resume: bool = True
    operator_go_required: bool = True
    research_only: bool = True
    score_claim: bool = False
    pointer_moved: bool = False

    def __post_init__(self) -> None:
        if (self.pair_start, self.pair_stop, self.design_stop) != (0, 600, 264):
            raise ValueError("decisive policy is sealed to design 0..263 and validation 264..599")
        if not self.profile_caps or tuple(sorted(set(self.profile_caps))) != self.profile_caps:
            raise ValueError("profile_caps must be non-empty, sorted, and unique")
        if self.profile_caps[0] < 8 or self.profile_caps[-1] > 31:
            raise ValueError("exact-int64 profile caps must remain within 8..31")
        if self.n_processes != 10:
            raise ValueError("cross-process certificate is sealed to ten fresh processes")
        if self.checkpoint_every_pairs != 1 or not self.resume:
            raise ValueError("every-pair atomic checkpointing and resume are mandatory")
        if self.accumulation != "exact_signed_int64":
            raise ValueError("lossy or floating accumulation is forbidden")
        if self.weight_scale_granularity != "per_output_channel":
            raise ValueError("per-output-channel weight scales are the minimum admitted granularity")
        if self.integer_operand_storage_buckets != (8, 16, 32):
            raise ValueError("native exact operand storage is sealed to signed int8/int16/int32")
        if self.spatial_waterfill_native_execution:
            raise ValueError("spatial waterfill is diagnostic until global dependency closure is solved")
        if not self.operator_go_required or not self.research_only:
            raise ValueError("Metal treatment remains operator-GO and research-only")
        if self.score_claim or self.pointer_moved:
            raise ValueError("a forward timing probe cannot claim score or move a pointer")

    def to_dict(self) -> dict[str, object]:
        return {"schema": "margin_adaptive_mixed_precision_policy.v1", **asdict(self)}


def compile_margin_adaptive_probe_argv(
    policy: MarginAdaptiveMixedPrecisionPolicy | None = None,
) -> tuple[str, ...]:
    """Compile the exact host argv without launching Metal or a subprocess."""

    cfg = policy or MarginAdaptiveMixedPrecisionPolicy()
    return (
        ".venv/bin/python",
        "tools/probe_margin_adaptive_mixed_precision_n600.py",
        "--pair-start",
        str(cfg.pair_start),
        "--pair-stop",
        str(cfg.pair_stop),
        "--profile-caps",
        ",".join(str(bits) for bits in cfg.profile_caps),
        "--n-processes",
        str(cfg.n_processes),
        "--gt-cache",
        cfg.gt_cache,
        "--uniform-nogo-receipt",
        cfg.uniform_nogo_receipt,
        "--calibration-receipt",
        cfg.calibration_receipt,
        "--integer-precursor-receipt",
        cfg.integer_precursor_receipt,
        "--resume",
        "--output",
        cfg.output,
    )


def validate_bound_paths(policy: MarginAdaptiveMixedPrecisionPolicy, *, repo: Path) -> None:
    """Fail closed when any immutable input receipt/cache is absent."""

    for value in (
        policy.gt_cache,
        policy.uniform_nogo_receipt,
        policy.calibration_receipt,
        policy.integer_precursor_receipt,
    ):
        path = repo / value
        if not path.is_file():
            raise FileNotFoundError(path)


__all__ = [
    "DEFAULT_PROFILE_CAPS",
    "MarginAdaptiveMixedPrecisionPolicy",
    "compile_margin_adaptive_probe_argv",
    "validate_bound_paths",
]
