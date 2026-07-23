# SPDX-License-Identifier: MIT
"""Paired painted-vs-GT SegNet amplitude telemetry.

The hook layer performs one concatenated frozen forward.  The first half of
the batch is the painted receiver trajectory and the second half is the
matched official-video trajectory.  Hooks reduce activations immediately;
no full activation tensor survives the forward.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any, Final

import torch
import torch.nn.functional as F

_ENCODER_BLOCK: Final = re.compile(r"^encoder\.model\.blocks\.\d+\.\d+$")
_DECODER_BLOCK: Final = re.compile(r"^decoder\.blocks\.\d+$")
_TRAJECTORY_EXACT: Final = frozenset(
    {
        "encoder.model.conv_stem",
        "segmentation_head",
    }
)


class SegNetAmplitudeTelemetryError(RuntimeError):
    """Raised when paired amplitude custody or aggregation does not close."""


def _tensor_from_output(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return next((item for item in value if isinstance(item, torch.Tensor)), None)
    return None


def _boundary_band(labels: torch.Tensor) -> torch.Tensor:
    if labels.ndim != 3:
        raise SegNetAmplitudeTelemetryError("reference labels must be BHW")
    boundary = torch.zeros_like(labels, dtype=torch.bool)
    horizontal = labels[:, :, 1:] != labels[:, :, :-1]
    vertical = labels[:, 1:, :] != labels[:, :-1, :]
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal
    boundary[:, 1:, :] |= vertical
    boundary[:, :-1, :] |= vertical
    return F.max_pool2d(
        boundary[:, None].to(dtype=torch.float32),
        kernel_size=3,
        stride=1,
        padding=1,
    )[:, 0].to(dtype=torch.bool)


def _split_nchw(value: torch.Tensor, split_count: int) -> tuple[torch.Tensor, torch.Tensor]:
    if value.ndim != 4 or value.shape[0] != 2 * split_count:
        raise SegNetAmplitudeTelemetryError(
            f"paired activation must be (2B,C,H,W), got {tuple(value.shape)}"
        )
    detached = value.detach()
    return detached[:split_count], detached[split_count:]


def _channel_moments(value: torch.Tensor) -> tuple[int, list[float], list[float]]:
    count = int(value.shape[0] * value.shape[2] * value.shape[3])
    return (
        count,
        value.sum(dim=(0, 2, 3), dtype=torch.float64).cpu().tolist(),
        value.square().sum(dim=(0, 2, 3), dtype=torch.float64).cpu().tolist(),
    )


def measure_paired_segnet_amplitude(
    model: torch.nn.Module,
    paired_model_input: torch.Tensor,
    *,
    split_count: int,
    reference_labels: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    """Run one frozen concatenated forward and reduce BN, SE, and trajectory statistics."""

    if model.training:
        raise SegNetAmplitudeTelemetryError("paired amplitude telemetry requires eval mode")
    if paired_model_input.ndim != 4 or paired_model_input.shape[0] != 2 * split_count:
        raise SegNetAmplitudeTelemetryError("paired model input must contain painted then GT batches")
    labels = reference_labels.detach().to(dtype=torch.long, device=paired_model_input.device)
    if labels.shape[0] != split_count:
        raise SegNetAmplitudeTelemetryError("reference-label batch differs from split_count")
    base_boundary = _boundary_band(labels)
    boundary_cache: dict[tuple[int, int], torch.Tensor] = {}
    bn_rows: dict[str, dict[str, Any]] = {}
    se_rows: dict[str, dict[str, Any]] = {}
    trajectory_rows: dict[str, dict[str, Any]] = {}
    hooks: list[Any] = []
    execution_order = 0

    def next_order() -> int:
        nonlocal execution_order
        execution_order += 1
        return execution_order

    def bn_pre_hook(name: str, module: torch.nn.Module, args: tuple[Any, ...]) -> None:
        value = _tensor_from_output(args)
        if value is None:
            raise SegNetAmplitudeTelemetryError(f"BN input is absent: {name}")
        painted, ground_truth = _split_nchw(value, split_count)
        count, painted_sum, painted_sumsq = _channel_moments(painted)
        gt_count, gt_sum, gt_sumsq = _channel_moments(ground_truth)
        if count != gt_count:
            raise SegNetAmplitudeTelemetryError(f"BN paired count differs: {name}")
        running_mean = getattr(module, "running_mean", None)
        running_var = getattr(module, "running_var", None)
        if running_mean is None or running_var is None:
            raise SegNetAmplitudeTelemetryError(f"BN running statistics are absent: {name}")
        bn_rows[name] = {
            "order": next_order(),
            "channel_count": int(painted.shape[1]),
            "sample_count_per_channel": count,
            "painted_sum": painted_sum,
            "painted_sumsq": painted_sumsq,
            "gt_sum": gt_sum,
            "gt_sumsq": gt_sumsq,
            "running_mean": running_mean.detach().to(dtype=torch.float64).cpu().tolist(),
            "running_variance": running_var.detach().to(dtype=torch.float64).cpu().tolist(),
            "eps": float(getattr(module, "eps", 1e-5)),
        }

    def se_hook(name: str, _module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        value = _tensor_from_output(output)
        if value is None:
            raise SegNetAmplitudeTelemetryError(f"SE gate output is absent: {name}")
        painted, ground_truth = _split_nchw(value, split_count)
        count, painted_sum, painted_sumsq = _channel_moments(painted)
        gt_count, gt_sum, gt_sumsq = _channel_moments(ground_truth)
        if count != gt_count:
            raise SegNetAmplitudeTelemetryError(f"SE paired count differs: {name}")
        se_rows[name] = {
            "order": next_order(),
            "channel_count": int(painted.shape[1]),
            "sample_count_per_channel": count,
            "painted_sum": painted_sum,
            "painted_sumsq": painted_sumsq,
            "gt_sum": gt_sum,
            "gt_sumsq": gt_sumsq,
        }

    def trajectory_hook(
        name: str,
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        output: Any,
    ) -> None:
        value = _tensor_from_output(output)
        if value is None or value.ndim != 4:
            raise SegNetAmplitudeTelemetryError(f"trajectory activation is absent: {name}")
        painted, ground_truth = _split_nchw(value, split_count)
        difference = painted - ground_truth
        spatial_mean = difference.mean(dim=(2, 3), keepdim=True)
        total_sse = float(difference.square().sum(dtype=torch.float64))
        uniform_sse = float(
            spatial_mean.square().sum(dtype=torch.float64)
            * difference.shape[2]
            * difference.shape[3]
        )
        geometry_sse = max(0.0, total_sse - uniform_sse)
        size = (int(difference.shape[2]), int(difference.shape[3]))
        if size not in boundary_cache:
            boundary_cache[size] = F.interpolate(
                base_boundary[:, None].to(dtype=torch.float32),
                size=size,
                mode="nearest",
            )[:, 0].to(dtype=torch.bool)
        boundary = boundary_cache[size]
        energy = difference.square().mean(dim=1)
        boundary_values = energy[boundary]
        interior_values = energy[~boundary]
        trajectory_rows[name] = {
            "order": next_order(),
            "element_count": int(difference.numel()),
            "total_sse": total_sse,
            "gt_energy_sum": float(ground_truth.square().sum(dtype=torch.float64)),
            "uniform_sse": uniform_sse,
            "geometry_sse": geometry_sse,
            "boundary_energy_sum": float(boundary_values.sum(dtype=torch.float64)),
            "boundary_site_count": int(boundary_values.numel()),
            "interior_energy_sum": float(interior_values.sum(dtype=torch.float64)),
            "interior_site_count": int(interior_values.numel()),
            "shape": list(value.shape),
        }

    modules = dict(model.named_modules())
    for name, module in modules.items():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            hooks.append(
                module.register_forward_pre_hook(
                    lambda mod, args, key=name: bn_pre_hook(key, mod, args)
                )
            )
        if name.endswith(".se.gate"):
            hooks.append(
                module.register_forward_hook(
                    lambda mod, args, output, key=name: se_hook(key, mod, args, output)
                )
            )
        if (
            name in _TRAJECTORY_EXACT
            or _ENCODER_BLOCK.fullmatch(name)
            or _DECODER_BLOCK.fullmatch(name)
        ):
            hooks.append(
                module.register_forward_hook(
                    lambda mod, args, output, key=name: trajectory_hook(
                        key,
                        mod,
                        args,
                        output,
                    )
                )
            )
    if not bn_rows and not hooks:
        raise SegNetAmplitudeTelemetryError("no amplitude telemetry hooks were registered")
    try:
        with torch.inference_mode():
            logits = model(paired_model_input)
    finally:
        for hook in hooks:
            hook.remove()
    if len(bn_rows) == 0 or len(se_rows) == 0 or len(trajectory_rows) == 0:
        raise SegNetAmplitudeTelemetryError("paired telemetry coverage is incomplete")
    return (
        logits[:split_count].detach(),
        logits[split_count:].detach(),
        {
            "schema": "segnet_paired_amplitude.batch.v1",
            "split_count": split_count,
            "bn_layers": bn_rows,
            "se_gates": se_rows,
            "trajectory_layers": trajectory_rows,
        },
    )


def _sum_vectors(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    lengths = {len(row[key]) for row in rows}
    if len(lengths) != 1:
        raise SegNetAmplitudeTelemetryError(f"channel-vector length drift: {key}")
    return [sum(float(row[key][index]) for row in rows) for index in range(lengths.pop())]


def _rms(values: Sequence[float]) -> float:
    return math.sqrt(sum(float(value) ** 2 for value in values) / len(values)) if values else 0.0


def finalize_paired_segnet_amplitude(
    batches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge resumable batch reductions into full-population layer curves."""

    if not batches:
        raise SegNetAmplitudeTelemetryError("amplitude finalization requires batch rows")
    bn_names = tuple(batches[0]["bn_layers"])
    se_names = tuple(batches[0]["se_gates"])
    trajectory_names = tuple(batches[0]["trajectory_layers"])
    if any(
        tuple(batch["bn_layers"]) != bn_names
        or tuple(batch["se_gates"]) != se_names
        or tuple(batch["trajectory_layers"]) != trajectory_names
        for batch in batches
    ):
        raise SegNetAmplitudeTelemetryError("amplitude layer coverage drifted across batches")

    bn_final: list[dict[str, Any]] = []
    for name in bn_names:
        rows = [batch["bn_layers"][name] for batch in batches]
        running_mean = [float(value) for value in rows[0]["running_mean"]]
        running_variance = [float(value) for value in rows[0]["running_variance"]]
        eps = float(rows[0]["eps"])
        if any(
            row["running_mean"] != rows[0]["running_mean"]
            or row["running_variance"] != rows[0]["running_variance"]
            or float(row["eps"]) != eps
            for row in rows[1:]
        ):
            raise SegNetAmplitudeTelemetryError(f"BN running-stat custody drift: {name}")
        count = sum(int(row["sample_count_per_channel"]) for row in rows)
        painted_sum = _sum_vectors(rows, "painted_sum")
        painted_sumsq = _sum_vectors(rows, "painted_sumsq")
        gt_sum = _sum_vectors(rows, "gt_sum")
        gt_sumsq = _sum_vectors(rows, "gt_sumsq")
        painted_mean = [value / count for value in painted_sum]
        gt_mean = [value / count for value in gt_sum]
        painted_variance = [
            max(0.0, square / count - mean**2)
            for square, mean in zip(painted_sumsq, painted_mean, strict=True)
        ]
        gt_variance = [
            max(0.0, square / count - mean**2)
            for square, mean in zip(gt_sumsq, gt_mean, strict=True)
        ]
        scale = [math.sqrt(value + eps) for value in running_variance]
        painted_mean_z = [
            (mean - running) / sigma
            for mean, running, sigma in zip(
                painted_mean,
                running_mean,
                scale,
                strict=True,
            )
        ]
        gt_mean_z = [
            (mean - running) / sigma
            for mean, running, sigma in zip(
                gt_mean,
                running_mean,
                scale,
                strict=True,
            )
        ]
        painted_log_variance_ratio = [
            math.log((value + eps) / (running + eps))
            for value, running in zip(
                painted_variance,
                running_variance,
                strict=True,
            )
        ]
        gt_log_variance_ratio = [
            math.log((value + eps) / (running + eps))
            for value, running in zip(
                gt_variance,
                running_variance,
                strict=True,
            )
        ]
        bn_final.append(
            {
                "layer": name,
                "order": int(rows[0]["order"]),
                "channel_count": len(painted_mean),
                "sample_count_per_channel": count,
                "painted_channel_mean": painted_mean,
                "painted_channel_variance": painted_variance,
                "gt_channel_mean": gt_mean,
                "gt_channel_variance": gt_variance,
                "running_mean": running_mean,
                "running_variance": running_variance,
                "eps": eps,
                "painted_running_mean_z_rms": _rms(painted_mean_z),
                "gt_running_mean_z_rms": _rms(gt_mean_z),
                "painted_running_log_variance_rms": _rms(painted_log_variance_ratio),
                "gt_running_log_variance_rms": _rms(gt_log_variance_ratio),
                "painted_vs_gt_mean_z_rms": _rms(
                    [
                        (painted - gt) / sigma
                        for painted, gt, sigma in zip(
                            painted_mean,
                            gt_mean,
                            scale,
                            strict=True,
                        )
                    ]
                ),
                "painted_vs_gt_log_variance_rms": _rms(
                    [
                        math.log((painted + eps) / (gt + eps))
                        for painted, gt in zip(
                            painted_variance,
                            gt_variance,
                            strict=True,
                        )
                    ]
                ),
            }
        )

    se_final: list[dict[str, Any]] = []
    for name in se_names:
        rows = [batch["se_gates"][name] for batch in batches]
        count = sum(int(row["sample_count_per_channel"]) for row in rows)
        painted_sum = _sum_vectors(rows, "painted_sum")
        painted_sumsq = _sum_vectors(rows, "painted_sumsq")
        gt_sum = _sum_vectors(rows, "gt_sum")
        gt_sumsq = _sum_vectors(rows, "gt_sumsq")
        painted_mean = [value / count for value in painted_sum]
        gt_mean = [value / count for value in gt_sum]
        painted_variance = [
            max(0.0, square / count - mean**2)
            for square, mean in zip(painted_sumsq, painted_mean, strict=True)
        ]
        gt_variance = [
            max(0.0, square / count - mean**2)
            for square, mean in zip(gt_sumsq, gt_mean, strict=True)
        ]
        se_final.append(
            {
                "layer": name,
                "order": int(rows[0]["order"]),
                "channel_count": len(painted_mean),
                "sample_count_per_channel": count,
                "painted_gate_mean": painted_mean,
                "painted_gate_variance": painted_variance,
                "gt_gate_mean": gt_mean,
                "gt_gate_variance": gt_variance,
                "painted_vs_gt_gate_mean_rms": _rms(
                    [
                        painted - gt
                        for painted, gt in zip(
                            painted_mean,
                            gt_mean,
                            strict=True,
                        )
                    ]
                ),
            }
        )

    trajectory_final: list[dict[str, Any]] = []
    for name in trajectory_names:
        rows = [batch["trajectory_layers"][name] for batch in batches]
        totals = {
            key: sum(float(row[key]) for row in rows)
            for key in (
                "element_count",
                "total_sse",
                "gt_energy_sum",
                "uniform_sse",
                "geometry_sse",
                "boundary_energy_sum",
                "boundary_site_count",
                "interior_energy_sum",
                "interior_site_count",
            )
        }
        total_sse = totals["total_sse"]
        boundary_mse = totals["boundary_energy_sum"] / max(
            totals["boundary_site_count"],
            1.0,
        )
        interior_mse = totals["interior_energy_sum"] / max(
            totals["interior_site_count"],
            1.0,
        )
        trajectory_final.append(
            {
                "layer": name,
                "order": int(rows[0]["order"]),
                "relative_mse_vs_gt_energy": total_sse
                / max(totals["gt_energy_sum"], torch.finfo(torch.float64).tiny),
                "uniform_shift_fraction": totals["uniform_sse"] / max(total_sse, 1e-300),
                "geometry_residual_fraction": totals["geometry_sse"] / max(total_sse, 1e-300),
                "boundary_mse": boundary_mse,
                "interior_mse": interior_mse,
                "boundary_to_interior_mse_ratio": (
                    boundary_mse / interior_mse if interior_mse > 0.0 else None
                ),
                **totals,
            }
        )
    trajectory_final.sort(key=lambda row: (row["order"], row["layer"]))
    peak = max(trajectory_final, key=lambda row: row["relative_mse_vs_gt_energy"])
    onset_threshold = 0.1 * float(peak["relative_mse_vs_gt_energy"])
    onset = next(
        row
        for row in trajectory_final
        if float(row["relative_mse_vs_gt_energy"]) >= onset_threshold
    )
    uniform_fraction = float(peak["uniform_shift_fraction"])
    if uniform_fraction >= 0.6:
        association = "AMPLITUDE_STATISTICS_DOMINANT_ASSOCIATION"
    elif uniform_fraction <= 0.4:
        association = "GEOMETRY_BOUNDARY_DOMINANT_ASSOCIATION"
    else:
        association = "MIXED_AMPLITUDE_GEOMETRY_ASSOCIATION"
    bn_peak_mean = max(bn_final, key=lambda row: row["painted_vs_gt_mean_z_rms"])
    bn_peak_variance = max(
        bn_final,
        key=lambda row: row["painted_vs_gt_log_variance_rms"],
    )
    se_peak = max(se_final, key=lambda row: row["painted_vs_gt_gate_mean_rms"])
    return {
        "schema": "segnet_paired_amplitude.n600.v1",
        "pair_count": sum(int(batch["split_count"]) for batch in batches),
        "bn_layers": sorted(bn_final, key=lambda row: (row["order"], row["layer"])),
        "se_gates": sorted(se_final, key=lambda row: (row["order"], row["layer"])),
        "trajectory_curve": trajectory_final,
        "summary": {
            "bn_peak_mean_shift_layer": bn_peak_mean["layer"],
            "bn_peak_mean_shift_z_rms": bn_peak_mean["painted_vs_gt_mean_z_rms"],
            "bn_peak_variance_shift_layer": bn_peak_variance["layer"],
            "bn_peak_log_variance_shift_rms": bn_peak_variance[
                "painted_vs_gt_log_variance_rms"
            ],
            "se_peak_gate_shift_layer": se_peak["layer"],
            "se_peak_gate_mean_rms": se_peak["painted_vs_gt_gate_mean_rms"],
            "trajectory_peak_layer": peak["layer"],
            "trajectory_peak_relative_mse": peak["relative_mse_vs_gt_energy"],
            "trajectory_onset_layer": onset["layer"],
            "trajectory_onset_rule": "first execution layer at or above 10% of measured peak relative MSE",
            "trajectory_peak_uniform_shift_fraction": uniform_fraction,
            "trajectory_peak_boundary_to_interior_mse_ratio": peak[
                "boundary_to_interior_mse_ratio"
            ],
            "mechanism_association": association,
            "causal_status": (
                "ASSOCIATION_ONLY: paired frozen-forward telemetry does not prove that "
                "BN or SE shift caused a particular argmax error"
            ),
        },
        "score_claim": False,
        "promotion_eligible": False,
    }


def compact_amplitude_context(finalized: Mapping[str, Any]) -> dict[str, Any]:
    """Return the bounded tensor-row reference to a full amplitude artifact."""

    summary = finalized["summary"]
    return {
        "bn_peak_mean_shift_layer": summary["bn_peak_mean_shift_layer"],
        "bn_peak_mean_shift_z_rms": summary["bn_peak_mean_shift_z_rms"],
        "bn_peak_variance_shift_layer": summary["bn_peak_variance_shift_layer"],
        "bn_peak_log_variance_shift_rms": summary["bn_peak_log_variance_shift_rms"],
        "se_peak_gate_shift_layer": summary["se_peak_gate_shift_layer"],
        "se_peak_gate_mean_rms": summary["se_peak_gate_mean_rms"],
        "trajectory_onset_layer": summary["trajectory_onset_layer"],
        "trajectory_peak_layer": summary["trajectory_peak_layer"],
        "mechanism_association": summary["mechanism_association"],
        "causal_status": summary["causal_status"],
    }


__all__ = [
    "SegNetAmplitudeTelemetryError",
    "compact_amplitude_context",
    "finalize_paired_segnet_amplitude",
    "measure_paired_segnet_amplitude",
]
