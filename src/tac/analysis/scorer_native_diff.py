# SPDX-License-Identifier: MIT
"""Reusable scorer-native analytic and empirical difference instrument.

The instrument keeps the frozen scorers observational.  It reduces activations
inside forward hooks and never retains a full-video activation cube.  Its
empirical object is a product of scorer layer, channel, pooled spatial cell,
spatial-frequency band, and temporal scope.  Its analytic object is derived
directly from frozen weights: BN affine statistics, SE gate functions, local
convolution frequency/phase responses, activation response curves, and the
exact phase-indexed resize operator contract.

Layerwise contraction is a measured directional secant along the painted->GT
trajectory.  It is not mislabeled as a full Jacobian singular-value spectrum.
Likewise, intermediate Fisher weighting is refused unless a custodied head
pullback exists.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any, Final

import numpy as np
import torch
import torch.nn.functional as F

SCHEMA: Final = "scorer_native_diff.v1"
SPATIAL_GRID: Final = (8, 8)
FREQUENCY_BANDS: Final = (
    ("LOW_0_TO_1_16", 0.0, 1.0 / 16.0),
    ("MIDLOW_1_16_TO_1_8", 1.0 / 16.0, 1.0 / 8.0),
    ("MIDHIGH_1_8_TO_1_4", 1.0 / 8.0, 1.0 / 4.0),
    ("HIGH_1_4_TO_CORNER", 1.0 / 4.0, math.sqrt(0.5)),
)
DFT_SAMPLE_FREQUENCIES: Final = (
    ("DC", 0.0, 0.0),
    ("X_PI_4", 0.0, math.pi / 4.0),
    ("Y_PI_4", math.pi / 4.0, 0.0),
    ("DIAG_PI_4", math.pi / 4.0, math.pi / 4.0),
    ("X_PI_2", 0.0, math.pi / 2.0),
    ("Y_PI_2", math.pi / 2.0, 0.0),
    ("DIAG_PI_2", math.pi / 2.0, math.pi / 2.0),
    ("X_NYQUIST", 0.0, math.pi),
    ("Y_NYQUIST", math.pi, 0.0),
    ("CORNER_NYQUIST", math.pi, math.pi),
)

_SEG_ENCODER_BLOCK = re.compile(r"^encoder\.model\.blocks\.\d+\.\d+$")
_SEG_DECODER_BLOCK = re.compile(r"^decoder\.blocks\.\d+$")
_POSE_BLOCK = re.compile(r"^vision\.stages\.\d+\.blocks\.\d+$")
_POSE_RELAY_EXACT = frozenset(
    {
        "vision.stem",
        "vision.final_conv",
        "vision.head",
        "summarizer",
        "hydra.resblock",
        "hydra.in_layer.pose",
        "hydra.res_layer.pose",
        "hydra.final_layer.pose",
    }
)
_SEG_RELAY_EXACT = frozenset(
    {
        "encoder.model.conv_stem",
        "segmentation_head",
    }
)


class ScorerNativeDiffError(RuntimeError):
    """Raised when scorer-native telemetry coverage or accounting fails."""


def _tensor_output(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return next((row for row in value if isinstance(row, torch.Tensor)), None)
    return None


def selected_relay_names(model: torch.nn.Module, scorer: str) -> tuple[str, ...]:
    """Return the stable, topologically ordered relay surface for one scorer."""

    if scorer not in {"segnet", "posenet"}:
        raise ScorerNativeDiffError(f"unsupported scorer: {scorer}")
    names: list[str] = []
    for name, _module in model.named_modules():
        if scorer == "segnet" and (
            name in _SEG_RELAY_EXACT
            or _SEG_ENCODER_BLOCK.fullmatch(name)
            or _SEG_DECODER_BLOCK.fullmatch(name)
        ):
            names.append(name)
        if scorer == "posenet" and (
            name in _POSE_RELAY_EXACT or _POSE_BLOCK.fullmatch(name)
        ):
            names.append(name)
    if not names:
        raise ScorerNativeDiffError(f"no relay modules found for {scorer}")
    return tuple(names)


def _sha256_tensor(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _activation_formula(module: torch.nn.Module) -> str | None:
    if isinstance(module, torch.nn.ReLU):
        return "y=max(0,x)"
    if isinstance(module, torch.nn.SiLU):
        return "y=x*sigmoid(x)"
    if isinstance(module, torch.nn.GELU):
        return f"y=gelu(x,approximate={module.approximate!r})"
    if module.__class__.__name__ == "GELUTanh":
        return "y=0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))"
    if _is_sigmoid(module):
        return "y=1/(1+exp(-x))"
    return None


def _is_sigmoid(module: torch.nn.Module) -> bool:
    return isinstance(module, torch.nn.Sigmoid) or (
        module.__class__.__name__ == "Sigmoid"
    )


def _conv_frequency_samples(module: torch.nn.Conv2d) -> list[dict[str, Any]]:
    """Evaluate the exact finite-kernel correlation polynomial at fixed points."""

    weight = module.weight.detach().to(dtype=torch.float64, device="cpu")
    out_channels, inputs_per_group, height, width = weight.shape
    y = torch.arange(height, dtype=torch.float64)[:, None]
    x = torch.arange(width, dtype=torch.float64)[None, :]
    rows: list[dict[str, Any]] = []
    for label, omega_y, omega_x in DFT_SAMPLE_FREQUENCIES:
        phase = torch.exp(
            -1j
            * (
                float(omega_y) * y.to(dtype=torch.complex128)
                + float(omega_x) * x.to(dtype=torch.complex128)
            )
        )
        response = (weight.to(dtype=torch.complex128) * phase).sum(dim=(-2, -1))
        magnitude = response.abs()
        peak = magnitude.argmax(dim=1)
        peak_response = response[
            torch.arange(out_channels, dtype=torch.long),
            peak,
        ]
        rows.append(
            {
                "frequency_id": label,
                "omega_y": omega_y,
                "omega_x": omega_x,
                "gain_l2_over_input_channels_per_output": (
                    magnitude.square().sum(dim=1).sqrt().tolist()
                ),
                "peak_input_channel_within_group_per_output": peak.tolist(),
                "peak_input_magnitude_per_output": peak_response.abs().tolist(),
                "peak_input_phase_radians_per_output": torch.angle(
                    peak_response
                ).tolist(),
            }
        )
    return rows


def _se_gate_rows(model: torch.nn.Module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    modules = dict(model.named_modules())
    for name, gate in modules.items():
        if not name.endswith(".gate") or not _is_sigmoid(gate):
            continue
        parent_name = name.rsplit(".", 1)[0]
        parent = modules[parent_name]
        fc1 = getattr(parent, "fc1", None)
        fc2 = getattr(parent, "fc2", None)
        if not isinstance(fc1, (torch.nn.Conv2d, torch.nn.Linear)) or not isinstance(
            fc2,
            (torch.nn.Conv2d, torch.nn.Linear),
        ):
            continue
        rows.append(
            {
                "layer": parent_name,
                "formula": (
                    "g(x)=sigmoid(fc2(relu(fc1(global_average_pool(x)))))"
                ),
                "fc1_weight_shape": list(fc1.weight.shape),
                "fc1_weight_sha256": _sha256_tensor(fc1.weight),
                "fc2_weight_shape": list(fc2.weight.shape),
                "fc2_weight_sha256": _sha256_tensor(fc2.weight),
                "fc1_bias_sha256": (
                    _sha256_tensor(fc1.bias) if fc1.bias is not None else None
                ),
                "fc2_bias_sha256": (
                    _sha256_tensor(fc2.bias) if fc2.bias is not None else None
                ),
                "derivation_status": "EXACT_FROM_FROZEN_WEIGHTS",
            }
        )
    return rows


def _layer_scale_rows(model: torch.nn.Module) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if module.__class__.__name__ != "LayerScale2d":
            continue
        gamma = getattr(module, "gamma", None)
        if not isinstance(gamma, torch.Tensor):
            raise ScorerNativeDiffError(f"LayerScale2d gamma absent: {name}")
        rows.append(
            {
                "layer": name,
                "gamma_shape": list(gamma.shape),
                "gamma": gamma.detach().to(dtype=torch.float64).cpu().tolist(),
                "gamma_sha256": _sha256_tensor(gamma),
                "formula": "y_c,h,w=gamma_c*x_c,h,w",
                "derivation_status": "EXACT_FROM_FROZEN_WEIGHTS",
            }
        )
    return rows


def analytic_scorer_knowledge(
    model: torch.nn.Module,
    *,
    scorer: str,
    weights_sha256: str,
) -> dict[str, Any]:
    """Derive the frozen scorer's local analytic amplitude/spectral contract."""

    if model.training:
        raise ScorerNativeDiffError("analytic scorer knowledge requires eval mode")
    if len(weights_sha256) != 64:
        raise ScorerNativeDiffError("weights SHA-256 is malformed")
    bn_rows: list[dict[str, Any]] = []
    conv_rows: list[dict[str, Any]] = []
    activation_rows: list[dict[str, Any]] = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            if module.running_mean is None or module.running_var is None:
                raise ScorerNativeDiffError(f"BN running statistics absent: {name}")
            gamma = (
                module.weight.detach().to(dtype=torch.float64)
                if module.affine
                else torch.ones_like(module.running_mean, dtype=torch.float64)
            )
            beta = (
                module.bias.detach().to(dtype=torch.float64)
                if module.affine
                else torch.zeros_like(module.running_mean, dtype=torch.float64)
            )
            running_mean = module.running_mean.detach().to(dtype=torch.float64)
            running_var = module.running_var.detach().to(dtype=torch.float64)
            scale = gamma / torch.sqrt(running_var + float(module.eps))
            offset = beta - running_mean * scale
            bn_rows.append(
                {
                    "layer": name,
                    "module_type": module.__class__.__name__,
                    "channel_count": int(module.num_features),
                    "running_mean": running_mean.tolist(),
                    "running_variance": running_var.tolist(),
                    "gamma": gamma.tolist(),
                    "beta": beta.tolist(),
                    "eps": float(module.eps),
                    "eval_affine_scale": scale.tolist(),
                    "eval_affine_offset": offset.tolist(),
                    "formula": "y=gamma*(x-running_mean)/sqrt(running_variance+eps)+beta",
                    "derivation_status": "EXACT_FROM_FROZEN_WEIGHTS",
                }
            )
        if isinstance(module, torch.nn.Conv2d):
            conv_rows.append(
                {
                    "layer": name,
                    "weight_shape": list(module.weight.shape),
                    "weight_sha256": _sha256_tensor(module.weight),
                    "groups": int(module.groups),
                    "stride": list(module.stride),
                    "dilation": list(module.dilation),
                    "padding": list(module.padding),
                    "operator_convention": (
                        "torch cross-correlation; exact response "
                        "H(omega)=sum_k weight[k]*exp(-i omega dot k)"
                    ),
                    "frequency_phase_samples": _conv_frequency_samples(module),
                    "exact_reconstruction": (
                        "arbitrary-frequency matrix response is exactly "
                        "re-derived from SHA-pinned finite kernel coefficients"
                    ),
                    "derivation_status": "EXACT_FROM_FROZEN_WEIGHTS",
                }
            )
        formula = _activation_formula(module)
        if formula is not None:
            activation_rows.append(
                {
                    "layer": name,
                    "module_type": module.__class__.__name__,
                    "formula": formula,
                    "derivation_status": "EXACT_MODULE_FUNCTION",
                }
            )
    from tac.optimization.resize_full_kernel import FullResizeKernel

    resize = FullResizeKernel.build()
    stem_conv = next(
        (
            row
            for row in conv_rows
            if row["stride"] != [1, 1]
        ),
        None,
    )
    return {
        "schema": "scorer_native_analytic_knowledge.v1",
        "scorer": scorer,
        "weights_sha256": weights_sha256,
        "relay_layers": list(selected_relay_names(model, scorer)),
        "batchnorm": bn_rows,
        "se_gate_functions": _se_gate_rows(model),
        "layer_scale_functions": _layer_scale_rows(model),
        "convolution_frequency_phase": conv_rows,
        "activation_response_curves": activation_rows,
        "module_inventory": dict(
            sorted(Counter(module.__class__.__name__ for module in model.modules()).items())
        ),
        "resize_and_sampling_overlay": {
            "camera_to_scorer": {
                "operator": (
                    "exact separable align_corners=false bilinear polyphase "
                    "matrix, 874x1164 -> 384x512"
                ),
                "coverage": resize.coverage().to_dict(),
                "single_lti_transfer_function": False,
                "reason": (
                    "rational resize is phase-indexed/polyphase and border "
                    "dependent; uint8 round makes the full bicubic-up/down R "
                    "piecewise affine, so one global scalar Fourier transfer "
                    "would be false"
                ),
                "exact_operator_source": (
                    "tac.optimization.resize_full_kernel.FullResizeKernel"
                ),
            },
            "first_strided_conv": stem_conv,
            "unreachable_rule": (
                "camera directions in exact resize kernel are unreachable at "
                "scorer input; post-resize frequencies above the first "
                "strided-conv Nyquist cannot be independently controlled"
            ),
        },
        "contrast_composition": {
            "contract": (
                "exact BN affine rows plus exact activation formulas; "
                "composition across residual/branch topology must be evaluated "
                "on a trajectory"
            ),
            "status": "ANALYTIC_LOCAL_FACTORS_COMPLETE",
        },
        "score_claim": False,
        "promotion_eligible": False,
    }


def _channel_moments(value: torch.Tensor) -> dict[str, Any]:
    if value.ndim < 2:
        raise ScorerNativeDiffError("relay activation must expose a channel axis")
    dims = tuple(index for index in range(value.ndim) if index != 1)
    count = math.prod(value.shape[index] for index in dims)
    detached = value.detach()
    channel_mean_per_sample = (
        detached.reshape(detached.shape[0], detached.shape[1], -1)
        .mean(dim=2)
        .to(dtype=torch.float64)
    )
    return {
        "sample_count_per_channel": int(count),
        "sum": detached.sum(dim=dims, dtype=torch.float64).cpu().tolist(),
        "sumsq": detached.square()
        .sum(dim=dims, dtype=torch.float64)
        .cpu()
        .tolist(),
        "min": detached.amin(dim=dims).to(dtype=torch.float64).cpu().tolist(),
        "max": detached.amax(dim=dims).to(dtype=torch.float64).cpu().tolist(),
        "channel_mean_trajectory": channel_mean_per_sample.cpu().tolist(),
    }


def _frequency_energy(value: torch.Tensor) -> dict[str, list[float]] | None:
    if value.ndim != 4:
        return None
    height = min(int(value.shape[2]), 32)
    width = min(int(value.shape[3]), 32)
    reduced = F.adaptive_avg_pool2d(value.detach(), (height, width))
    spectrum = torch.fft.rfft2(reduced.to(dtype=torch.float64), norm="ortho")
    power = spectrum.abs().square()
    fy = torch.fft.fftfreq(height, dtype=torch.float64)[:, None]
    fx = torch.fft.rfftfreq(width, dtype=torch.float64)[None, :]
    radius = torch.sqrt(fy.square() + fx.square())
    rows: dict[str, list[float]] = {}
    for index, (name, lower, upper) in enumerate(FREQUENCY_BANDS):
        selected = (radius >= lower) & (
            radius <= upper if index == len(FREQUENCY_BANDS) - 1 else radius < upper
        )
        rows[name] = (
            power[:, :, selected]
            .sum(dim=(0, 2), dtype=torch.float64)
            .cpu()
            .tolist()
        )
    return rows


def _contrast_stats(value: torch.Tensor) -> dict[str, Any]:
    detached = value.detach()
    total_sse = float(detached.square().sum(dtype=torch.float64))
    if detached.ndim == 4:
        uniform = detached.mean(dim=(2, 3), keepdim=True)
        uniform_sse = float(
            uniform.square().sum(dtype=torch.float64)
            * detached.shape[2]
            * detached.shape[3]
        )
        spatial = F.adaptive_avg_pool2d(
            detached.square().mean(dim=1),
            SPATIAL_GRID,
        ).sum(dim=0, dtype=torch.float64)
        spatial_grid = spatial.cpu().tolist()
    else:
        uniform_sse = total_sse
        spatial_grid = None
    dims = tuple(index for index in range(detached.ndim) if index != 1)
    return {
        "element_count": int(detached.numel()),
        "total_sse": total_sse,
        "channel_sse": detached.square()
        .sum(dim=dims, dtype=torch.float64)
        .cpu()
        .tolist(),
        "uniform_sse": uniform_sse,
        "geometry_sse": max(0.0, total_sse - uniform_sse),
        "spatial_energy_grid": spatial_grid,
        "spatial_grid_shape": list(SPATIAL_GRID) if spatial_grid is not None else None,
        "frequency_energy_by_channel": _frequency_energy(detached),
    }


def _feature_transport_stats(
    frame0: torch.Tensor,
    frame1: torch.Tensor,
    *,
    xi: np.ndarray,
    pitch_rad: float,
) -> dict[str, Any] | None:
    if frame0.ndim != 4 or frame1.shape != frame0.shape:
        return None
    from tac.boundary_math.warp_real_luma_frame0 import (
        GroundHomographyGeom,
        homography_from_xi_numpy,
    )

    batch, _channels, height, width = frame0.shape
    if xi.shape != (batch, 6):
        raise ScorerNativeDiffError("xi batch differs from feature transport batch")
    geom = GroundHomographyGeom.eon(
        native_hw=(int(height), int(width)),
        pitch=float(pitch_rad),
    )
    target_y, target_x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    target = torch.stack(
        (
            target_x.reshape(-1),
            target_y.reshape(-1),
            torch.ones(height * width, dtype=torch.float64),
        )
    )
    grids: list[torch.Tensor] = []
    valid_fractions: list[float] = []
    for row in xi:
        homography = homography_from_xi_numpy(row, geom)
        source = torch.from_numpy(np.linalg.inv(homography)) @ target
        denominator = source[2]
        valid = denominator > 1e-9
        source_x = source[0] / denominator.clamp_min(1e-9)
        source_y = source[1] / denominator.clamp_min(1e-9)
        valid &= (
            (source_x >= 0.0)
            & (source_x <= width - 1)
            & (source_y >= 0.0)
            & (source_y <= height - 1)
        )
        normalized_x = 2.0 * (source_x + 0.5) / width - 1.0
        normalized_y = 2.0 * (source_y + 0.5) / height - 1.0
        grid = torch.stack((normalized_x, normalized_y), dim=1).reshape(
            height,
            width,
            2,
        )
        grids.append(grid.to(dtype=frame0.dtype, device=frame0.device))
        valid_fractions.append(float(valid.to(dtype=torch.float64).mean()))
    warped = F.grid_sample(
        frame0,
        torch.stack(grids),
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )
    stats = _contrast_stats(warped - frame1)
    stats["valid_projection_fraction_mean"] = sum(valid_fractions) / len(
        valid_fractions
    )
    stats["transport"] = (
        "stored-PoseNet full-screw ground-homography proxy, scaled to layer grid"
    )
    return stats


def measure_scorer_native_product(
    model: torch.nn.Module,
    *,
    scorer: str,
    grouped_inputs: Mapping[str, torch.Tensor],
    contrasts: Mapping[str, Mapping[str, float]],
    xi: np.ndarray | None = None,
    pitch_rad: float | None = None,
    transport_groups: tuple[str, str] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Run one grouped forward and reduce the scorer-native product in hooks."""

    if model.training:
        raise ScorerNativeDiffError("scorer-native measurement requires eval mode")
    names = tuple(grouped_inputs)
    if len(names) < 2:
        raise ScorerNativeDiffError("grouped measurement needs at least two groups")
    batch_sizes = {int(value.shape[0]) for value in grouped_inputs.values()}
    tail_shapes = {tuple(value.shape[1:]) for value in grouped_inputs.values()}
    if len(batch_sizes) != 1 or len(tail_shapes) != 1:
        raise ScorerNativeDiffError("grouped inputs must share batch and tail shape")
    batch_size = batch_sizes.pop()
    for contrast_name, coefficients in contrasts.items():
        if set(coefficients) - set(names) or not coefficients:
            raise ScorerNativeDiffError(
                f"contrast {contrast_name} references an unknown group"
            )
    relay_names = selected_relay_names(model, scorer)
    modules = dict(model.named_modules())
    rows: dict[str, dict[str, Any]] = {}
    bn_rows: dict[str, dict[str, Any]] = {}
    se_rows: dict[str, dict[str, Any]] = {}
    layer_scale_rows: dict[str, dict[str, Any]] = {}
    hooks: list[Any] = []
    execution_order = 0

    def capture(name: str, _module: torch.nn.Module, _args: Any, output: Any) -> None:
        nonlocal execution_order
        value = _tensor_output(output)
        if value is None or value.shape[0] != batch_size * len(names):
            raise ScorerNativeDiffError(
                f"relay {name} does not expose grouped tensor output"
            )
        execution_order += 1
        split = {
            group: value[index * batch_size : (index + 1) * batch_size]
            for index, group in enumerate(names)
        }
        contrast_rows = {
            contrast_name: _contrast_stats(
                sum(
                    float(coefficient) * split[group]
                    for group, coefficient in coefficients.items()
                )
            )
            for contrast_name, coefficients in contrasts.items()
        }
        transport = None
        if (
            transport_groups is not None
            and xi is not None
            and pitch_rad is not None
        ):
            transport = _feature_transport_stats(
                split[transport_groups[0]],
                split[transport_groups[1]],
                xi=xi,
                pitch_rad=pitch_rad,
            )
        rows[name] = {
            "order": execution_order,
            "module_type": modules[name].__class__.__name__,
            "shape_per_group": list(split[names[0]].shape),
            "feature_values": {
                group: _channel_moments(group_value)
                for group, group_value in split.items()
            },
            "contrasts": contrast_rows,
            "xi_advected_transport": transport,
            "fisher_margin_weighting": (
                "FINAL_HEAD_CALCULATED_BY_CALLER"
                if name in {"segmentation_head", "hydra.final_layer.pose"}
                else "NO_INTERMEDIATE_HEAD_PULLBACK_CUSTODY"
            ),
        }

    def capture_auxiliary(
        destination: dict[str, dict[str, Any]],
        name: str,
        module: torch.nn.Module,
        value: Any,
        *,
        kind: str,
    ) -> None:
        nonlocal execution_order
        tensor = _tensor_output(value)
        if (
            tensor is None
            or tensor.shape[0] % len(names) != 0
            or tensor.shape[0] // len(names) % batch_size != 0
        ):
            raise ScorerNativeDiffError(
                f"{kind} {name} does not expose grouped tensor output"
            )
        execution_order += 1
        leading_per_group = tensor.shape[0] // len(names)
        split = {
            group: tensor[
                index * leading_per_group : (index + 1) * leading_per_group
            ]
            for index, group in enumerate(names)
        }
        row: dict[str, Any] = {
            "order": execution_order,
            "module_type": module.__class__.__name__,
            "shape_per_group": list(split[names[0]].shape),
            "feature_values": {
                group: _channel_moments(group_value)
                for group, group_value in split.items()
            },
        }
        if kind == "batchnorm":
            running_mean = getattr(module, "running_mean", None)
            running_var = getattr(module, "running_var", None)
            if running_mean is None or running_var is None:
                raise ScorerNativeDiffError(
                    f"BN running statistics absent during forward: {name}"
                )
            row.update(
                {
                    "running_mean": running_mean.detach()
                    .to(dtype=torch.float64)
                    .cpu()
                    .tolist(),
                    "running_variance": running_var.detach()
                    .to(dtype=torch.float64)
                    .cpu()
                    .tolist(),
                    "eps": float(getattr(module, "eps", 1e-5)),
                }
            )
        destination[name] = row

    for name in relay_names:
        hooks.append(modules[name].register_forward_hook(
            lambda module, args, output, key=name: capture(
                key,
                module,
                args,
                output,
            )
        ))
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            hooks.append(
                module.register_forward_pre_hook(
                    lambda mod, args, key=name: capture_auxiliary(
                        bn_rows,
                        key,
                        mod,
                        args,
                        kind="batchnorm",
                    )
                )
            )
        if name.endswith(".gate") and _is_sigmoid(module):
            hooks.append(
                module.register_forward_hook(
                    lambda mod, args, output, key=name: capture_auxiliary(
                        se_rows,
                        key,
                        mod,
                        output,
                        kind="se_gate",
                    )
                )
            )
        if module.__class__.__name__ == "LayerScale2d":
            hooks.append(
                module.register_forward_hook(
                    lambda mod, args, output, key=name: capture_auxiliary(
                        layer_scale_rows,
                        key,
                        mod,
                        output,
                        kind="layer_scale",
                    )
                )
            )
    combined = torch.cat([grouped_inputs[name] for name in names], dim=0)
    try:
        with torch.inference_mode():
            output = model(combined)
    finally:
        for hook in hooks:
            hook.remove()
    if tuple(rows) != relay_names:
        raise ScorerNativeDiffError(
            f"relay execution coverage differs: {tuple(rows)} != {relay_names}"
        )
    return output, {
        "schema": "scorer_native_diff.batch.v1",
        "scorer": scorer,
        "group_names": list(names),
        "batch_size": batch_size,
        "contrast_definitions": {
            name: dict(coefficients)
            for name, coefficients in contrasts.items()
        },
        "layers": rows,
        "batchnorm": bn_rows,
        "se_gates": se_rows,
        "layer_scales": layer_scale_rows,
        "transport_scope": (
            "GROUND_HOMOGRAPHY_PROXY"
            if transport_groups is not None
            else "NOT_REQUESTED"
        ),
    }


def _sum_vectors(rows: Sequence[Mapping[str, Any]], key: str) -> list[float]:
    lengths = {len(row[key]) for row in rows}
    if len(lengths) != 1:
        raise ScorerNativeDiffError(f"vector length drift: {key}")
    length = lengths.pop()
    return [
        sum(float(row[key][index]) for row in rows)
        for index in range(length)
    ]


def _sum_nested_vectors(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, list[float]] | None:
    if rows[0][key] is None:
        if any(row[key] is not None for row in rows):
            raise ScorerNativeDiffError(f"optional nested vector drift: {key}")
        return None
    names = tuple(rows[0][key])
    if any(tuple(row[key]) != names for row in rows):
        raise ScorerNativeDiffError(f"nested vector key drift: {key}")
    return {
        name: _sum_vectors([row[key] for row in rows], name)
        for name in names
    }


def _stable_rank_proxy(trajectory: np.ndarray) -> dict[str, float | int]:
    centered = trajectory - trajectory.mean(axis=0, keepdims=True)
    scale = float(np.max(np.abs(centered), initial=0.0))
    if scale == 0.0:
        return {"stable_rank": 0.0, "top_singular_value": 0.0, "samples": len(centered)}
    if not math.isfinite(scale):
        raise ScorerNativeDiffError("trajectory contains non-finite feature values")
    normalized = centered / scale
    if not np.isfinite(normalized).all():
        raise ScorerNativeDiffError("normalized trajectory contains non-finite values")
    matrix = torch.from_numpy(np.ascontiguousarray(normalized))
    frobenius_sq = float(matrix.square().sum())
    vector = torch.ones(matrix.shape[1], dtype=torch.float64)
    vector /= torch.linalg.vector_norm(vector)
    for _ in range(16):
        left = torch.mv(matrix, vector)
        left_norm = float(torch.linalg.vector_norm(left))
        if left_norm == 0.0:
            break
        left /= left_norm
        vector = torch.mv(matrix.T, left)
        norm = float(torch.linalg.vector_norm(vector))
        if norm == 0.0:
            break
        vector /= norm
    top_normalized = float(torch.linalg.vector_norm(torch.mv(matrix, vector)))
    if not math.isfinite(top_normalized):
        raise ScorerNativeDiffError("stable-rank top singular proxy is non-finite")
    top = top_normalized * scale
    return {
        "stable_rank": frobenius_sq
        / max(top_normalized * top_normalized, np.finfo(np.float64).tiny),
        "top_singular_value": top,
        "samples": len(centered),
    }


def _temporal_spectrum(trajectory: np.ndarray) -> dict[str, Any]:
    centered = trajectory - trajectory.mean(axis=0, keepdims=True)
    spectrum = np.fft.rfft(centered, axis=0)
    power = np.square(np.abs(spectrum))
    frequencies = np.fft.rfftfreq(len(centered))
    rows: dict[str, list[float]] = {}
    for index, (name, lower, upper) in enumerate(FREQUENCY_BANDS):
        selected = (frequencies >= lower) & (
            frequencies <= upper if index == len(FREQUENCY_BANDS) - 1 else frequencies < upper
        )
        rows[name] = power[selected].sum(axis=0).tolist()
    return {
        "sampling_axis": "source_pair_index",
        "frequency_energy_by_channel": rows,
    }


def _finalize_auxiliary(
    batches: Sequence[Mapping[str, Any]],
    *,
    key: str,
    group_names: Sequence[str],
) -> list[dict[str, Any]]:
    names = tuple(
        sorted(
            batches[0][key],
            key=lambda name: (
                int(batches[0][key][name]["order"]),
                name,
            ),
        )
    )
    if any(
        set(batch[key]) != set(names)
        or any(
            int(batch[key][name]["order"])
            != int(batches[0][key][name]["order"])
            for name in names
        )
        for batch in batches
    ):
        raise ScorerNativeDiffError(f"{key} layer coverage drift")
    painted_group = (
        "painted_f1"
        if "painted_f1" in group_names
        else "painted_pair"
    )
    gt_group = "gt_f1" if "gt_f1" in group_names else "gt_pair"
    output: list[dict[str, Any]] = []
    for name in names:
        rows = [batch[key][name] for batch in batches]
        values: dict[str, dict[str, Any]] = {}
        for group in group_names:
            group_rows = [row["feature_values"][group] for row in rows]
            count = sum(
                int(row["sample_count_per_channel"]) for row in group_rows
            )
            sums = _sum_vectors(group_rows, "sum")
            sumsq = _sum_vectors(group_rows, "sumsq")
            means = [value / count for value in sums]
            variances = [
                max(0.0, square / count - mean * mean)
                for square, mean in zip(sumsq, means, strict=True)
            ]
            values[group] = {
                "sample_count_per_channel": count,
                "channel_mean": means,
                "channel_variance": variances,
            }
        painted = values[painted_group]
        ground_truth = values[gt_group]
        comparison: dict[str, Any] = {
            "painted_group": painted_group,
            "gt_group": gt_group,
            "mean_rms": math.sqrt(
                sum(
                    (left - right) ** 2
                    for left, right in zip(
                        painted["channel_mean"],
                        ground_truth["channel_mean"],
                        strict=True,
                    )
                )
                / len(painted["channel_mean"])
            ),
            "log_variance_rms": math.sqrt(
                sum(
                    math.log(
                        (left + 1e-12) / (right + 1e-12)
                    )
                    ** 2
                    for left, right in zip(
                        painted["channel_variance"],
                        ground_truth["channel_variance"],
                        strict=True,
                    )
                )
                / len(painted["channel_variance"])
            ),
        }
        result = {
            "layer": name,
            "order": int(rows[0]["order"]),
            "module_type": rows[0]["module_type"],
            "shape_per_group": rows[0]["shape_per_group"],
            "feature_values": values,
            "painted_vs_gt": comparison,
        }
        if key == "batchnorm":
            if any(
                row["running_mean"] != rows[0]["running_mean"]
                or row["running_variance"] != rows[0]["running_variance"]
                or row["eps"] != rows[0]["eps"]
                for row in rows[1:]
            ):
                raise ScorerNativeDiffError(f"BN custody drift: {name}")
            running_mean = rows[0]["running_mean"]
            running_variance = rows[0]["running_variance"]
            eps = float(rows[0]["eps"])
            scale = [
                math.sqrt(float(value) + eps)
                for value in running_variance
            ]
            comparison["painted_running_mean_z_rms"] = math.sqrt(
                sum(
                    ((mean - float(running)) / sigma) ** 2
                    for mean, running, sigma in zip(
                        painted["channel_mean"],
                        running_mean,
                        scale,
                        strict=True,
                    )
                )
                / len(scale)
            )
            comparison["gt_running_mean_z_rms"] = math.sqrt(
                sum(
                    ((mean - float(running)) / sigma) ** 2
                    for mean, running, sigma in zip(
                        ground_truth["channel_mean"],
                        running_mean,
                        scale,
                        strict=True,
                    )
                )
                / len(scale)
            )
            result["running_mean"] = running_mean
            result["running_variance"] = running_variance
            result["eps"] = eps
        output.append(result)
    return sorted(output, key=lambda row: (row["order"], row["layer"]))


def finalize_scorer_native_product(
    batches: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Merge resumable product batches and derive relay/reachability rankings."""

    if not batches:
        raise ScorerNativeDiffError("scorer-native finalization requires batches")
    first = batches[0]
    layer_names = tuple(
        sorted(
            first["layers"],
            key=lambda name: (
                int(first["layers"][name]["order"]),
                name,
            ),
        )
    )
    group_names = tuple(first["group_names"])
    contrast_names = tuple(first["contrast_definitions"])
    if any(
        set(batch["layers"]) != set(layer_names)
        or any(
            int(batch["layers"][name]["order"])
            != int(first["layers"][name]["order"])
            for name in layer_names
        )
        or tuple(batch["group_names"]) != group_names
        or tuple(batch["contrast_definitions"]) != contrast_names
        or set(batch["batchnorm"]) != set(first["batchnorm"])
        or set(batch["se_gates"]) != set(first["se_gates"])
        or set(batch["layer_scales"]) != set(first["layer_scales"])
        or batch["scorer"] != first["scorer"]
        for batch in batches
    ):
        raise ScorerNativeDiffError("scorer-native batch schema drift")
    final_layers: list[dict[str, Any]] = []
    for layer_name in layer_names:
        layer_rows = [batch["layers"][layer_name] for batch in batches]
        values: dict[str, Any] = {}
        trajectories: dict[str, np.ndarray] = {}
        for group in group_names:
            group_rows = [row["feature_values"][group] for row in layer_rows]
            count = sum(int(row["sample_count_per_channel"]) for row in group_rows)
            sums = _sum_vectors(group_rows, "sum")
            sumsq = _sum_vectors(group_rows, "sumsq")
            means = [value / count for value in sums]
            variances = [
                max(0.0, square / count - mean * mean)
                for square, mean in zip(sumsq, means, strict=True)
            ]
            trajectory = np.asarray(
                [
                    sample
                    for row in group_rows
                    for sample in row["channel_mean_trajectory"]
                ],
                dtype=np.float64,
            )
            trajectories[group] = trajectory
            values[group] = {
                "sample_count_per_channel": count,
                "channel_mean": means,
                "channel_variance": variances,
                "channel_min": [
                    min(float(row["min"][index]) for row in group_rows)
                    for index in range(len(means))
                ],
                "channel_max": [
                    max(float(row["max"][index]) for row in group_rows)
                    for index in range(len(means))
                ],
                "trajectory_stable_rank": _stable_rank_proxy(trajectory),
                "across_pair_stationarity_mse": (
                    float(np.square(np.diff(trajectory, axis=0)).mean())
                    if len(trajectory) > 1
                    else None
                ),
                "temporal_frequency": _temporal_spectrum(trajectory),
            }
        contrasts: dict[str, Any] = {}
        for contrast in contrast_names:
            rows = [row["contrasts"][contrast] for row in layer_rows]
            total_sse = sum(float(row["total_sse"]) for row in rows)
            channel_sse = _sum_vectors(rows, "channel_sse")
            spatial = None
            if rows[0]["spatial_energy_grid"] is not None:
                spatial = np.sum(
                    [
                        np.asarray(row["spatial_energy_grid"], dtype=np.float64)
                        for row in rows
                    ],
                    axis=0,
                ).tolist()
            frequency = _sum_nested_vectors(
                rows,
                "frequency_energy_by_channel",
            )
            contrasts[contrast] = {
                "element_count": sum(int(row["element_count"]) for row in rows),
                "total_sse": total_sse,
                "rms": math.sqrt(
                    total_sse
                    / max(sum(int(row["element_count"]) for row in rows), 1)
                ),
                "channel_sse": channel_sse,
                "channel_energy_fraction": [
                    value / max(sum(channel_sse), np.finfo(np.float64).tiny)
                    for value in channel_sse
                ],
                "uniform_shift_fraction": sum(
                    float(row["uniform_sse"]) for row in rows
                )
                / max(total_sse, np.finfo(np.float64).tiny),
                "geometry_residual_fraction": sum(
                    float(row["geometry_sse"]) for row in rows
                )
                / max(total_sse, np.finfo(np.float64).tiny),
                "spatial_energy_grid": spatial,
                "spatial_grid_shape": rows[0]["spatial_grid_shape"],
                "frequency_energy_by_channel": frequency,
            }
        transport_rows = [
            row["xi_advected_transport"]
            for row in layer_rows
            if row["xi_advected_transport"] is not None
        ]
        transport = None
        if transport_rows:
            transport_sse = sum(float(row["total_sse"]) for row in transport_rows)
            transport = {
                "total_sse": transport_sse,
                "element_count": sum(
                    int(row["element_count"]) for row in transport_rows
                ),
                "rms": math.sqrt(
                    transport_sse
                    / max(
                        sum(int(row["element_count"]) for row in transport_rows),
                        1,
                    )
                ),
                "valid_projection_fraction_mean": sum(
                    float(row["valid_projection_fraction_mean"])
                    for row in transport_rows
                )
                / len(transport_rows),
                "transport": transport_rows[0]["transport"],
                "verdict_scope": (
                    "stored-PoseNet full-screw one-ground-plane proxy; prior "
                    "n16 formulation failed rate gate, family remains open"
                ),
            }
        final_layers.append(
            {
                "layer": layer_name,
                "order": int(layer_rows[0]["order"]),
                "module_type": layer_rows[0]["module_type"],
                "shape_per_group": layer_rows[0]["shape_per_group"],
                "feature_values": values,
                "contrasts": contrasts,
                "xi_advected_transport": transport,
                "fisher_margin_weighting": layer_rows[0][
                    "fisher_margin_weighting"
                ],
            }
        )
    primary = (
        "painted_f1_vs_gt_f1"
        if "painted_f1_vs_gt_f1" in contrast_names
        else "painted_pair_vs_gt_pair"
    )
    previous_gap: float | None = None
    for layer in final_layers:
        gap = float(layer["contrasts"][primary]["rms"])
        layer["directional_secant"] = {
            "gap_rms": gap,
            "local_expansion_vs_previous_relay": (
                gap / previous_gap if previous_gap not in (None, 0.0) else None
            ),
            "status": (
                "MEASURED_TRAJECTORY_SECANT_NOT_FULL_JACOBIAN_CONDITION_NUMBER"
            ),
        }
        previous_gap = gap
    downstream = 1.0
    for layer in reversed(final_layers):
        expansion = layer["directional_secant"]["local_expansion_vs_previous_relay"]
        if expansion is not None:
            downstream *= float(expansion)
        gap = float(layer["directional_secant"]["gap_rms"])
        stable_rank = float(
            layer["feature_values"][group_names[-1]][
                "trajectory_stable_rank"
            ]["stable_rank"]
        )
        layer["reachability"] = {
            "downstream_directional_secant_product": downstream,
            "gap_relative_to_downstream_product": gap
            / max(abs(downstream), np.finfo(np.float64).tiny),
            "gt_trajectory_stable_rank": stable_rank,
            "relay_score": (
                abs(downstream)
                / max(
                    gap * max(stable_rank, 1.0),
                    np.finfo(np.float64).tiny,
                )
            ),
            "interpretation": (
                "higher favors low-rank, forgiving relay; advisory directional "
                "selector only"
            ),
        }
    relay_ranking = [
        {
            "rank": rank,
            "layer": layer["layer"],
            **layer["reachability"],
        }
        for rank, layer in enumerate(
            sorted(
                final_layers,
                key=lambda row: -float(row["reachability"]["relay_score"]),
            ),
            start=1,
        )
    ]
    batchnorm = _finalize_auxiliary(
        batches,
        key="batchnorm",
        group_names=group_names,
    )
    se_gates = _finalize_auxiliary(
        batches,
        key="se_gates",
        group_names=group_names,
    )
    layer_scales = _finalize_auxiliary(
        batches,
        key="layer_scales",
        group_names=group_names,
    )
    return {
        "schema": SCHEMA,
        "scorer": first["scorer"],
        "pair_count": sum(int(batch["batch_size"]) for batch in batches),
        "group_names": list(group_names),
        "contrast_definitions": first["contrast_definitions"],
        "layers": final_layers,
        "batchnorm": batchnorm,
        "se_gates": se_gates,
        "layer_scales": layer_scales,
        "relay_ranking": relay_ranking,
        "product_axes": {
            "layer": "every declared relay",
            "channel": "exact channel moments and contrast energy",
            "spatial": f"{SPATIAL_GRID[0]}x{SPATIAL_GRID[1]} pooled native-feature energy grid",
            "within_frame": "painted-vs-GT contrasts",
            "across_frame": "painted and GT temporal contrasts",
            "xi_advected": first["transport_scope"],
            "across_pair": "channel-mean trajectory stationarity",
            "clip": "n-pair aggregate plus temporal spectrum",
            "frequency": [row[0] for row in FREQUENCY_BANDS],
        },
        "limitations": {
            "intermediate_fisher": (
                "NO_INTERMEDIATE_HEAD_PULLBACK_CUSTODY; final decision surface "
                "only, no fabricated per-layer Fisher weight"
            ),
            "jacobian": (
                "directional secants measured; full singular spectrum not "
                "claimed"
            ),
            "spatial": (
                "joint channel x pooled-spatial energy retained; raw full-video "
                "feature tensors deliberately not persisted"
            ),
        },
        "score_claim": False,
        "promotion_eligible": False,
    }


__all__ = [
    "DFT_SAMPLE_FREQUENCIES",
    "FREQUENCY_BANDS",
    "SCHEMA",
    "SPATIAL_GRID",
    "ScorerNativeDiffError",
    "analytic_scorer_knowledge",
    "finalize_scorer_native_product",
    "measure_scorer_native_product",
    "selected_relay_names",
]
