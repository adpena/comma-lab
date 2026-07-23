# SPDX-License-Identifier: MIT
"""Score-neutral internal telemetry for the frozen Torch SegNet.

The immutable upstream scorer remains the forward authority.  This module only
registers read-only hooks, reduces activations while they are live, and removes
every hook on exit.  Analysis callers default telemetry on; training callers
default it off and must state why/cadence before opting in.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn.functional as F

CLASS_NAMES = ("Road", "Lane", "Undrivable", "Movable", "MyCar")
_ENCODER_BLOCK = re.compile(r"^encoder\.model\.blocks\.\d+\.\d+$")
_SE_MODULE = re.compile(r"^encoder\.model\.blocks\.\d+\.\d+\.se$")
_DECODER_BLOCK = re.compile(r"^decoder\.blocks\.\d+$")


class SegNetTelemetryError(RuntimeError):
    """Raised when telemetry policy, coverage, or score neutrality fails."""


@dataclass(frozen=True, slots=True)
class SegNetTelemetryPolicy:
    """Invocation-specific policy for the optional hook layer."""

    invocation: Literal["analysis", "training"]
    enabled: bool
    cadence: int
    reason: str

    def __post_init__(self) -> None:
        if self.cadence < 0:
            raise SegNetTelemetryError("telemetry cadence must be nonnegative")
        if not self.reason.strip():
            raise SegNetTelemetryError("telemetry policy requires a reason")
        if self.invocation == "analysis" and not self.enabled:
            raise SegNetTelemetryError("analysis policy must default telemetry on")
        if self.invocation == "training" and self.enabled and self.cadence < 1:
            raise SegNetTelemetryError("training telemetry opt-in requires cadence >= 1")

    @classmethod
    def analysis_default(cls) -> SegNetTelemetryPolicy:
        return cls(
            invocation="analysis",
            enabled=True,
            cadence=1,
            reason="analysis invocation requires complete internal observability",
        )

    @classmethod
    def training_default(
        cls,
        *,
        reason: str = "disabled in the training hot loop to avoid activation retention",
    ) -> SegNetTelemetryPolicy:
        return cls(invocation="training", enabled=False, cadence=0, reason=reason)

    @classmethod
    def training_sampled(cls, *, cadence: int, reason: str) -> SegNetTelemetryPolicy:
        return cls(invocation="training", enabled=True, cadence=int(cadence), reason=reason)


def _tensor_from_output(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        return next((item for item in value if isinstance(item, torch.Tensor)), None)
    return None


def _boundary_mask(labels: torch.Tensor) -> torch.Tensor:
    """Return a two-sided four-neighbor boundary mask for ``(N,H,W)`` labels."""

    if labels.ndim != 3:
        raise SegNetTelemetryError(f"labels must be (N,H,W), got {tuple(labels.shape)}")
    boundary = torch.zeros_like(labels, dtype=torch.bool)
    horizontal = labels[:, :, 1:] != labels[:, :, :-1]
    vertical = labels[:, 1:, :] != labels[:, :-1, :]
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal
    boundary[:, 1:, :] |= vertical
    boundary[:, :-1, :] |= vertical
    return boundary


def _quantiles(values: torch.Tensor) -> dict[str, float | None]:
    flat = values.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if flat.numel() == 0:
        return {"min": None, "q01": None, "q10": None, "median": None, "q90": None, "q99": None, "max": None}
    probs = torch.tensor([0.0, 0.01, 0.10, 0.50, 0.90, 0.99, 1.0], dtype=torch.float64)
    measured = torch.quantile(flat, probs)
    return {
        name: float(value)
        for name, value in zip(
            ("min", "q01", "q10", "median", "q90", "q99", "max"),
            measured,
            strict=True,
        )
    }


def summarize_ordered_pair_margins(
    logits: torch.Tensor,
    *,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> list[dict[str, Any]]:
    """Summarize all 20 sided winner-neighbor margins at boundary pixels."""

    if logits.ndim != 4 or logits.shape[1] != len(class_names):
        raise SegNetTelemetryError(
            f"logits must be (N,{len(class_names)},H,W), got {tuple(logits.shape)}"
        )
    labels = logits.argmax(dim=1)
    rows: list[dict[str, Any]] = []
    for winner in range(len(class_names)):
        for rival in range(len(class_names)):
            if winner == rival:
                continue
            side = torch.zeros_like(labels, dtype=torch.bool)
            horizontal = labels[:, :, 1:] != labels[:, :, :-1]
            side[:, :, :-1] |= (
                horizontal
                & (labels[:, :, :-1] == winner)
                & (labels[:, :, 1:] == rival)
            )
            side[:, :, 1:] |= (
                horizontal
                & (labels[:, :, 1:] == winner)
                & (labels[:, :, :-1] == rival)
            )
            vertical = labels[:, 1:, :] != labels[:, :-1, :]
            side[:, :-1, :] |= (
                vertical
                & (labels[:, :-1, :] == winner)
                & (labels[:, 1:, :] == rival)
            )
            side[:, 1:, :] |= (
                vertical
                & (labels[:, 1:, :] == winner)
                & (labels[:, :-1, :] == rival)
            )
            margin = logits[:, winner] - logits[:, rival]
            samples = margin[side]
            rows.append(
                {
                    "winner_id": winner,
                    "winner": class_names[winner],
                    "rival_id": rival,
                    "rival": class_names[rival],
                    "orientation": f"{class_names[winner]}->{class_names[rival]}",
                    "boundary_pixel_count": int(samples.numel()),
                    "margin_quantiles": _quantiles(samples),
                }
            )
    return rows


def extract_ordered_pair_boundary_samples(
    logits: torch.Tensor,
    *,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict[str, dict[str, torch.Tensor]]:
    """Return ephemeral exact margin/coordinate tensors for all 20 orientations."""

    if logits.ndim != 4 or logits.shape[1] != len(class_names):
        raise SegNetTelemetryError(
            f"logits must be (N,{len(class_names)},H,W), got {tuple(logits.shape)}"
        )
    labels = logits.argmax(dim=1)
    result: dict[str, dict[str, torch.Tensor]] = {}
    for winner in range(len(class_names)):
        for rival in range(len(class_names)):
            if winner == rival:
                continue
            side = torch.zeros_like(labels, dtype=torch.bool)
            horizontal = labels[:, :, 1:] != labels[:, :, :-1]
            side[:, :, :-1] |= (
                horizontal
                & (labels[:, :, :-1] == winner)
                & (labels[:, :, 1:] == rival)
            )
            side[:, :, 1:] |= (
                horizontal
                & (labels[:, :, 1:] == winner)
                & (labels[:, :, :-1] == rival)
            )
            vertical = labels[:, 1:, :] != labels[:, :-1, :]
            side[:, :-1, :] |= (
                vertical
                & (labels[:, :-1, :] == winner)
                & (labels[:, 1:, :] == rival)
            )
            side[:, 1:, :] |= (
                vertical
                & (labels[:, 1:, :] == winner)
                & (labels[:, :-1, :] == rival)
            )
            margins = (logits[:, winner] - logits[:, rival])[side]
            result[f"{class_names[winner]}->{class_names[rival]}"] = {
                "margins": margins.detach(),
                "coordinates_nyx": torch.nonzero(side, as_tuple=False).detach(),
            }
    return result


def _activation_energy(
    activation: torch.Tensor,
    boundary: torch.Tensor,
) -> dict[str, Any]:
    if activation.ndim != 4:
        return {
            "shape": list(activation.shape),
            "supported": False,
            "reason": "activation is not NCHW",
        }
    layer_boundary = F.interpolate(
        boundary[:, None].to(dtype=torch.float32),
        size=activation.shape[-2:],
        mode="nearest",
    )[:, 0].to(dtype=torch.bool)
    energy = activation.detach().to(dtype=torch.float64).square().mean(dim=1)
    boundary_values = energy[layer_boundary]
    interior_values = energy[~layer_boundary]
    return {
        "shape": list(activation.shape),
        "supported": True,
        "boundary_sample_count": int(boundary_values.numel()),
        "interior_sample_count": int(interior_values.numel()),
        "boundary_mean_square": (
            float(boundary_values.mean()) if boundary_values.numel() else None
        ),
        "interior_mean_square": (
            float(interior_values.mean()) if interior_values.numel() else None
        ),
        "boundary_to_interior_energy_ratio": (
            float(boundary_values.mean() / interior_values.mean())
            if boundary_values.numel()
            and interior_values.numel()
            and float(interior_values.mean()) > 0.0
            else None
        ),
    }


class SegNetInternalTelemetry:
    """Context-managed read-only hook layer for upstream Torch SegNet."""

    def __init__(
        self,
        model: torch.nn.Module,
        *,
        policy: SegNetTelemetryPolicy | None = None,
        class_names: tuple[str, ...] = CLASS_NAMES,
    ) -> None:
        self.model = model
        self.policy = policy or SegNetTelemetryPolicy.analysis_default()
        self.class_names = class_names
        self._hooks: list[Any] = []
        self._captures: dict[str, torch.Tensor | None] = {}
        self._expected: dict[str, list[str]] = {
            "stem": [],
            "encoder_blocks": [],
            "se_pre": [],
            "se_post": [],
            "decoder_blocks": [],
            "decoder_skips": [],
            "final_logits": [],
        }
        self._closed = False
        if self.policy.enabled:
            self._register_hooks()

    def __enter__(self) -> SegNetInternalTelemetry:
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _tb: Any) -> None:
        self.close()

    def _capture(self, key: str, value: Any) -> None:
        tensor = _tensor_from_output(value)
        self._captures[key] = None if tensor is None else tensor.detach()

    def _register_hooks(self) -> None:
        modules = dict(self.model.named_modules())
        stem_name = "encoder.model.conv_stem"
        if stem_name not in modules:
            raise SegNetTelemetryError(f"required SegNet stem is absent: {stem_name}")
        self._expected["stem"].append(stem_name)
        self._hooks.append(
            modules[stem_name].register_forward_hook(
                lambda _module, _args, output, key=stem_name: self._capture(key, output)
            )
        )

        for name, module in modules.items():
            if _ENCODER_BLOCK.fullmatch(name):
                self._expected["encoder_blocks"].append(name)
                self._hooks.append(
                    module.register_forward_hook(
                        lambda _module, _args, output, key=name: self._capture(key, output)
                    )
                )
            if _SE_MODULE.fullmatch(name):
                pre_key = f"{name}.pre"
                post_key = f"{name}.post"
                self._expected["se_pre"].append(pre_key)
                self._expected["se_post"].append(post_key)
                self._hooks.append(
                    module.register_forward_pre_hook(
                        lambda _module, args, key=pre_key: self._capture(key, args)
                    )
                )
                self._hooks.append(
                    module.register_forward_hook(
                        lambda _module, _args, output, key=post_key: self._capture(key, output)
                    )
                )
            if _DECODER_BLOCK.fullmatch(name):
                skip_key = f"{name}.skip"
                self._expected["decoder_blocks"].append(name)
                self._expected["decoder_skips"].append(skip_key)

                def capture_decoder_input(
                    _module: torch.nn.Module,
                    args: tuple[Any, ...],
                    kwargs: dict[str, Any],
                    *,
                    key: str = skip_key,
                ) -> None:
                    skip = args[3] if len(args) > 3 else kwargs.get("skip_connection")
                    self._captures[key] = skip.detach() if isinstance(skip, torch.Tensor) else None

                self._hooks.append(
                    module.register_forward_pre_hook(capture_decoder_input, with_kwargs=True)
                )
                self._hooks.append(
                    module.register_forward_hook(
                        lambda _module, _args, output, key=name: self._capture(key, output)
                    )
                )

        head_name = "segmentation_head"
        if head_name not in modules:
            raise SegNetTelemetryError(f"required SegNet head is absent: {head_name}")
        self._expected["final_logits"].append(head_name)
        self._hooks.append(
            modules[head_name].register_forward_hook(
                lambda _module, _args, output, key=head_name: self._capture(key, output)
            )
        )
        if not self._expected["encoder_blocks"] or not self._expected["se_pre"]:
            raise SegNetTelemetryError("SegNet encoder block or squeeze-excite taps are absent")
        if not self._expected["decoder_blocks"]:
            raise SegNetTelemetryError("SegNet decoder block taps are absent")

    def close(self) -> None:
        if self._closed:
            return
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._captures.clear()
        self._closed = True

    def run(self, model_input: torch.Tensor) -> tuple[torch.Tensor, dict[str, Any] | None]:
        """Run one frozen forward and return logits plus an ephemeral summary."""

        if self._closed:
            raise SegNetTelemetryError("telemetry context is closed")
        if not self.policy.enabled:
            return self.model(model_input), None
        self._captures.clear()
        logits = self.model(model_input)
        summary = self._summarize(logits)
        self._captures.clear()
        return logits, summary

    def _summarize(self, logits: torch.Tensor) -> dict[str, Any]:
        captured_logits = self._captures.get("segmentation_head")
        if captured_logits is None or not torch.equal(captured_logits, logits.detach()):
            raise SegNetTelemetryError("final-logit hook did not capture the exact forward output")
        missing = [
            key
            for category, keys in self._expected.items()
            if category != "decoder_skips"
            for key in keys
            if key not in self._captures or self._captures[key] is None
        ]
        if missing:
            raise SegNetTelemetryError(f"required telemetry taps were not populated: {missing[:5]}")
        labels = logits.detach().argmax(dim=1)
        boundary = _boundary_mask(labels)
        layer_rows: dict[str, Any] = {}
        for name in sorted(self._captures):
            value = self._captures[name]
            if value is None:
                layer_rows[name] = {
                    "present": False,
                    "reason": "decoder block has no skip connection",
                }
            else:
                layer_rows[name] = {"present": True, **_activation_energy(value, boundary)}
        class_energy = logits.detach().to(dtype=torch.float64).square().mean(dim=(0, 2, 3))
        return {
            "schema": "segnet_internal_telemetry.forward.v1",
            "policy": {
                "invocation": self.policy.invocation,
                "enabled": self.policy.enabled,
                "cadence": self.policy.cadence,
                "reason": self.policy.reason,
            },
            "coverage": {
                category: list(keys) for category, keys in sorted(self._expected.items())
            },
            "layer_boundary_energy": layer_rows,
            "per_class_logit_energy": {
                name: float(class_energy[index])
                for index, name in enumerate(self.class_names)
            },
            "boundary_pixel_count": int(boundary.sum()),
            "ordered_pair_margins": summarize_ordered_pair_margins(
                logits.detach(),
                class_names=self.class_names,
            ),
        }


def assert_telemetry_argmax_identity(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    *,
    class_names: tuple[str, ...] = CLASS_NAMES,
) -> dict[str, Any]:
    """Run OFF and ON forwards and fail unless argmax is exactly identical."""

    with torch.inference_mode():
        off_logits = model(model_input)
        with SegNetInternalTelemetry(
            model,
            policy=SegNetTelemetryPolicy.analysis_default(),
            class_names=class_names,
        ) as telemetry:
            on_logits, summary = telemetry.run(model_input)
    assert summary is not None
    off_argmax = off_logits.argmax(dim=1)
    on_argmax = on_logits.argmax(dim=1)
    if not torch.equal(off_argmax, on_argmax):
        mismatch = int(torch.count_nonzero(off_argmax != on_argmax))
        raise SegNetTelemetryError(
            f"telemetry changed frozen SegNet argmax at {mismatch} pixels"
        )
    return {
        "schema": "segnet_internal_telemetry.identity.v1",
        "argmax_identical": True,
        "argmax_mismatch_count": 0,
        "logits_bitwise_identical": bool(torch.equal(off_logits, on_logits)),
        "sample_count": int(off_logits.shape[0]),
        "telemetry_summary": summary,
    }


def measure_erf_response(
    model: torch.nn.Module,
    model_input: torch.Tensor,
    *,
    y: int,
    x: int,
    winner: int,
    rival: int,
) -> dict[str, Any]:
    """Measure one input-gradient ERF response for a winner/rival margin probe."""

    if model_input.ndim != 4 or model_input.shape[0] != 1:
        raise SegNetTelemetryError("ERF probe requires one NCHW model input")
    probe = model_input.detach().clone().requires_grad_(True)
    logits = model(probe)
    if not (0 <= y < logits.shape[-2] and 0 <= x < logits.shape[-1]):
        raise SegNetTelemetryError("ERF probe point is outside the logit grid")
    if winner == rival or not (0 <= winner < logits.shape[1]) or not (0 <= rival < logits.shape[1]):
        raise SegNetTelemetryError("ERF winner/rival ids are invalid")
    margin = logits[0, winner, y, x] - logits[0, rival, y, x]
    gradient = torch.autograd.grad(margin, probe, retain_graph=False, create_graph=False)[0]
    energy = gradient.detach().to(dtype=torch.float64).square().sum(dim=1)[0]
    yy = torch.arange(energy.shape[0], device=energy.device, dtype=torch.float64)
    xx = torch.arange(energy.shape[1], device=energy.device, dtype=torch.float64)
    grid_y, grid_x = torch.meshgrid(yy, xx, indexing="ij")
    radius = torch.sqrt((grid_y - float(y)) ** 2 + (grid_x - float(x)) ** 2)
    order = torch.argsort(radius.reshape(-1))
    cumulative = torch.cumsum(energy.reshape(-1)[order], dim=0)
    total = cumulative[-1]
    if float(total) <= 0.0:
        raise SegNetTelemetryError("ERF probe has zero gradient energy")
    normalized = cumulative / total
    ordered_radius = radius.reshape(-1)[order]

    def radius_at(fraction: float) -> float:
        index = int(torch.searchsorted(normalized, torch.tensor(fraction, device=normalized.device)))
        return float(ordered_radius[min(index, len(ordered_radius) - 1)])

    return {
        "schema": "segnet_internal_telemetry.erf_probe.v1",
        "probe": {"y": int(y), "x": int(x), "winner": int(winner), "rival": int(rival)},
        "margin": float(margin.detach()),
        "gradient_energy": float(total),
        "r50_pixels": radius_at(0.50),
        "r90_pixels": radius_at(0.90),
        "r99_pixels": radius_at(0.99),
    }


__all__ = [
    "CLASS_NAMES",
    "SegNetInternalTelemetry",
    "SegNetTelemetryError",
    "SegNetTelemetryPolicy",
    "assert_telemetry_argmax_identity",
    "extract_ordered_pair_boundary_samples",
    "measure_erf_response",
    "summarize_ordered_pair_margins",
]
