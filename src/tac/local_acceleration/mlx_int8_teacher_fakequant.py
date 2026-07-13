# SPDX-License-Identifier: MIT
"""Default-OFF W8A8 fake-quant instrumentation for the frozen MLX scorers.

This module simulates int8 storage/activation quantization while retaining
float32 convolution accumulation.  It is a quality/gradient probe, not a claim
that MLX dispatches an int8 convolution kernel.  Weight groups are explicit in
the returned receipt and activation quantization uses an identity STE.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Int8FakeQuantPolicy:
    qmin: int = -127
    qmax: int = 127
    weight_scale: str = "symmetric_absmax_per_operator_tensor"
    explicit_head_weight_scale: str = "symmetric_absmax_per_stored_kernel_slice"
    activation_scale: str = "symmetric_absmax_dynamic_per_operator_input"
    accumulation: str = "float32"
    backward: str = "identity_ste_through_qdq"

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def fake_quantize_int8_ste(value: Any, *, qmax: int = 127) -> Any:
    """Symmetric per-tensor int8 QDQ with identity STE for MLX arrays."""

    import mlx.core as mx

    if qmax < 1 or qmax > 127:
        raise ValueError("qmax must be in 1..127")
    x = value.astype(mx.float32)
    absmax = mx.max(mx.abs(x))
    scale = mx.maximum(absmax / float(qmax), mx.array(1.0e-12, dtype=mx.float32))
    quantized = mx.clip(mx.round(x / scale), -qmax, qmax)
    dequantized = quantized * scale
    return x + mx.stop_gradient(dequantized - x)


class _ActivationInt8STEProxy:
    def __init__(self, operator: Any, *, qmax: int):
        self.operator = operator
        self.qmax = int(qmax)

    def __call__(self, value: Any, *args: Any, **kwargs: Any) -> Any:
        return self.operator(fake_quantize_int8_ste(value, qmax=self.qmax), *args, **kwargs)


def instrument_frozen_scorer_w8a8_fakequant(
    root: Any, policy: Int8FakeQuantPolicy | None = None
) -> tuple[Any, dict[str, object]]:
    """Instrument every supported Conv2d/Linear leaf in a scorer adapter.

    Standard MLX Conv2d/Linear and the repository's reference/custom convolution
    adapters receive one symmetric scale per operator weight tensor.  The
    explicit SegNet head stores weights as kernel-position/input-channel slices,
    so each stored slice is one declared group.  Bias and normalization arrays
    remain float32.  Unsupported weight-bearing callables fail closed.
    """

    import mlx.core as mx

    policy = policy or Int8FakeQuantPolicy()
    seen: set[int] = set()
    wrapped_paths: list[str] = []
    weight_groups: list[dict[str, object]] = []

    def qdq_weight(value: Any) -> Any:
        quantized = fake_quantize_int8_ste(value, qmax=policy.qmax)
        return mx.stop_gradient(quantized)

    def is_standard_leaf(value: Any) -> bool:
        module = type(value).__module__
        name = type(value).__name__
        return module.startswith("mlx.nn.layers") and name in {"Conv2d", "Linear"}

    def transform(value: Any, path: str) -> Any:
        ident = id(value)
        if ident in seen:
            return value
        seen.add(ident)
        class_name = type(value).__name__

        if is_standard_leaf(value):
            if "weight" not in value:
                raise TypeError(f"{path} is a weight-bearing MLX leaf without a weight")
            value["weight"] = qdq_weight(value["weight"])
            weight_groups.append({"path": f"{path}.weight", "grouping": "operator_tensor", "groups": 1})
            wrapped_paths.append(path)
            return _ActivationInt8STEProxy(value, qmax=policy.qmax)

        if class_name in {
            "MLXReferenceConv2dAdapter",
            "MLXCustomKernelStridedGroupedConvAdapter",
        }:
            value.weight = qdq_weight(value.weight)
            weight_groups.append({"path": f"{path}.weight", "grouping": "operator_tensor", "groups": 1})
            wrapped_paths.append(path)
            return _ActivationInt8STEProxy(value, qmax=policy.qmax)

        if class_name == "MLXExplicitSpatialConv2dAdapter":
            rebuilt = []
            for index, (kh, kw, channel, weight) in enumerate(value.terms):
                rebuilt.append((kh, kw, channel, qdq_weight(weight)))
                weight_groups.append(
                    {
                        "path": f"{path}.terms[{index}].weight",
                        "grouping": "stored_kernel_slice",
                        "groups": 1,
                    }
                )
            value.terms = rebuilt
            wrapped_paths.append(path)
            return _ActivationInt8STEProxy(value, qmax=policy.qmax)

        if isinstance(value, list):
            for index, child in enumerate(value):
                value[index] = transform(child, f"{path}[{index}]")
            return value
        if isinstance(value, tuple):
            return tuple(transform(child, f"{path}[{index}]") for index, child in enumerate(value))
        if isinstance(value, dict):
            for key in list(value):
                value[key] = transform(value[key], f"{path}[{key!r}]")
            return value
        if hasattr(value, "__dict__"):
            for name, child in list(vars(value).items()):
                setattr(value, name, transform(child, f"{path}.{name}"))
        return value

    instrumented = transform(root, "adapter")
    if not wrapped_paths or not weight_groups:
        raise RuntimeError("no supported Conv2d/Linear leaves were instrumented")
    return instrumented, {
        "policy": policy.to_dict(),
        "wrapped_operator_count": len(wrapped_paths),
        "weight_group_count": len(weight_groups),
        "wrapped_operator_paths": wrapped_paths,
        "weight_groups": weight_groups,
        "native_int8_kernel_claim": False,
        "quality_interpretation": "W8A8_QDQ_FLOAT32_ACCUM_WITH_STE",
    }


__all__ = [
    "Int8FakeQuantPolicy",
    "fake_quantize_int8_ste",
    "instrument_frozen_scorer_w8a8_fakequant",
]
