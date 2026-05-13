"""Score-aware ternary quantization-aware training helpers.

The helpers in this module quantize tensors to ``{-scale, 0, +scale}`` with a
calibration objective that can weight errors by scorer sensitivity. They are
archive-building primitives only: metadata records calibration loss and packed
bytes, not contest scores.
"""

from __future__ import annotations

import dataclasses
import json
import math
import struct
from typing import Any

import torch

_MAGIC = b"TQAT"
_VERSION = 1
_PACK_HEADER = ">4sBII"
_CODE_TO_BITS = {-1: 0, 0: 1, 1: 2}
_BITS_TO_CODE = {-1: -1, 0: -1, 1: 0, 2: 1}


@dataclasses.dataclass(frozen=True)
class TernaryQATConfig:
    """Calibration controls for ternary QAT.

    ``rate_lambda`` adds a deterministic nonzero-density penalty during
    threshold search. This makes the helper usable by byte-allocation loops
    without presenting the result as score authority.
    """

    threshold_factor: float = 0.7
    rate_lambda: float = 0.0
    search_threshold: bool = True

    def validate(self) -> None:
        if not math.isfinite(self.threshold_factor) or self.threshold_factor < 0.0:
            raise ValueError("threshold_factor must be finite and non-negative")
        if not math.isfinite(self.rate_lambda) or self.rate_lambda < 0.0:
            raise ValueError("rate_lambda must be finite and non-negative")


@dataclasses.dataclass(frozen=True)
class TernaryCalibration:
    """JSON-serializable calibration metadata for a ternary tensor."""

    shape: tuple[int, ...]
    scale: float
    threshold: float
    threshold_factor: float
    nonzero_count: int
    numel: int
    weighted_mse: float
    score_weight_sum: float
    rate_lambda: float

    def __post_init__(self) -> None:
        if self.numel < 0:
            raise ValueError("numel must be non-negative")
        if self.nonzero_count < 0 or self.nonzero_count > self.numel:
            raise ValueError("nonzero_count must be in [0, numel]")
        if math.prod(self.shape) != self.numel:
            raise ValueError("shape product must equal numel")
        for name in ("scale", "threshold", "weighted_mse", "score_weight_sum"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.scale <= 0.0:
            raise ValueError("scale must be positive")
        if self.threshold < 0.0:
            raise ValueError("threshold must be non-negative")
        if self.numel == 0:
            if self.score_weight_sum != 0.0:
                raise ValueError("empty tensors must have zero score_weight_sum")
        elif self.score_weight_sum <= 0.0:
            raise ValueError("score_weight_sum must be positive")

    def to_dict(self) -> dict[str, Any]:
        """Return stable JSON metadata for artifact manifests."""

        return {
            "shape": list(self.shape),
            "scale": self.scale,
            "threshold": self.threshold,
            "threshold_factor": self.threshold_factor,
            "nonzero_count": self.nonzero_count,
            "numel": self.numel,
            "weighted_mse": self.weighted_mse,
            "score_weight_sum": self.score_weight_sum,
            "rate_lambda": self.rate_lambda,
            "score_claim": False,
            "promotion_eligible": False,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TernaryCalibration:
        """Rehydrate metadata emitted by :meth:`to_dict`."""

        return cls(
            shape=tuple(int(x) for x in data["shape"]),
            scale=float(data["scale"]),
            threshold=float(data["threshold"]),
            threshold_factor=float(data["threshold_factor"]),
            nonzero_count=int(data["nonzero_count"]),
            numel=int(data["numel"]),
            weighted_mse=float(data["weighted_mse"]),
            score_weight_sum=float(data["score_weight_sum"]),
            rate_lambda=float(data["rate_lambda"]),
        )


@dataclasses.dataclass(frozen=True)
class TernaryQuantizedTensor:
    """Ternary codes plus calibration metadata."""

    codes: torch.Tensor
    calibration: TernaryCalibration

    def __post_init__(self) -> None:
        if self.codes.shape != self.calibration.shape:
            raise ValueError("codes shape must match calibration shape")
        if self.codes.dtype != torch.int8:
            raise ValueError("codes must have dtype torch.int8")
        if self.codes.numel() != self.calibration.numel:
            raise ValueError("codes numel must match calibration")
        if self.codes.numel() and not torch.isin(
            self.codes.cpu(), torch.tensor([-1, 0, 1], dtype=torch.int8)
        ).all():
            raise ValueError("codes must be ternary values -1, 0, or +1")

    def dequantize(
        self,
        *,
        dtype: torch.dtype = torch.float32,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        """Decode ternary codes to a floating-point tensor."""

        target_device = device if device is not None else self.codes.device
        return self.codes.to(device=target_device, dtype=dtype) * float(
            self.calibration.scale
        )

    def pack(self) -> bytes:
        """Return a deterministic binary packet for this ternary tensor."""

        return pack_ternary_tensor(self)


def _as_float_tensor(tensor: torch.Tensor, *, label: str) -> torch.Tensor:
    if not torch.is_tensor(tensor):
        raise TypeError(f"{label} must be a torch.Tensor")
    if not torch.is_floating_point(tensor):
        raise ValueError(f"{label} must be floating-point")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{label} must be finite")
    return tensor.detach().float().cpu()


def _broadcast_sensitivity(
    tensor: torch.Tensor,
    score_sensitivity: torch.Tensor | None,
) -> torch.Tensor:
    if score_sensitivity is None:
        return torch.ones_like(tensor)
    sens = _as_float_tensor(score_sensitivity, label="score_sensitivity")
    try:
        sens = torch.broadcast_to(sens, tensor.shape).clone()
    except RuntimeError as exc:
        raise ValueError("score_sensitivity must be broadcastable to tensor") from exc
    if torch.any(sens < 0):
        raise ValueError("score_sensitivity must be non-negative")
    if tensor.numel() == 0:
        return sens
    if float(sens.sum().item()) <= 0.0:
        raise ValueError("score_sensitivity must have positive total weight")
    return sens


def _weighted_mse(
    tensor: torch.Tensor,
    codes: torch.Tensor,
    scale: float,
    weights: torch.Tensor,
) -> float:
    deq = codes.float() * float(scale)
    return float(((tensor - deq).square() * weights).sum().item() / weights.sum().item())


def calibrate_ternary_tensor(
    tensor: torch.Tensor,
    *,
    score_sensitivity: torch.Tensor | None = None,
    config: TernaryQATConfig | None = None,
) -> TernaryCalibration:
    """Find deterministic ternary scale/threshold metadata for ``tensor``.

    With ``score_sensitivity`` supplied, thresholds are selected by weighted
    squared error plus ``rate_lambda * nonzero_fraction``. This is a scorer
    sensitivity hook, not a score claim.
    """

    cfg = config or TernaryQATConfig()
    cfg.validate()
    x = _as_float_tensor(tensor, label="tensor")
    weights = _broadcast_sensitivity(x, score_sensitivity)
    numel = int(x.numel())
    weight_sum = float(weights.sum().item())
    if numel == 0:
        return TernaryCalibration(
            shape=tuple(x.shape),
            scale=1.0,
            threshold=0.0,
            threshold_factor=cfg.threshold_factor,
            nonzero_count=0,
            numel=0,
            weighted_mse=0.0,
            score_weight_sum=weight_sum,
            rate_lambda=cfg.rate_lambda,
        )

    abs_x = x.abs()
    if float(abs_x.max().item()) == 0.0:
        return TernaryCalibration(
            shape=tuple(x.shape),
            scale=1.0,
            threshold=0.0,
            threshold_factor=cfg.threshold_factor,
            nonzero_count=0,
            numel=numel,
            weighted_mse=0.0,
            score_weight_sum=weight_sum,
            rate_lambda=cfg.rate_lambda,
        )

    if cfg.search_threshold:
        candidates = torch.unique(abs_x).sort().values.tolist()
        if 0.0 not in candidates:
            candidates.insert(0, 0.0)
    else:
        weighted_mean_abs = float((abs_x * weights).sum().item() / weight_sum)
        candidates = [max(0.0, cfg.threshold_factor * weighted_mean_abs)]

    best: tuple[float, int, float, float, torch.Tensor] | None = None
    for threshold in candidates:
        threshold_f = float(threshold)
        mask = abs_x > threshold_f
        nonzero = int(mask.sum().item())
        if nonzero == 0:
            scale = 1.0
            codes = torch.zeros_like(x, dtype=torch.int8)
        else:
            selected_w = weights[mask]
            scale = float((abs_x[mask] * selected_w).sum().item() / selected_w.sum().item())
            scale = max(scale, 1e-12)
            codes = torch.where(
                mask,
                torch.sign(x).to(torch.int8),
                torch.zeros_like(x, dtype=torch.int8),
            )
        mse = _weighted_mse(x, codes, scale, weights)
        objective = mse + cfg.rate_lambda * (nonzero / max(numel, 1))
        key = (objective, nonzero, threshold_f, scale, codes)
        if best is None or key[:4] < best[:4]:
            best = key

    assert best is not None
    _objective, nonzero, threshold, scale, codes = best
    mse = _weighted_mse(x, codes, scale, weights)
    return TernaryCalibration(
        shape=tuple(x.shape),
        scale=float(scale),
        threshold=float(threshold),
        threshold_factor=cfg.threshold_factor,
        nonzero_count=int(nonzero),
        numel=numel,
        weighted_mse=float(mse),
        score_weight_sum=weight_sum,
        rate_lambda=cfg.rate_lambda,
    )


def quantize_ternary_tensor(
    tensor: torch.Tensor,
    *,
    calibration: TernaryCalibration | None = None,
    score_sensitivity: torch.Tensor | None = None,
    config: TernaryQATConfig | None = None,
) -> TernaryQuantizedTensor:
    """Quantize ``tensor`` to deterministic ternary codes."""

    x = _as_float_tensor(tensor, label="tensor")
    cal = calibration or calibrate_ternary_tensor(
        x, score_sensitivity=score_sensitivity, config=config
    )
    if tuple(x.shape) != cal.shape:
        raise ValueError("tensor shape must match calibration shape")
    mask = x.abs() > float(cal.threshold)
    codes = torch.where(
        mask,
        torch.sign(x).to(torch.int8),
        torch.zeros_like(x, dtype=torch.int8),
    )
    return TernaryQuantizedTensor(codes=codes, calibration=cal)


def dequantize_ternary_tensor(quantized: TernaryQuantizedTensor) -> torch.Tensor:
    """Decode a :class:`TernaryQuantizedTensor` to float32."""

    return quantized.dequantize()


def ternary_ste(
    tensor: torch.Tensor,
    *,
    calibration: TernaryCalibration | None = None,
    score_sensitivity: torch.Tensor | None = None,
    config: TernaryQATConfig | None = None,
) -> torch.Tensor:
    """Straight-through estimator path for ternary QAT.

    Forward uses ternary dequantized values. Backward is identity with respect
    to ``tensor`` by the standard ``x + (q - x).detach()`` construction.
    """

    quantized = quantize_ternary_tensor(
        tensor,
        calibration=calibration,
        score_sensitivity=score_sensitivity,
        config=config,
    )
    deq = quantized.dequantize(dtype=tensor.dtype, device=tensor.device)
    return tensor + (deq - tensor).detach()


def _pack_codes_2bit(codes: torch.Tensor) -> bytes:
    flat = codes.reshape(-1).cpu().tolist()
    out = bytearray((len(flat) + 3) // 4)
    for idx, code in enumerate(flat):
        code_int = int(code)
        if code_int not in _CODE_TO_BITS:
            raise ValueError("codes must be ternary values -1, 0, or +1")
        shift = 6 - 2 * (idx % 4)
        out[idx // 4] |= _CODE_TO_BITS[code_int] << shift
    return bytes(out)


def _unpack_codes_2bit(payload: bytes, *, numel: int, shape: tuple[int, ...]) -> torch.Tensor:
    values: list[int] = []
    for byte in payload:
        for shift in (6, 4, 2, 0):
            if len(values) == numel:
                break
            bits = (byte >> shift) & 0b11
            if bits == 3:
                raise ValueError("reserved ternary code 3 encountered")
            values.append(_BITS_TO_CODE[bits])
    if len(values) != numel:
        raise ValueError("packed payload shorter than calibration numel")
    return torch.tensor(values, dtype=torch.int8).reshape(shape)


def pack_ternary_tensor(quantized: TernaryQuantizedTensor) -> bytes:
    """Pack ternary codes and metadata into deterministic bytes."""

    meta = json.dumps(
        quantized.calibration.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload = _pack_codes_2bit(quantized.codes)
    return struct.pack(_PACK_HEADER, _MAGIC, _VERSION, len(meta), len(payload)) + meta + payload


def unpack_ternary_tensor(blob: bytes) -> TernaryQuantizedTensor:
    """Decode bytes emitted by :func:`pack_ternary_tensor`."""

    header_size = struct.calcsize(_PACK_HEADER)
    if len(blob) < header_size:
        raise ValueError("blob too short for ternary QAT header")
    magic, version, meta_len, payload_len = struct.unpack(_PACK_HEADER, blob[:header_size])
    if magic != _MAGIC:
        raise ValueError(f"bad ternary QAT magic: {magic!r}")
    if version != _VERSION:
        raise ValueError(f"unsupported ternary QAT version: {version}")
    expected_len = header_size + meta_len + payload_len
    if len(blob) != expected_len:
        raise ValueError("ternary QAT blob length mismatch")
    meta_raw = blob[header_size : header_size + meta_len]
    payload = blob[header_size + meta_len :]
    calibration = TernaryCalibration.from_dict(json.loads(meta_raw.decode("utf-8")))
    codes = _unpack_codes_2bit(
        payload,
        numel=calibration.numel,
        shape=calibration.shape,
    )
    return TernaryQuantizedTensor(codes=codes, calibration=calibration)


__all__ = [
    "TernaryCalibration",
    "TernaryQATConfig",
    "TernaryQuantizedTensor",
    "calibrate_ternary_tensor",
    "dequantize_ternary_tensor",
    "pack_ternary_tensor",
    "quantize_ternary_tensor",
    "ternary_ste",
    "unpack_ternary_tensor",
]
