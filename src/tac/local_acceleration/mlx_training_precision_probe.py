# SPDX-License-Identifier: MIT
"""Utilities for advisory MLX mixed-precision training-signal probes.

Nothing in this module changes score authority.  It provides an explicit
scorer-array cast receipt plus deterministic NumPy gradient-fidelity metrics for
the default-OFF fp16/bf16 probe.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

FLOAT_DTYPE_NAMES = frozenset({"float16", "bfloat16", "float32", "float64"})


@dataclass(frozen=True)
class PrecisionGoBars:
    minimum_speedup: float = 1.5
    minimum_global_gradient_cosine: float = 0.99
    minimum_pair_gradient_cosine: float = 0.99
    required_quality_pairs: int = 600

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _dtype_name(value: Any) -> str:
    return str(getattr(value, "dtype", "")).split(".")[-1]


def _is_mlx_like_array(value: Any) -> bool:
    module = type(value).__module__
    return (
        module.startswith("mlx")
        and hasattr(value, "dtype")
        and hasattr(value, "astype")
        and hasattr(value, "shape")
    )


def cast_floating_mlx_arrays(root: Any, target_dtype: Any) -> dict[str, Any]:
    """Recursively cast floating arrays in a plain MLX scorer adapter in place.

    The scorer adapters are lightweight Python objects rather than
    ``mlx.nn.Module`` trees.  This traversal follows object attributes,
    list/tuple entries, and dict values while refusing cycles.  Callables and
    non-MLX objects are left untouched.  The receipt makes promotion-by-input-
    only-cast impossible: an fp16/bf16 probe must show arrays actually changed.
    """

    target_name = str(target_dtype).split(".")[-1]
    if target_name not in {"float16", "bfloat16", "float32"}:
        raise ValueError(f"unsupported target dtype {target_dtype!r}")
    visited: set[int] = set()
    before: dict[str, int] = {}
    after: dict[str, int] = {}
    changed_paths: list[str] = []

    def transform(value: Any, path: str) -> Any:
        if _is_mlx_like_array(value):
            source = _dtype_name(value)
            before[source] = before.get(source, 0) + 1
            if source in FLOAT_DTYPE_NAMES and source != target_name:
                value = value.astype(target_dtype)
                changed_paths.append(path)
            dest = _dtype_name(value)
            after[dest] = after.get(dest, 0) + 1
            return value
        ident = id(value)
        if ident in visited:
            return value
        if isinstance(value, list):
            visited.add(ident)
            for index, item in enumerate(value):
                value[index] = transform(item, f"{path}[{index}]")
            return value
        if isinstance(value, tuple):
            visited.add(ident)
            return tuple(
                transform(item, f"{path}[{index}]") for index, item in enumerate(value)
            )
        if isinstance(value, dict):
            visited.add(ident)
            for key in list(value):
                value[key] = transform(value[key], f"{path}[{key!r}]")
            return value
        if hasattr(value, "__dict__") and not callable(value):
            visited.add(ident)
            for name, item in list(vars(value).items()):
                setattr(value, name, transform(item, f"{path}.{name}"))
        return value

    transform(root, "adapter")
    return {
        "target_dtype": target_name,
        "arrays_before_by_dtype": dict(sorted(before.items())),
        "arrays_after_by_dtype": dict(sorted(after.items())),
        "n_arrays_changed": len(changed_paths),
        "changed_paths": changed_paths,
        "cast_established": len(changed_paths) > 0 and after.get(target_name, 0) > 0,
    }


def gradient_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    """Fixed-order fp64 comparison of two pixel cotangents."""

    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float64).reshape(-1)
    if ref.shape != cand.shape:
        raise ValueError(f"gradient shape mismatch {ref.shape} != {cand.shape}")
    if not np.isfinite(ref).all() or not np.isfinite(cand).all():
        raise ValueError("non-finite gradient")
    ref_norm = float(np.linalg.norm(ref))
    cand_norm = float(np.linalg.norm(cand))
    if ref_norm == 0.0 and cand_norm == 0.0:
        cosine = 1.0
    elif ref_norm == 0.0 or cand_norm == 0.0:
        cosine = 0.0
    else:
        cosine = float(np.dot(ref, cand) / (ref_norm * cand_norm))
    diff = cand - ref
    return {
        "cosine": max(-1.0, min(1.0, cosine)),
        "relative_l2": float(np.linalg.norm(diff) / max(ref_norm, np.finfo(np.float64).tiny)),
        "max_abs": float(np.max(np.abs(diff), initial=0.0)),
        "reference_l2": ref_norm,
        "candidate_l2": cand_norm,
    }


def aggregate_pair_gradient_metrics(rows: Iterable[dict[str, float]]) -> dict[str, float | int]:
    values = list(rows)
    if not values:
        raise ValueError("at least one pair metric is required")
    cosines = np.asarray([float(row["cosine"]) for row in values], dtype=np.float64)
    rel_l2 = np.asarray([float(row["relative_l2"]) for row in values], dtype=np.float64)
    return {
        "n_pairs": len(values),
        "cosine_min": float(np.min(cosines)),
        "cosine_p05": float(np.quantile(cosines, 0.05)),
        "cosine_median": float(np.median(cosines)),
        "cosine_mean": float(np.mean(cosines)),
        "relative_l2_median": float(np.median(rel_l2)),
        "relative_l2_p95": float(np.quantile(rel_l2, 0.95)),
    }


def evaluate_precision_gate(
    *,
    fp32_seconds: float,
    candidate_seconds: float,
    global_cosine: float,
    pair_cosine_min: float,
    quality_pairs: int,
    bars: PrecisionGoBars | None = None,
) -> dict[str, Any]:
    if bars is None:
        bars = PrecisionGoBars()
    for name, value in {
        "fp32_seconds": fp32_seconds,
        "candidate_seconds": candidate_seconds,
    }.items():
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            raise ValueError(f"{name} must be finite and >0, got {value}")
    speedup = float(fp32_seconds) / float(candidate_seconds)
    tests = {
        "speed": speedup >= bars.minimum_speedup,
        "global_gradient_cosine": float(global_cosine)
        >= bars.minimum_global_gradient_cosine,
        "minimum_pair_gradient_cosine": float(pair_cosine_min)
        >= bars.minimum_pair_gradient_cosine,
        "n600_quality_coverage": int(quality_pairs) >= bars.required_quality_pairs,
    }
    return {
        "speedup_x": speedup,
        "go_bars": bars.to_dict(),
        "tests": tests,
        "verdict": "GO" if all(tests.values()) else "NO_GO",
        "verdict_scope": "this dtype policy on the measured real-state MLX scorer window",
    }


__all__ = [
    "PrecisionGoBars",
    "aggregate_pair_gradient_metrics",
    "cast_floating_mlx_arrays",
    "evaluate_precision_gate",
    "gradient_metrics",
]
