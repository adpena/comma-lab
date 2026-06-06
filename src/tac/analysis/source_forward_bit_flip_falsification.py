# SPDX-License-Identifier: MIT
"""Shared source-forward parity negative-control proof helpers."""

from __future__ import annotations

import json
from collections.abc import Mapping
from hashlib import sha256
from typing import Any

import numpy as np

BIT_FLIP_FALSIFICATION_SCHEMA = (
    "snerv_official_source_forward_bit_flip_falsification.v1"
)


def build_array_bit_flip_falsification(
    *,
    component_id: str,
    official_output: np.ndarray,
    portable_output: np.ndarray,
    tolerance: float,
    output_name: str = "output",
    false_authority: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    tolerance_value = _nonnegative_tolerance_or_none(tolerance)
    official = np.ascontiguousarray(np.asarray(official_output, dtype="<f8"))
    portable = np.ascontiguousarray(np.asarray(portable_output, dtype="<f8"))
    perturbed, byte_offset, bit_mask = _single_bit_flip_perturbation(
        portable,
        reference=official,
        tolerance=tolerance,
    )
    official_hash = _hash_array(official)
    portable_hash = _hash_array(portable)
    perturbed_hash = _hash_array(perturbed)
    negative_control_max_abs_error = _max_abs_error_or_none(official, perturbed)
    falsifies_when_perturbed = bool(
        byte_offset is not None
        and bit_mask is not None
        and negative_control_max_abs_error is not None
        and tolerance_value is not None
        and negative_control_max_abs_error > tolerance_value
        and official_hash == portable_hash
        and perturbed_hash != official_hash
    )
    return {
        "schema": BIT_FLIP_FALSIFICATION_SCHEMA,
        "component_id": component_id,
        "mode": "single_bit_flip_on_portable_output",
        "output_name": output_name,
        "bit_flip_byte_offset": byte_offset,
        "bit_flip_mask": bit_mask,
        "tolerance": tolerance_value,
        "baseline_official_output_sha256": official_hash,
        "baseline_portable_output_sha256": portable_hash,
        "perturbed_portable_output_sha256": perturbed_hash,
        "negative_control_max_abs_error": negative_control_max_abs_error,
        "negative_control_output_hashes_bit_identical": perturbed_hash
        == official_hash,
        "falsifies_when_perturbed": falsifies_when_perturbed,
        "passed": falsifies_when_perturbed,
        **dict(false_authority or {}),
    }


def build_named_arrays_bit_flip_falsification(
    *,
    component_id: str,
    official_outputs: Mapping[str, np.ndarray],
    portable_outputs: Mapping[str, np.ndarray],
    tolerance: float,
    false_authority: Mapping[str, bool] | None = None,
) -> dict[str, Any]:
    tolerance_value = _nonnegative_tolerance_or_none(tolerance)
    official_hash = _hash_named_arrays(official_outputs)
    portable_hash = _hash_named_arrays(portable_outputs)
    names = sorted(set(official_outputs) & set(portable_outputs))
    proof: dict[str, Any] | None = None
    for name in names:
        candidate_proof = build_array_bit_flip_falsification(
            component_id=component_id,
            official_output=official_outputs[name],
            portable_output=portable_outputs[name],
            tolerance=tolerance,
            output_name=name,
            false_authority=false_authority,
        )
        if candidate_proof["passed"] is True:
            proof = candidate_proof
            break
    if proof is None:
        proof = {
            "schema": BIT_FLIP_FALSIFICATION_SCHEMA,
            "component_id": component_id,
            "mode": "single_bit_flip_on_portable_named_output",
            "output_name": None,
            "bit_flip_byte_offset": None,
            "bit_flip_mask": None,
            "tolerance": tolerance_value,
            "baseline_official_output_sha256": official_hash,
            "baseline_portable_output_sha256": portable_hash,
            "perturbed_portable_output_sha256": portable_hash,
            "negative_control_max_abs_error": None,
            "negative_control_output_hashes_bit_identical": True,
            "falsifies_when_perturbed": False,
            "passed": False,
            **dict(false_authority or {}),
        }

    perturbed_outputs = {
        name: np.ascontiguousarray(np.asarray(array, dtype="<f8"))
        for name, array in portable_outputs.items()
    }
    output_name = proof.get("output_name")
    if isinstance(output_name, str) and output_name in perturbed_outputs:
        perturbed_outputs[output_name], _, _ = _single_bit_flip_perturbation(
            perturbed_outputs[output_name],
            reference=np.asarray(official_outputs[output_name], dtype="<f8"),
            tolerance=tolerance,
        )
    perturbed_named_hash = _hash_named_arrays(perturbed_outputs)
    negative_control_hashes_bit_identical = perturbed_named_hash == official_hash
    passed = bool(
        proof.get("passed") is True
        and tolerance_value is not None
        and official_hash == portable_hash
        and not negative_control_hashes_bit_identical
    )
    return {
        **proof,
        "mode": "single_bit_flip_on_portable_named_output",
        "baseline_official_output_sha256": official_hash,
        "baseline_portable_output_sha256": portable_hash,
        "perturbed_portable_output_sha256": perturbed_named_hash,
        "negative_control_output_hashes_bit_identical": negative_control_hashes_bit_identical,
        "falsifies_when_perturbed": passed,
        "passed": passed,
        **dict(false_authority or {}),
    }


def _single_bit_flip_perturbation(
    array: np.ndarray,
    *,
    reference: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, int | None, int | None]:
    candidate = np.ascontiguousarray(np.asarray(array, dtype="<f8")).copy()
    tolerance_bound = _tolerance_bound(tolerance)
    if candidate.size == 0:
        return candidate, None, None
    if not np.isfinite(tolerance_bound):
        return candidate, None, None
    reference_array = np.ascontiguousarray(np.asarray(reference, dtype="<f8"))
    raw = candidate.view(np.uint8).reshape(-1)
    for byte_offset in range(raw.size):
        original = int(raw[byte_offset])
        for bit_mask in (1, 2, 4, 8, 16, 32, 64, 128):
            raw[byte_offset] = original ^ bit_mask
            max_abs_error = _max_abs_error_or_none(reference_array, candidate)
            if (
                max_abs_error is not None
                and max_abs_error > tolerance_bound
                and _hash_array(reference_array) != _hash_array(candidate)
            ):
                return candidate.copy(), byte_offset, bit_mask
            raw[byte_offset] = original
    return candidate, None, None


def _nonnegative_tolerance_or_none(value: float) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) and out >= 0.0 else None


def _tolerance_bound(value: float) -> float:
    tolerance = _nonnegative_tolerance_or_none(value)
    return float("inf") if tolerance is None else tolerance


def _max_abs_error_or_none(left: np.ndarray, right: np.ndarray) -> float | None:
    left_array = np.asarray(left, dtype="<f8")
    right_array = np.asarray(right, dtype="<f8")
    if left_array.shape != right_array.shape or left_array.size == 0:
        return None
    delta = np.abs(left_array - right_array)
    if not np.all(np.isfinite(delta)):
        return None
    return float(np.max(delta))


def _hash_named_arrays(arrays: Mapping[str, np.ndarray]) -> str:
    h = sha256()
    for name in sorted(arrays):
        arr = np.ascontiguousarray(np.asarray(arrays[name], dtype="<f8"))
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        h.update(json.dumps(list(arr.shape), sort_keys=True).encode("utf-8"))
        h.update(b"\0")
        h.update(arr.tobytes())
        h.update(b"\0")
    return h.hexdigest()


def _hash_array(array: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(array, dtype="<f8"))
    return sha256(arr.tobytes()).hexdigest()


__all__ = [
    "BIT_FLIP_FALSIFICATION_SCHEMA",
    "build_array_bit_flip_falsification",
    "build_named_arrays_bit_flip_falsification",
]
