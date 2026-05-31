# SPDX-License-Identifier: MIT
"""Canonical MLX-to-PyTorch weight export pipeline.

OVERNIGHT-WW Phase 4 per operator directive 2026-05-21 *"experimenting with
cuda t4 with eval against weights and substrates trained using MLX; is they
possible?"*

This module implements the canonical "train anywhere, eval anywhere" export
path:

    MLX-trained state_dict (numpy intermediary)
      |
      v   :func:`export_mlx_state_dict_to_torch_pt`
      v
    .pt file (PyTorch state_dict; canonical layout)
      |
      v   :func:`experiments/contest_auth_eval.py`
      v
    CUDA T4 / A100 contest-axis [contest-CUDA] score

The canonical contract:
- MLX-native substrate produces ``state_dict`` via numpy (its
  ``export_state_dict()`` method returns numpy arrays in PyTorch layout
  per the portable primitives convention)
- This module serializes via ``torch.save(...)`` so canonical PyTorch
  inflate / eval helpers consume the file unchanged
- The contest-axis evaluator runs against the saved .pt file via the
  canonical PyTorch substrate architecture — MLX numerics never reach
  the contest scorer

Per CLAUDE.md non-negotiables PRESERVED:
- **Submission auth eval — BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT
  HARDWARE**: this pipeline targets that gate explicitly — MLX is the
  training surface; CUDA T4 (or paired Linux x86_64) is the eval surface.
- **Catalog #1 + #192 + #317**: MLX-trained weights are non-promotable
  until evaluated via PyTorch on CUDA T4; the export step does NOT change
  the non-promotable evidence_grade — that's a property of the evidence
  (where it was generated), not of the byte format.
- **Catalog #110/#113 APPEND-ONLY**: export writes NEW .pt files; never
  mutates existing forensic artifacts.

Operator-routable next step after export:
    1. Operator stages exported .pt + sister archive grammar (per
       ``tac.substrates.grayscale_lut.archive.pack_archive``)
    2. Operator fires ``tools/operator_authorize.py --recipe <recipe>
       --target modal`` to dispatch eval on CUDA T4
    3. Canonical Modal call_id ledger per Catalog #245 records the
       dispatch + outcome
    4. Contest-axis [contest-CUDA] score lands; canonical Provenance per
       Catalog #287/#323 carries axis + hardware + archive sha
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX

__all__ = [
    "MLX_BRIDGE_FALSE_AUTHORITY_BLOCKERS",
    "assign_mlx_param_by_dotted_name",
    "build_export_manifest",
    "build_mlx_pytorch_forward_parity_proof",
    "build_substrate_bridge_manifest",
    "export_mlx_state_dict_to_torch_pt",
    "hash_file_sha256",
    "load_pytorch_state_dict_from_pt",
    "transpose_mlx_hwio_state_dict_to_pytorch_oihw",
]


# Canonical false-authority blockers per Catalog #287/#323/#192/#1/#317/#341.
# Lifted verbatim from the 6 substrate bridge tools to extinct duplication.
MLX_BRIDGE_FALSE_AUTHORITY_BLOCKERS: tuple[str, ...] = (
    "predicted_axis_not_contest_authority_per_catalog_127_192_317_341",
    "requires_paired_contest_cpu_plus_cuda_for_score_claim",
    "bridge_export_step_does_not_change_evidence_grade",
)


def hash_file_sha256(path: Path | str) -> str:
    """Return the SHA-256 digest for a file without duplicating bridge logic."""

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assign_mlx_param_by_dotted_name(
    mlx_model: Any,
    dotted_name: str,
    np_arr: Any,
    *,
    leaf_name_overrides: Mapping[str, str] | None = None,
) -> None:
    """Assign an MLX parameter or buffer by dotted state-dict name.

    The bridge tools all need the same dotted walk when replaying a numpy
    state_dict into an MLX model for parity.  VQ-style private buffers can pass
    full-name overrides such as ``{"quantizer.codebook": "_codebook"}``.
    """

    from tac.framework_agnostic import require_mlx_core

    mx = require_mlx_core()
    overrides = dict(leaf_name_overrides or {})
    parts = str(dotted_name).split(".")
    if not parts or any(not part for part in parts):
        raise ValueError(f"dotted_name must be non-empty dotted path: {dotted_name!r}")

    obj: Any = mlx_model
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    leaf = parts[-1]
    target_leaf = overrides.get(dotted_name) or overrides.get(leaf) or leaf
    new_val = mx.array(np_arr)
    if target_leaf.startswith("_"):
        setattr(obj, target_leaf, new_val)
        return
    if hasattr(obj, "update"):
        try:
            obj.update({target_leaf: new_val})
            return
        except Exception:
            pass
    setattr(obj, target_leaf, new_val)


def transpose_mlx_hwio_state_dict_to_pytorch_oihw(
    mlx_state_dict_numpy: Mapping[str, Any],
    *,
    skip_buffer_name_predicate: Any = None,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    """Canonical MLX-HWIO to PyTorch-OIHW state-dict conversion wrapper."""

    from tac.framework_agnostic.helpers import convert_mlx_state_dict_to_pytorch_oihw

    return convert_mlx_state_dict_to_pytorch_oihw(
        mlx_state_dict_numpy,
        skip_buffer_name_predicate=skip_buffer_name_predicate,
    )


def build_mlx_pytorch_forward_parity_proof(
    *,
    mlx_output: Any,
    pytorch_output: Any,
    sample_pair_indices: Sequence[int] = (),
    atol_max_01: float = 0.001,
    atol_mean_01: float = 1e-4,
    output_space: str = "sigmoid_0_to_1",
    backends_compared: str = "mlx_vs_pytorch_forward",
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fail-closed MLX/PyTorch forward-parity proof payload."""

    mlx_arr = np.asarray(mlx_output, dtype=np.float32)
    pytorch_arr = np.asarray(pytorch_output, dtype=np.float32)
    blockers = list(MLX_BRIDGE_FALSE_AUTHORITY_BLOCKERS)
    shape_match = mlx_arr.shape == pytorch_arr.shape
    if shape_match:
        drift = np.abs(mlx_arr - pytorch_arr)
        max_abs = float(drift.max(initial=0.0))
        mean_abs = float(drift.mean()) if drift.size else 0.0
        within_band = bool(max_abs <= atol_max_01 and mean_abs <= atol_mean_01)
    else:
        max_abs = None
        mean_abs = None
        within_band = False
        blockers.insert(0, "mlx_pytorch_forward_shape_mismatch")
    if not within_band and shape_match:
        blockers.insert(0, "mlx_pytorch_forward_parity_failed")
    payload: dict[str, Any] = {
        "schema": "mlx_pytorch_forward_parity_proof.v1",
        "axis_tag": EVIDENCE_TAG_MLX,
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "sample_pair_indices": [int(index) for index in sample_pair_indices],
        "backends_compared": backends_compared,
        "decoder_output_space": output_space,
        "mlx_output_shape": list(mlx_arr.shape),
        "pytorch_output_shape": list(pytorch_arr.shape),
        "max_abs_drift_01": max_abs,
        "mean_abs_drift_01": mean_abs,
        "atol_max_01": float(atol_max_01),
        "atol_mean_01": float(atol_mean_01),
        "drift_within_band": within_band,
        "parity_passed": within_band,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers,
    }
    payload.update(dict(extra or {}))
    _reject_truthy_authority_fields(payload, context="mlx_pytorch_forward_parity_proof")
    return payload


def _reject_truthy_authority_fields(payload: Mapping[str, Any], *, context: str) -> None:
    for key in (
        "score_claim",
        "score_claim_valid",
        "score_claim_eligible",
        "promotion_eligible",
        "promotable",
        "rank_or_kill_eligible",
        "ready_for_exact_eval_dispatch",
        "exact_cuda_auth_eval",
        "contest_cuda_auth_eval",
    ):
        if payload.get(key) is True:
            raise ValueError(f"{context}: {key}=True forbidden for MLX bridge evidence")


def build_substrate_bridge_manifest(
    *,
    schema_version: str,
    tool: str,
    source_state_dict_path: Path | str,
    output_pytorch_state_dict: Path | str,
    source_state_dict_sha256: str,
    pytorch_state_dict_sha256: str,
    pytorch_state_dict_bytes: int,
    tensor_count: int,
    config: Mapping[str, Any] | None = None,
    per_tensor: Mapping[str, Any] | None = None,
    forward_parity: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    operator_routable_next_step: str | None = None,
    axis_tag: str = "[predicted]",
    evidence_grade: str = "predicted",
    generated_utc: str | None = None,
    extra_fields: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one canonical fail-closed manifest for MLX export bridge tools."""

    manifest: dict[str, Any] = {
        "schema_version": schema_version,
        "generated_utc": generated_utc or datetime.now(UTC).isoformat(),
        "tool": tool,
        "mlx_state_dict_path": str(source_state_dict_path),
        "mlx_state_dict_sha256": source_state_dict_sha256,
        "output_pytorch_state_dict": str(output_pytorch_state_dict),
        "pytorch_state_dict_sha256": pytorch_state_dict_sha256,
        "pytorch_state_dict_bytes": int(pytorch_state_dict_bytes),
        "tensor_count": int(tensor_count),
        "config": dict(config or {}),
        "per_tensor": dict(per_tensor or {}),
        "forward_parity": dict(forward_parity or {}),
        "axis_tag": axis_tag,
        "evidence_grade": evidence_grade,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "promotable": False,
        "blockers": list(MLX_BRIDGE_FALSE_AUTHORITY_BLOCKERS),
        "operator_routable_next_step": operator_routable_next_step,
        "provenance": dict(provenance or {}),
    }
    manifest.update(dict(extra_fields or {}))
    _reject_truthy_authority_fields(manifest, context=schema_version)
    return manifest


def export_mlx_state_dict_to_torch_pt(
    state_dict_np: dict[str, np.ndarray],
    output_pt_path: Path | str,
    *,
    substrate_id: str,
    run_id: str,
    overwrite: bool = False,
    force_float32_names: set[str] | frozenset[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    """Serialize an MLX-trained state_dict (numpy intermediary) as a PyTorch .pt file.

    Args:
        state_dict_np: dict mapping parameter name -> numpy array, in PyTorch
            layout (e.g. Conv2d weights as (out_channels, in_channels, kH, kW)).
            Typically produced by an MLX-native substrate's
            ``export_state_dict()`` method.
        output_pt_path: destination .pt file path.
        substrate_id: canonical substrate identifier (e.g. "grayscale_lut").
        run_id: canonical run identifier (e.g. UTC timestamp + smoke marker).
        overwrite: if False (default), raise FileExistsError when path exists.
        force_float32_names: optional explicit tensor-name allowlist to cast to
            float32. Dtypes are otherwise preserved exactly so integer indices,
            masks, quantized buffers, and codebook state are not silently
            corrupted.

    Returns:
        Export manifest dict with canonical Provenance fields (per Catalog
        #287/#323) + per-tensor sha256 + output path + size.
    """
    import torch

    out_path = Path(output_pt_path)
    if out_path.exists() and not overwrite:
        raise FileExistsError(f"output path exists; pass overwrite=True to replace: {out_path}")

    if not state_dict_np:
        raise ValueError("state_dict_np must contain at least one parameter")

    force_float32 = {str(name) for name in force_float32_names}
    unknown_force_names = sorted(force_float32 - set(state_dict_np))
    if unknown_force_names:
        raise KeyError(f"force_float32_names contains unknown tensors: {unknown_force_names}")

    # Convert numpy arrays -> torch tensors (canonical PyTorch layout preserved).
    torch_state: dict[str, torch.Tensor] = {}
    per_tensor_sha: dict[str, str] = {}
    per_tensor: dict[str, dict[str, Any]] = {}
    for name, arr in state_dict_np.items():
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"value for {name!r} is not numpy array: got {type(arr).__name__}")
        arr_out = arr.astype(np.float32) if name in force_float32 else arr
        tensor = torch.from_numpy(np.ascontiguousarray(arr_out).copy())
        torch_state[name] = tensor
        tensor_sha = hashlib.sha256(np.ascontiguousarray(arr_out).tobytes()).hexdigest()
        per_tensor_sha[name] = tensor_sha[:16]
        per_tensor[name] = {
            "source_dtype": str(arr.dtype),
            "export_dtype": str(arr_out.dtype),
            "shape": [int(x) for x in arr_out.shape],
            "sha256": tensor_sha,
            "forced_float32": name in force_float32,
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch_state, out_path)

    # Compute the file's sha256 for canonical custody.
    file_sha = hashlib.sha256(out_path.read_bytes()).hexdigest()
    file_size = out_path.stat().st_size

    return build_export_manifest(
        substrate_id=substrate_id,
        run_id=run_id,
        output_pt_path=str(out_path),
        per_tensor_sha=per_tensor_sha,
        per_tensor=per_tensor,
        file_sha256=file_sha,
        file_size_bytes=file_size,
        tensor_count=len(state_dict_np),
        force_float32_names=sorted(force_float32),
    )


def load_pytorch_state_dict_from_pt(
    pt_path: Path | str,
) -> dict[str, Any]:
    """Load a .pt file produced by :func:`export_mlx_state_dict_to_torch_pt`.

    Returns the raw PyTorch state_dict (caller routes through their canonical
    substrate architecture's ``load_state_dict()``).
    """
    import torch

    path = Path(pt_path)
    if not path.exists():
        raise FileNotFoundError(f"PT file not found: {path}")
    # weights_only=True for safety per Catalog #14 + #98 sister discipline.
    return torch.load(path, weights_only=True)


def build_export_manifest(
    *,
    substrate_id: str,
    run_id: str,
    output_pt_path: str,
    per_tensor_sha: dict[str, str],
    file_sha256: str,
    file_size_bytes: int,
    tensor_count: int,
    per_tensor: dict[str, dict[str, Any]] | None = None,
    force_float32_names: list[str] | None = None,
) -> dict[str, Any]:
    """Build canonical export manifest with non-promotable markers.

    Per Catalog #287/#323 canonical Provenance: every export carries the
    canonical (axis, hardware, evidence_grade) triple making the
    non-promotable nature explicit. The export STEP doesn't promote weights
    — only PyTorch CUDA T4 eval can do that per CLAUDE.md "Submission auth
    eval — BOTH CPU AND CUDA".
    """
    return {
        "schema_version": "mlx_to_pytorch_export.v1",
        "substrate_id": substrate_id,
        "run_id": run_id,
        "output_pt_path": output_pt_path,
        "file_sha256": file_sha256,
        "file_size_bytes": file_size_bytes,
        "tensor_count": tensor_count,
        "per_tensor_sha256_prefix": per_tensor_sha,
        "per_tensor": per_tensor or {},
        "dtype_policy": {
            "default": "preserve_numpy_dtype",
            "force_float32_names": force_float32_names or [],
        },
        # Canonical Provenance markers per Catalog #287/#323/#192/#1.
        # The exported weights themselves were generated under MLX; routing
        # to CUDA T4 for eval is the promotion path (separate event).
        "training_evidence_grade": EVIDENCE_GRADE_MLX,
        "training_evidence_tag": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "promotion_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": list(MLX_BRIDGE_FALSE_AUTHORITY_BLOCKERS),
        "operator_routable_next_step": (
            "Stage exported .pt + sister archive grammar; fire "
            "tools/operator_authorize.py --recipe <recipe> --target modal "
            "for [contest-CUDA] T4 eval via experiments/contest_auth_eval.py"
        ),
    }
