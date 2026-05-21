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
import json
from pathlib import Path
from typing import Any

import numpy as np

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX

__all__ = [
    "export_mlx_state_dict_to_torch_pt",
    "load_pytorch_state_dict_from_pt",
    "build_export_manifest",
]


def export_mlx_state_dict_to_torch_pt(
    state_dict_np: dict[str, np.ndarray],
    output_pt_path: Path | str,
    *,
    substrate_id: str,
    run_id: str,
    overwrite: bool = False,
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

    # Convert numpy arrays -> torch tensors (canonical PyTorch layout preserved).
    torch_state: dict[str, torch.Tensor] = {}
    per_tensor_sha: dict[str, str] = {}
    for name, arr in state_dict_np.items():
        if not isinstance(arr, np.ndarray):
            raise TypeError(f"value for {name!r} is not numpy array: got {type(arr).__name__}")
        # Force fp32 for canonical contest-runtime path (Selfcomp + sister
        # substrates use fp32 weights at inflate time).
        arr_f32 = arr.astype(np.float32)
        torch_state[name] = torch.from_numpy(arr_f32.copy())
        per_tensor_sha[name] = hashlib.sha256(arr_f32.tobytes()).hexdigest()[:16]

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
        file_sha256=file_sha,
        file_size_bytes=file_size,
        tensor_count=len(state_dict_np),
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
        # Canonical Provenance markers per Catalog #287/#323/#192/#1.
        # The exported weights themselves were generated under MLX; routing
        # to CUDA T4 for eval is the promotion path (separate event).
        "training_evidence_grade": EVIDENCE_GRADE_MLX,
        "training_evidence_tag": EVIDENCE_TAG_MLX,
        "score_claim": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": [
            "macos_mlx_research_signal_training_axis_only",
            "requires_paired_cuda_t4_or_linux_x86_64_eval_for_promotion",
            "export_step_does_not_change_evidence_grade",
        ],
        "operator_routable_next_step": (
            "Stage exported .pt + sister archive grammar; fire "
            "tools/operator_authorize.py --recipe <recipe> --target modal "
            "for [contest-CUDA] T4 eval via experiments/contest_auth_eval.py"
        ),
    }
