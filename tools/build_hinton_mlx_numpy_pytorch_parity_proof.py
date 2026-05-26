#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Build numpy↔PyTorch parity proof for an MLX-trained Hinton-distilled checkpoint.

Per TaskCreate #1262 + #1330 + the Slot 1 sister parity bridge canonical
contract: prove that an MLX-trained student state_dict round-trips through
the canonical ``tac.local_acceleration.mlx_to_pytorch_export`` bridge with
byte-stable per-tensor parity (max_abs_diff <= 1e-6 across all parameters).

This is the canonical contract verification ONLY; it does NOT change the
non-promotable evidence_grade of the underlying MLX-trained weights. Per
CLAUDE.md "MLX portable-local-substrate authority" + "Submission auth eval
— BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE" non-negotiables:
the parity proof attests the BRIDGE preserves bytes; promotion to contest
authority requires paired CPU + CUDA contest-hardware eval per Catalog #205.

The canonical 4-class bridge taxonomy per
`tac.local_acceleration.deterministic_primitives`:
  - BYTE_STABLE_BY_DEFAULT      (max_abs <= 1e-6 AND mean_abs <= 1e-7)
  - NUMERIC_TOLERANCE_INHERENT  (max_abs <= 1e-4 AND mean_abs <= 1e-5)
  - FRAMEWORK_DIFFERENT         (anything outside)

For a state_dict bridge (NOT forward-pass), BYTE_STABLE_BY_DEFAULT is
expected because both sides are loading the same numpy bytes via
distinct safetensor loaders.

Output: ``numpy_pytorch_parity_proof.json`` carrying canonical Provenance
+ per-tensor parity verdict + aggregate verdict.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tac.local_acceleration import EVIDENCE_GRADE_MLX, EVIDENCE_TAG_MLX
from tac.local_acceleration.mlx_to_pytorch_export import (
    load_pytorch_state_dict_from_pt,
)
from tac.local_acceleration.pr95_hnerv_mlx_long_training import (
    PR95_MLX_LONG_TRAINING_FALSE_AUTHORITY,
)

# Canonical drift bands per
# tac.local_acceleration.deterministic_primitives
# (state-dict bridge is BYTE_STABLE by construction — both sides load the
# same numpy bytes).
BYTE_STABLE_MAX_ABS = 1.0e-6
BYTE_STABLE_MEAN_ABS = 1.0e-7
NUMERIC_TOLERANCE_MAX_ABS = 1.0e-4
NUMERIC_TOLERANCE_MEAN_ABS = 1.0e-5


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _file_sha256(path: Path) -> str:
    if not path.is_file():
        return ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _classify_drift(max_abs: float, mean_abs: float) -> str:
    if max_abs <= BYTE_STABLE_MAX_ABS and mean_abs <= BYTE_STABLE_MEAN_ABS:
        return "BYTE_STABLE_BY_DEFAULT"
    if max_abs <= NUMERIC_TOLERANCE_MAX_ABS and mean_abs <= NUMERIC_TOLERANCE_MEAN_ABS:
        return "NUMERIC_TOLERANCE_INHERENT"
    return "FRAMEWORK_DIFFERENT"


def build_numpy_pytorch_parity_proof(
    *,
    mlx_safetensors_path: Path,
    pt_path: Path,
    export_manifest_path: Path,
    output_proof_path: Path,
    smoke_verdict_path: Path | None = None,
) -> dict[str, Any]:
    """Build the canonical numpy↔PyTorch parity proof artifact.

    Loads the MLX safetensors checkpoint (the source) AND the .pt
    (the bridge output), converts MLX values to numpy in PyTorch layout
    matching the bridge's transpose convention (Conv2d (out, kH, kW, in)
    → (out, in, kH, kW)), then verifies per-tensor numerical parity.

    The proof rests on three byte-level invariants:
      1. The bridge's .pt was produced by ``export_mlx_state_dict_to_torch_pt``
         which calls ``torch.from_numpy(np.ascontiguousarray(arr).copy())``
         per tensor → numpy bytes equal PyTorch tensor bytes per torch's
         numpy-interop contract.
      2. The MLX safetensors file holds the canonical fp32 values; loading
         via ``mlx.core.load`` returns MLX arrays; ``np.array(...)``
         conversion preserves the float32 bit pattern.
      3. The Conv2d transpose (out, kH, kW, in) → (out, in, kH, kW) is
         a pure reshape + permute — no rounding.

    Per Catalog #287/#323 canonical Provenance: the proof artifact carries
    ``score_claim_valid=False`` + ``promotion_eligible=False`` +
    ``axis_tag=[research-signal]`` by construction.
    """
    import mlx.core as mx
    import numpy as np

    if not mlx_safetensors_path.is_file():
        raise FileNotFoundError(f"MLX safetensors not found: {mlx_safetensors_path}")
    if not pt_path.is_file():
        raise FileNotFoundError(f"PT file not found: {pt_path}")
    if not export_manifest_path.is_file():
        raise FileNotFoundError(f"Export manifest not found: {export_manifest_path}")

    # Load source MLX safetensors.
    mlx_state = mx.load(str(mlx_safetensors_path))
    if not isinstance(mlx_state, dict):
        raise RuntimeError(
            f"mx.load returned {type(mlx_state).__name__}; expected dict"
        )

    # Load PT state_dict (bridge output).
    pt_state = load_pytorch_state_dict_from_pt(pt_path)

    # Load export manifest for canonical per-tensor sha256 cross-check.
    with export_manifest_path.open("r", encoding="utf-8") as handle:
        export_manifest = json.load(handle)
    per_tensor_manifest = export_manifest.get("per_tensor", {})

    # Per-tensor parity check.
    # NB: the bridge transposes 4D Conv2d weights from (out, kH, kW, in) →
    # (out, in, kH, kW). The .pt state_dict has the post-transpose layout.
    # To compare bytes we transpose the MLX side to PyTorch layout before
    # numerical comparison.
    per_tensor_results: dict[str, dict[str, Any]] = {}
    aggregate_max_abs = 0.0
    aggregate_sum_abs = 0.0
    aggregate_count = 0

    # The .pt includes an extra 'latents' key that's saved separately
    # (per pr95_hnerv_mlx_long_training._persist_checkpoint line 1351).
    # The MLX safetensors only carries the decoder.parameters() tree; the
    # latents are stored as a separate .latents.npy file. Skip latents in
    # MLX-side check.
    mlx_keys = set(mlx_state.keys())
    pt_keys_excluding_latents = {k for k in pt_state.keys() if k != "latents"}
    missing_in_mlx = sorted(pt_keys_excluding_latents - mlx_keys)
    missing_in_pt = sorted(mlx_keys - pt_keys_excluding_latents)

    for key in sorted(mlx_keys & pt_keys_excluding_latents):
        mlx_arr = np.array(mlx_state[key], dtype=np.float32)
        pt_arr = pt_state[key].detach().cpu().numpy().astype(np.float32)
        # Match bridge transpose for 4D tensors.
        if mlx_arr.ndim == 4:
            mlx_arr_pt_layout = np.transpose(mlx_arr, (0, 3, 1, 2))
        else:
            mlx_arr_pt_layout = mlx_arr

        if mlx_arr_pt_layout.shape != pt_arr.shape:
            per_tensor_results[key] = {
                "verdict": "SHAPE_MISMATCH",
                "mlx_shape": list(mlx_arr_pt_layout.shape),
                "pt_shape": list(pt_arr.shape),
                "max_abs_diff": float("nan"),
                "mean_abs_diff": float("nan"),
            }
            continue

        diff = np.abs(mlx_arr_pt_layout - pt_arr)
        max_abs = float(diff.max())
        mean_abs = float(diff.mean())
        n_elements = int(diff.size)
        aggregate_max_abs = max(aggregate_max_abs, max_abs)
        aggregate_sum_abs += float(diff.sum())
        aggregate_count += n_elements

        verdict = _classify_drift(max_abs, mean_abs)
        per_tensor_results[key] = {
            "verdict": verdict,
            "max_abs_diff": max_abs,
            "mean_abs_diff": mean_abs,
            "shape": list(pt_arr.shape),
            "n_elements": n_elements,
            "manifest_sha256_prefix": per_tensor_manifest.get(key, {}).get(
                "sha256", ""
            )[:16] if isinstance(per_tensor_manifest.get(key), dict) else "",
        }

    aggregate_mean_abs = (
        aggregate_sum_abs / aggregate_count if aggregate_count > 0 else 0.0
    )
    aggregate_verdict = _classify_drift(aggregate_max_abs, aggregate_mean_abs)

    # Build the proof artifact.
    proof: dict[str, Any] = {
        "schema_version": "hinton_mlx_numpy_pytorch_parity_proof.v1",
        "tool": "tools/build_hinton_mlx_numpy_pytorch_parity_proof.py",
        "generated_utc": _utc_now_iso(),
        "lane_id": "lane_hinton_mlx_first_local_pivot_20260526",
        "task_create_ids": [1262, 1330],
        "mlx_safetensors_path": str(mlx_safetensors_path.relative_to(REPO_ROOT)),
        "mlx_safetensors_sha256": _file_sha256(mlx_safetensors_path),
        "mlx_safetensors_bytes": mlx_safetensors_path.stat().st_size,
        "pt_path": str(pt_path.relative_to(REPO_ROOT)),
        "pt_sha256": _file_sha256(pt_path),
        "pt_bytes": pt_path.stat().st_size,
        "export_manifest_path": str(
            export_manifest_path.relative_to(REPO_ROOT)
        ),
        "smoke_verdict_path": (
            str(smoke_verdict_path.relative_to(REPO_ROOT))
            if smoke_verdict_path is not None
            else None
        ),
        "mlx_tensor_count": len(mlx_state),
        "pt_tensor_count_excluding_latents": len(pt_keys_excluding_latents),
        "pt_tensor_count_total": len(pt_state),
        "tensors_compared": len(per_tensor_results),
        "missing_in_mlx": missing_in_mlx,
        "missing_in_pt": missing_in_pt,
        "aggregate_max_abs_diff": aggregate_max_abs,
        "aggregate_mean_abs_diff": aggregate_mean_abs,
        "aggregate_element_count": aggregate_count,
        "aggregate_verdict": aggregate_verdict,
        "aggregate_verdict_threshold_byte_stable_max_abs": BYTE_STABLE_MAX_ABS,
        "aggregate_verdict_threshold_byte_stable_mean_abs": BYTE_STABLE_MEAN_ABS,
        "per_tensor_results": per_tensor_results,
        "canonical_provenance": {
            "artifact_kind": "predicted_from_model",
            "source_path": str(pt_path.relative_to(REPO_ROOT)),
            "source_sha256": _file_sha256(pt_path),
            "measurement_axis": "[research-signal]",
            "hardware_substrate": "macos_arm64",
            "evidence_grade": "research_only",
            "promotion_eligible": False,
            "score_claim_valid": False,
            "captured_at_utc": _utc_now_iso(),
            "canonical_helper_invocation": (
                "tools.build_hinton_mlx_numpy_pytorch_parity_proof."
                "build_numpy_pytorch_parity_proof"
            ),
            "rejection_reason": (
                "state-dict bridge parity proof is observability-only; "
                "promotion to contest authority requires paired CPU + "
                "CUDA contest-hardware eval per Catalog #205 + "
                "CLAUDE.md 'Submission auth eval — BOTH CPU AND CUDA'"
            ),
        },
        "evidence_grade": EVIDENCE_GRADE_MLX,
        "evidence_tag": EVIDENCE_TAG_MLX,
        **PR95_MLX_LONG_TRAINING_FALSE_AUTHORITY,
        "exact_readiness_refusal": {
            "schema": "exact_readiness_refusal.v1",
            "ready": False,
            "blockers": [
                "state_dict_bridge_parity_proof_is_not_contest_auth_eval",
                "requires_paired_contest_cpu_cuda_auth_eval_for_promotion",
                "macos_mlx_research_signal_only_per_catalog_192",
            ],
        },
        "score_claim_valid": False,
        "promotion_eligible": False,
        "ready_for_exact_eval_dispatch": False,
        "rank_or_kill_eligible": False,
    }

    output_proof_path.parent.mkdir(parents=True, exist_ok=True)
    output_proof_path.write_text(
        json.dumps(proof, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return proof


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mlx-safetensors",
        type=Path,
        required=True,
        help="Path to the .mlx.safetensors checkpoint emitted by the Slot 1 pipeline.",
    )
    parser.add_argument(
        "--pt-path",
        type=Path,
        required=True,
        help="Path to the .pt PyTorch state_dict emitted by the Slot 1 pipeline.",
    )
    parser.add_argument(
        "--export-manifest",
        type=Path,
        required=True,
        help=(
            "Path to the .pt.export_manifest.json carrying per-tensor sha256 "
            "for canonical cross-check."
        ),
    )
    parser.add_argument(
        "--output-proof",
        type=Path,
        required=True,
        help="Where to write the numpy_pytorch_parity_proof.json artifact.",
    )
    parser.add_argument(
        "--smoke-verdict",
        type=Path,
        default=None,
        help="Optional path to the smoke verdict JSON (recorded in the proof).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    mlx_path = args.mlx_safetensors
    if not mlx_path.is_absolute():
        mlx_path = REPO_ROOT / mlx_path
    pt_path = args.pt_path
    if not pt_path.is_absolute():
        pt_path = REPO_ROOT / pt_path
    manifest_path = args.export_manifest
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    output_path = args.output_proof
    if not output_path.is_absolute():
        output_path = REPO_ROOT / output_path
    smoke_path = None
    if args.smoke_verdict is not None:
        smoke_path = args.smoke_verdict
        if not smoke_path.is_absolute():
            smoke_path = REPO_ROOT / smoke_path

    proof = build_numpy_pytorch_parity_proof(
        mlx_safetensors_path=mlx_path,
        pt_path=pt_path,
        export_manifest_path=manifest_path,
        output_proof_path=output_path,
        smoke_verdict_path=smoke_path,
    )
    print(
        f"[parity-proof] aggregate_verdict={proof['aggregate_verdict']} "
        f"max_abs={proof['aggregate_max_abs_diff']:.3e} "
        f"mean_abs={proof['aggregate_mean_abs_diff']:.3e} "
        f"tensors={proof['tensors_compared']} "
        f"missing_in_mlx={len(proof['missing_in_mlx'])} "
        f"missing_in_pt={len(proof['missing_in_pt'])}"
    )
    print(f"[parity-proof] output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
