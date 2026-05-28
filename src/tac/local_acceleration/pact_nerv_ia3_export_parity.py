# SPDX-License-Identifier: MIT
"""Pact-NeRV-IA3 MLX-to-PyTorch forward parity proof."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from tac.local_acceleration.mlx_to_pytorch_export import (
    export_mlx_state_dict_to_torch_pt,
)
from tac.optimization.dqs1_materializer_feedback_bridge import FALSE_AUTHORITY
from tac.optimization.proxy_candidate_contract import require_no_truthy_authority_fields

PACT_NERV_IA3_MLX_PYTORCH_FORWARD_PARITY_SCHEMA = (
    "pact_nerv_ia3_mlx_pytorch_forward_parity.v1"
)


class PactNervIa3ExportParityError(ValueError):
    """Raised when the Pact-NeRV-IA3 export parity proof cannot run."""


def _state_dict_sha256(state_dict: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        arr = np.ascontiguousarray(state_dict[name])
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(arr.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(tuple(arr.shape)).encode("ascii"))
        digest.update(b"\0")
        digest.update(arr.tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _smoke_config_kwargs(
    *,
    num_pairs: int,
    output_height: int,
    output_width: int,
) -> dict[str, Any]:
    return {
        "latent_dim": 8,
        "embed_dim": 24,
        "initial_grid_h": 3,
        "initial_grid_w": 4,
        "decoder_channels": (20, 16, 12),
        "sin_frequency": 30.0,
        "num_upsample_blocks": 3,
        "pose_dim": 6,
        "ia3_init_delta_std": 0.01,
        "num_pairs": num_pairs,
        "output_height": output_height,
        "output_width": output_width,
    }


def prove_pact_nerv_ia3_mlx_pytorch_forward_parity(
    *,
    pair_indices: list[int],
    output_pt_path: str | Path | None = None,
    seed: int = 0,
    tolerance: float = 1e-6,
    output_height: int = 24,
    output_width: int = 32,
    overwrite_pt: bool = False,
) -> dict[str, Any]:
    """Export an MLX IA3 renderer and compare PyTorch forward output exactly."""

    if not pair_indices:
        raise PactNervIa3ExportParityError("pair_indices must be non-empty")
    if any(index < 0 for index in pair_indices):
        raise PactNervIa3ExportParityError("pair_indices must be non-negative")
    if tolerance < 0:
        raise PactNervIa3ExportParityError("tolerance must be non-negative")

    try:
        import mlx.core as mx
        import torch
    except Exception as exc:  # pragma: no cover - host dependent.
        raise PactNervIa3ExportParityError(f"MLX/PyTorch import failed: {exc}") from exc

    from tac.substrates.pact_nerv_ia3.architecture import (
        PactNervIa3Config,
        PactNervIa3Substrate,
    )
    from tac.substrates.pact_nerv_ia3.mlx_renderer import PactNervIa3SubstrateMLX

    max_index = max(pair_indices)
    cfg = PactNervIa3Config(
        **_smoke_config_kwargs(
            num_pairs=max_index + 1,
            output_height=output_height,
            output_width=output_width,
        )
    )
    mx.random.seed(seed)
    torch.manual_seed(seed)
    mlx_model = PactNervIa3SubstrateMLX(cfg)
    state_np = mlx_model.export_state_dict()
    torch_model = PactNervIa3Substrate(cfg).eval()
    torch_state = {name: torch.from_numpy(value) for name, value in state_np.items()}
    torch_model.load_state_dict(torch_state, strict=True)

    pair_indices_np = np.array(pair_indices, dtype=np.int64)
    mlx_out = np.asarray(mlx_model(mx.array(pair_indices_np)), dtype=np.float32)
    with torch.no_grad():
        rgb_0, rgb_1 = torch_model(torch.tensor(pair_indices_np, dtype=torch.long))
    torch_out = (
        torch.stack([rgb_0, rgb_1], dim=1)
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32)
        * 255.0
    )
    diff = np.abs(mlx_out - torch_out)
    max_abs = float(diff.max(initial=0.0))
    mean_abs = float(diff.mean()) if diff.size else 0.0
    parity_passed = max_abs <= tolerance

    raw_state_bytes = int(sum(value.nbytes for value in state_np.values()))
    export_manifest: dict[str, Any] | None = None
    if output_pt_path is not None:
        export_manifest = export_mlx_state_dict_to_torch_pt(
            state_np,
            output_pt_path,
            substrate_id="pact_nerv_ia3",
            run_id=f"pact_nerv_ia3_mlx_pytorch_parity_seed{seed}",
            overwrite=overwrite_pt,
        )
    blockers = [
        "macos_mlx_research_signal_has_no_score_authority",
        "requires_archive_native_pack_and_receiver_proof_before_exact_dispatch",
        "contest_cpu_or_cuda_auth_eval_required_before_score_claim",
    ]
    if not parity_passed:
        blockers.insert(0, "mlx_pytorch_forward_parity_failed")
    payload: dict[str, Any] = {
        "schema": PACT_NERV_IA3_MLX_PYTORCH_FORWARD_PARITY_SCHEMA,
        "axis": "[macOS-MLX research-signal]",
        "config": _smoke_config_kwargs(
            num_pairs=cfg.num_pairs,
            output_height=cfg.output_height,
            output_width=cfg.output_width,
        ),
        "seed": seed,
        "pair_indices": pair_indices,
        "tolerance": tolerance,
        "parity_passed": parity_passed,
        "max_abs_diff_255": max_abs,
        "mean_abs_diff_255": mean_abs,
        "mlx_output_shape": list(mlx_out.shape),
        "pytorch_output_shape": list(torch_out.shape),
        "state_dict": {
            "tensor_count": len(state_np),
            "raw_state_bytes": raw_state_bytes,
            "sha256": _state_dict_sha256(state_np),
        },
        "byte_tax": {
            "raw_state_bytes": raw_state_bytes,
            "torch_pt_file_size_bytes": (
                export_manifest.get("file_size_bytes") if export_manifest else None
            ),
            "archive_native_pack_required": True,
            "archive_grammar": "pact_nerv_ia3 PIA3 monolithic 0.bin pack_archive",
        },
        "export_manifest": export_manifest,
        "score_claim": False,
        "promotion_eligible": False,
        "rank_or_kill_eligible": False,
        "promotable": False,
        "ready_for_exact_eval_dispatch": False,
        "blockers": blockers,
        **FALSE_AUTHORITY,
    }
    require_no_truthy_authority_fields(payload, context="pact_nerv_ia3_export_parity")
    return payload


__all__ = [
    "PACT_NERV_IA3_MLX_PYTORCH_FORWARD_PARITY_SCHEMA",
    "PactNervIa3ExportParityError",
    "prove_pact_nerv_ia3_mlx_pytorch_forward_parity",
]
