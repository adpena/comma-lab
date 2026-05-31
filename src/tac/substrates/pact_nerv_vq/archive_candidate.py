# SPDX-License-Identifier: MIT
"""Byte-closed PVQ archive export helpers for MLX/local training artifacts.

Mirrors the canonical sister at
:mod:`tac.substrates.pact_nerv_selector_v4.archive_candidate` per the 11th
INDIVIDUALLY-FRACTAL standing directive 2026-05-27 — PACT-NeRV-VQ's OWN
archive export pass, NOT a shared-helper shortcut from SELECTOR-V4.

The substrate-distinguishing primitive (van den Oord 2017 VQ-VAE codebook +
per-pair discrete index per arXiv:1711.00937 §3.1-3.2) is delegated to the
canonical :class:`tac.substrates.pact_nerv_vq.architecture.VectorQuantizerEMA`;
this module bridges the MLX-trained state_dict into the PVQ byte-closed
archive used by the inflate runtime.

Per CLAUDE.md non-negotiables PRESERVED:
- "Submission auth eval - BOTH CPU AND CUDA, ON 1:1 CONTEST-COMPLIANT HARDWARE":
  pack_archive_from_exported_state_dict produces the byte-closed archive that
  the L2 paired CUDA + Linux x86_64 CPU dispatch consumes.
- "MLX portable-local-substrate authority": helper is invoked from the
  PyTorch sister after the canonical MLX -> PyTorch bridge.
- Catalog #110/#113 APPEND-ONLY: this module is NEW; never mutates existing
  forensic artifacts.
- HNeRV parity L4: ~150 LOC + ≤2 ext deps (numpy + torch).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from tac.optimization.archive_bound_candidate_runtime_bridge import (
    emit_archive_bound_candidate_runtime_package,
)
from tac.repo_io import sha256_file
from tac.substrates._shared.pact_nerv_full_main import (
    build_archive_zip,
    write_contest_runtime,
)
from tac.substrates.pact_nerv_vq.archive import pack_archive

if TYPE_CHECKING:
    from tac.substrates.pact_nerv_vq.architecture import PactNervVqConfig

PACT_NERV_VQ_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA = (
    "pact_nerv_vq_mlx_archive_bound_adapter_package.v1"
)
PACT_NERV_VQ_MLX_RECEIVER_PROOF_SCHEMA = (
    "pact_nerv_vq_mlx_generated_receiver_proof.v1"
)
PACT_NERV_VQ_MLX_ARCHIVE_BOUND_ADAPTER_ID = "pact_nerv_vq_mlx_archive_export"
PACT_NERV_VQ_MLX_ARCHIVE_CANDIDATE_FAMILY = "pact_nerv_vq_mlx"
PACT_NERV_VQ_MLX_ARCHIVE_TRANSFORM_KIND = "pact_nerv_vq_mlx_archive"


def vq_meta_from_config(cfg: PactNervVqConfig) -> dict[str, object]:
    """Return the minimal receiver metadata needed to rebuild the decoder."""

    return {
        "embed_dim": int(cfg.embed_dim),
        "initial_grid_h": int(cfg.initial_grid_h),
        "initial_grid_w": int(cfg.initial_grid_w),
        "decoder_channels": [int(value) for value in cfg.decoder_channels],
        "sin_frequency": float(cfg.sin_frequency),
        "num_upsample_blocks": int(cfg.num_upsample_blocks),
        "output_height": int(cfg.output_height),
        "output_width": int(cfg.output_width),
        "codebook_decay": float(cfg.codebook_decay),
        "commitment_weight": float(cfg.commitment_weight),
    }


def _quantize_latents_via_codebook(
    latents: torch.Tensor, codebook: torch.Tensor
) -> torch.Tensor:
    """Compute nearest-codebook indices per van den Oord §3.1 distance metric.

    Args:
        latents: (num_pairs, latent_dim) per-pair encoder outputs.
        codebook: (codebook_size, latent_dim) learned VQ codebook.

    Returns:
        (num_pairs,) int64 codebook indices.
    """
    if latents.dim() != 2:
        raise ValueError(
            f"latents must be 2-D (num_pairs, latent_dim); got {tuple(latents.shape)}"
        )
    if codebook.dim() != 2:
        raise ValueError(
            f"codebook must be 2-D (codebook_size, latent_dim); got {tuple(codebook.shape)}"
        )
    if int(latents.shape[1]) != int(codebook.shape[1]):
        raise ValueError(
            f"latent_dim mismatch: latents {latents.shape[1]} vs codebook {codebook.shape[1]}"
        )
    # Euclidean distance per van den Oord §3.1:
    # ||z_e||^2 - 2 * z_e @ cb.T + ||cb||^2
    ze = latents.to(dtype=torch.float32).cpu()
    cb = codebook.to(dtype=torch.float32).cpu()
    ze_sq = ze.pow(2).sum(dim=1, keepdim=True)
    cb_sq = cb.pow(2).sum(dim=1).unsqueeze(0)
    dists = ze_sq - 2.0 * (ze @ cb.t()) + cb_sq
    return dists.argmin(dim=1).to(dtype=torch.int64)


def pack_archive_from_exported_state_dict(
    *,
    exported_state_dict: dict[str, np.ndarray],
    cfg: PactNervVqConfig,
) -> bytes:
    """Pack a PyTorch-layout exported MLX state dict into PVQ ``0.bin`` bytes.

    The exported state dict carries (per the VQ MLX bridge):
    - ``latents``: (num_pairs, latent_dim) — the per-pair encoder outputs.
    - ``quantizer.codebook``: (codebook_size, latent_dim) — learned VQ codebook.
    - ``quantizer.ema_cluster_size`` + ``quantizer.ema_w``: EMA buffers
      (DROPPED at archive time; the receiver only needs the static codebook).
    - ``latent_embed.weight`` / ``latent_embed.bias``: linear projection.
    - ``blocks.<i>.dsc.depthwise.{weight,bias}`` + ``.pointwise.{weight,bias}``:
      depth-separable upsample stack.
    - ``head_rgb_0.{weight,bias}`` / ``head_rgb_1.{weight,bias}``: RGB heads.

    The substrate-distinguishing archive packing:
    1. Compute per-pair codebook indices via nearest-codebook lookup
       (van den Oord §3.1 Euclidean distance metric).
    2. Ship the static codebook (int16-quantized to uint16 range).
    3. Ship per-pair indices (uint16 since codebook_size <= 65535).
    4. Ship the decoder state_dict via brotli-compressed pickle.

    NO per-pair latents are shipped — the receiver reconstructs them via
    ``codebook[indices[i]]`` per van den Oord §3.1.
    """

    if "latents" not in exported_state_dict:
        raise ValueError("exported_state_dict missing 'latents' tensor")
    if "quantizer.codebook" not in exported_state_dict:
        raise ValueError("exported_state_dict missing 'quantizer.codebook' tensor")

    latents_np = exported_state_dict["latents"]
    codebook_np = exported_state_dict["quantizer.codebook"]

    latents = torch.from_numpy(np.asarray(latents_np).copy()).to(dtype=torch.float32)
    codebook = torch.from_numpy(np.asarray(codebook_np).copy()).to(dtype=torch.float32)

    if int(latents.shape[0]) != int(cfg.num_pairs):
        raise ValueError(
            f"latents.shape[0]={int(latents.shape[0])} != cfg.num_pairs={int(cfg.num_pairs)}"
        )
    if int(codebook.shape[0]) != int(cfg.codebook_size):
        raise ValueError(
            f"codebook.shape[0]={int(codebook.shape[0])} != cfg.codebook_size={int(cfg.codebook_size)}"
        )

    indices = _quantize_latents_via_codebook(latents, codebook)

    # Build decoder state dict (excludes per-pair latents + VQ EMA buffers).
    decoder_state: dict[str, torch.Tensor] = {}
    excluded = {
        "latents",
        "quantizer.codebook",
        "quantizer.ema_cluster_size",
        "quantizer.ema_w",
    }
    for name, arr in exported_state_dict.items():
        if name in excluded:
            continue
        decoder_state[name] = torch.from_numpy(
            np.asarray(arr).copy()
        ).to(dtype=torch.float32)

    return pack_archive(
        decoder_state,
        codebook,
        indices,
        vq_meta_from_config(cfg),
    )


def export_pact_nerv_vq_mlx_archive(
    model: Any,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    emit_archive_bound_candidate_package: bool = True,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
) -> tuple[Path, str, int]:
    """Export an MLX VQ model as a contest-shaped ``archive.zip``."""

    root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[4]
    )
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = model.cfg
    exported = model.export_state_dict()
    bin_bytes = pack_archive_from_exported_state_dict(
        exported_state_dict=exported,
        cfg=cfg,
    )
    bin_path = out_dir / "0.bin"
    bin_path.write_bytes(bin_bytes)

    submission_dir = out_dir / "submission"
    write_contest_runtime(
        submission_dir,
        substrate_pkg_name="pact_nerv_vq",
        repo_root=root,
    )
    (submission_dir / "0.bin").write_bytes(bin_bytes)
    archive_zip_path = out_dir / "archive.zip"
    build_archive_zip(
        archive_zip_path,
        bin_bytes=bin_bytes,
        submission_dir=submission_dir,
    )
    archive_sha256 = sha256_file(archive_zip_path)
    archive_bytes = archive_zip_path.stat().st_size
    if emit_archive_bound_candidate_package:
        emit_archive_bound_candidate_runtime_package(
            adapter_id=PACT_NERV_VQ_MLX_ARCHIVE_BOUND_ADAPTER_ID,
            candidate_family=PACT_NERV_VQ_MLX_ARCHIVE_CANDIDATE_FAMILY,
            candidate_id_prefix="pact_nerv_vq_mlx",
            transform_kind=PACT_NERV_VQ_MLX_ARCHIVE_TRANSFORM_KIND,
            archive_zip_path=archive_zip_path,
            archive_sha256=archive_sha256,
            archive_bytes=archive_bytes,
            submission_dir=submission_dir,
            output_dir=out_dir,
            repo_root=root,
            receiver_contract_kind="pact_nerv_vq_mlx_generated_inflate_sh_decode_only_receiver",
            proof_schema=PACT_NERV_VQ_MLX_RECEIVER_PROOF_SCHEMA,
            proof_filename="pact_nerv_vq_mlx_receiver_proof.json",
            candidate_label="pact_nerv_vq",
            retain_receiver_output=retain_receiver_proof_output,
            runtime_adapter_manifest_extra={
                "schema": "pact_nerv_vq_mlx_runtime_adapter_manifest.v1",
                "discrete_representation": "vector_quantized_codebook_indices",
                "codebook_size": int(cfg.codebook_size),
            },
            candidate_row_schema="pact_nerv_vq_mlx_archive_bound_candidate_row.v1",
            wrapper_schema=PACT_NERV_VQ_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
            mlx_triage_argv=mlx_triage_argv,
        )
    return (archive_zip_path, archive_sha256, archive_bytes)


def export_pact_nerv_vq_mlx_archive_bound_candidate_package(
    model: Any,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Export PACT-NeRV-VQ MLX bytes and emit the shared package."""

    archive_zip_path, archive_sha256, archive_bytes = (
        export_pact_nerv_vq_mlx_archive(
            model,
            output_dir,
            repo_root=repo_root,
            emit_archive_bound_candidate_package=False,
        )
    )
    root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[4]
    )
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    cfg = model.cfg
    return emit_archive_bound_candidate_runtime_package(
        adapter_id=PACT_NERV_VQ_MLX_ARCHIVE_BOUND_ADAPTER_ID,
        candidate_family=PACT_NERV_VQ_MLX_ARCHIVE_CANDIDATE_FAMILY,
        candidate_id_prefix="pact_nerv_vq_mlx",
        transform_kind=PACT_NERV_VQ_MLX_ARCHIVE_TRANSFORM_KIND,
        archive_zip_path=archive_zip_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        submission_dir=out_dir / "submission",
        output_dir=out_dir,
        repo_root=root,
        receiver_contract_kind="pact_nerv_vq_mlx_generated_inflate_sh_decode_only_receiver",
        proof_schema=PACT_NERV_VQ_MLX_RECEIVER_PROOF_SCHEMA,
        proof_filename="pact_nerv_vq_mlx_receiver_proof.json",
        candidate_label="pact_nerv_vq",
        retain_receiver_output=retain_receiver_proof_output,
        runtime_adapter_manifest_extra={
            "schema": "pact_nerv_vq_mlx_runtime_adapter_manifest.v1",
            "discrete_representation": "vector_quantized_codebook_indices",
            "codebook_size": int(cfg.codebook_size),
        },
        candidate_row_schema="pact_nerv_vq_mlx_archive_bound_candidate_row.v1",
        wrapper_schema=PACT_NERV_VQ_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        mlx_triage_argv=mlx_triage_argv,
    )


__all__ = [
    "PACT_NERV_VQ_MLX_ARCHIVE_BOUND_ADAPTER_ID",
    "PACT_NERV_VQ_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA",
    "PACT_NERV_VQ_MLX_ARCHIVE_CANDIDATE_FAMILY",
    "PACT_NERV_VQ_MLX_ARCHIVE_TRANSFORM_KIND",
    "export_pact_nerv_vq_mlx_archive",
    "export_pact_nerv_vq_mlx_archive_bound_candidate_package",
    "pack_archive_from_exported_state_dict",
    "vq_meta_from_config",
]
