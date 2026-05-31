# SPDX-License-Identifier: MIT
"""Byte-closed Z6V2 archive export helpers for MLX/local training artifacts.

Mirrors the canonical sister at
:mod:`tac.substrates.pact_nerv_selector_v3.archive_candidate` per the 11th
INDIVIDUALLY-FRACTAL standing directive 2026-05-27 — Z6-v2's OWN archive
export pass, NOT shared-helper shortcut from PACT-NeRV sister cascade.

Z6-v2's substrate-distinguishing primitive (2-level Rao-Ballard hierarchical
FiLM-ego-motion predictor) is implemented at the ``architecture.py`` +
``mlx_renderer.py`` surfaces; this module bridges the MLX-trained state_dict
into the Z6V2CU1 byte-closed archive used by the inflate runtime.

[verified-against: Catalog #146 contest-compliant inflate runtime template]
[verified-against: Catalog #205 canonical select_inflate_device]
[verified-against: Catalog #295 PYTHONPATH self-containment]
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from tac.optimization.archive_bound_candidate_runtime_bridge import (
    build_archive_bound_candidate_runtime_package,
    run_generated_inflate_receiver_proof,
)
from tac.repo_io import sha256_file
from tac.substrates._shared.pact_nerv_full_main import (
    build_archive_zip,
    write_contest_runtime,
)
from tac.substrates.z6_v2_cargo_cult_unwind.archive import pack_archive

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tac.substrates.z6_v2_cargo_cult_unwind.architecture import Z6V2Config

Z6_V2_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA = (
    "z6_v2_mlx_archive_bound_adapter_package.v1"
)
Z6_V2_MLX_RECEIVER_PROOF_SCHEMA = "z6_v2_mlx_generated_receiver_proof.v1"
Z6_V2_MLX_ARCHIVE_BOUND_ADAPTER_ID = "z6_v2_mlx_archive_export"
Z6_V2_MLX_ARCHIVE_CANDIDATE_FAMILY = "z6_v2_cargo_cult_unwind_mlx"
Z6_V2_MLX_ARCHIVE_TRANSFORM_KIND = (
    "z6_v2_mlx_rao_ballard_predictive_coding_archive"
)


def z6_v2_meta_from_config(cfg: Z6V2Config) -> dict[str, object]:
    """Minimal receiver metadata needed to rebuild Z6V2Substrate at inflate time."""
    return {
        "embed_dim": int(cfg.embed_dim),
        "initial_grid_h": int(cfg.initial_grid_h),
        "initial_grid_w": int(cfg.initial_grid_w),
        "decoder_channels": [int(c) for c in cfg.decoder_channels],
        "sin_frequency": float(cfg.sin_frequency),
        "num_upsample_blocks": int(cfg.num_upsample_blocks),
        "output_height": int(cfg.output_height),
        "output_width": int(cfg.output_width),
        "rao_ballard_level_boundary": int(cfg.rao_ballard_level_boundary),
        "film_generator_depth": int(cfg.film_generator_depth),
        "film_hidden_width": int(cfg.film_hidden_width),
        "cooperative_receiver_beta": float(cfg.cooperative_receiver_beta),
    }


def pack_archive_from_exported_state_dict(
    *,
    exported_state_dict: dict[str, np.ndarray],
    cfg: Z6V2Config,
) -> bytes:
    """Pack a PyTorch-layout exported MLX state dict into Z6V2CU1 ``0.bin`` bytes."""
    if "latents" not in exported_state_dict:
        raise ValueError("exported_state_dict missing latents")
    if "ego_vecs" not in exported_state_dict:
        raise ValueError("exported_state_dict missing ego_vecs")

    decoder_state: dict[str, torch.Tensor] = {}
    latents = None
    ego_vecs = None
    for name, arr in exported_state_dict.items():
        tensor = torch.from_numpy(np.asarray(arr).copy())
        if name == "latents":
            latents = tensor.to(dtype=torch.float32)
        elif name == "ego_vecs":
            ego_vecs = tensor.to(dtype=torch.float32)
        else:
            decoder_state[name] = tensor.to(dtype=torch.float32)
    if latents is None or ego_vecs is None:
        raise ValueError(
            "exported_state_dict must contain both 'latents' and 'ego_vecs'"
        )
    return pack_archive(
        decoder_state,
        latents,
        ego_vecs,
        z6_v2_meta_from_config(cfg),
    )


def export_z6_v2_mlx_archive(
    model: Any,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    emit_archive_bound_candidate_package: bool = True,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
) -> tuple[Path, str, int]:
    """Export an MLX Z6-v2 model as a contest-shaped ``archive.zip``."""
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
    bin_bytes = pack_archive_from_exported_state_dict(
        exported_state_dict=model.export_state_dict(),
        cfg=cfg,
    )
    bin_path = out_dir / "0.bin"
    bin_path.write_bytes(bin_bytes)

    submission_dir = out_dir / "submission"
    write_contest_runtime(
        submission_dir,
        substrate_pkg_name="z6_v2_cargo_cult_unwind",
        repo_root=root,
        vendor_shared_inflate_runtime=True,
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
        receiver_proof = run_generated_inflate_receiver_proof(
            archive_zip_path=archive_zip_path,
            archive_sha256=archive_sha256,
            archive_bytes=archive_bytes,
            submission_dir=submission_dir,
            output_dir=out_dir,
            repo_root=root,
            proof_schema=Z6_V2_MLX_RECEIVER_PROOF_SCHEMA,
            proof_filename="z6_v2_mlx_receiver_proof.json",
            candidate_label="z6_v2",
            retain_receiver_output=retain_receiver_proof_output,
        )
        build_archive_bound_candidate_runtime_package(
            adapter_id=Z6_V2_MLX_ARCHIVE_BOUND_ADAPTER_ID,
            candidate_family=Z6_V2_MLX_ARCHIVE_CANDIDATE_FAMILY,
            candidate_id_prefix="z6_v2_mlx",
            transform_kind=Z6_V2_MLX_ARCHIVE_TRANSFORM_KIND,
            archive_zip_path=archive_zip_path,
            archive_sha256=archive_sha256,
            archive_bytes=archive_bytes,
            submission_dir=submission_dir,
            output_dir=out_dir,
            repo_root=root,
            receiver_proof=receiver_proof,
            receiver_contract_kind="z6_v2_mlx_generated_inflate_sh_decode_only_receiver",
            runtime_adapter_manifest_extra={
                "schema": "z6_v2_mlx_runtime_adapter_manifest.v1",
                "predictive_coding_family": "rao_ballard_hierarchical_film",
                "cooperative_receiver_beta": float(cfg.cooperative_receiver_beta),
            },
            candidate_row_schema="z6_v2_mlx_archive_bound_candidate_row.v1",
            wrapper_schema=Z6_V2_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
            mlx_triage_argv=mlx_triage_argv,
        )
    return (archive_zip_path, archive_sha256, archive_bytes)


def export_z6_v2_mlx_archive_bound_candidate_package(
    model: Any,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Export Z6-v2 MLX bytes and emit the shared archive-bound package."""
    archive_zip_path, archive_sha256, archive_bytes = export_z6_v2_mlx_archive(
        model,
        output_dir,
        repo_root=repo_root,
        emit_archive_bound_candidate_package=False,
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
    submission_dir = out_dir / "submission"
    receiver_proof = run_generated_inflate_receiver_proof(
        archive_zip_path=archive_zip_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        submission_dir=submission_dir,
        output_dir=out_dir,
        repo_root=root,
        proof_schema=Z6_V2_MLX_RECEIVER_PROOF_SCHEMA,
        proof_filename="z6_v2_mlx_receiver_proof.json",
        candidate_label="z6_v2",
        retain_receiver_output=retain_receiver_proof_output,
    )
    return build_archive_bound_candidate_runtime_package(
        adapter_id=Z6_V2_MLX_ARCHIVE_BOUND_ADAPTER_ID,
        candidate_family=Z6_V2_MLX_ARCHIVE_CANDIDATE_FAMILY,
        candidate_id_prefix="z6_v2_mlx",
        transform_kind=Z6_V2_MLX_ARCHIVE_TRANSFORM_KIND,
        archive_zip_path=archive_zip_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        submission_dir=submission_dir,
        output_dir=out_dir,
        repo_root=root,
        receiver_proof=receiver_proof,
        receiver_contract_kind="z6_v2_mlx_generated_inflate_sh_decode_only_receiver",
        runtime_adapter_manifest_extra={
            "schema": "z6_v2_mlx_runtime_adapter_manifest.v1",
            "predictive_coding_family": "rao_ballard_hierarchical_film",
            "cooperative_receiver_beta": float(cfg.cooperative_receiver_beta),
        },
        candidate_row_schema="z6_v2_mlx_archive_bound_candidate_row.v1",
        wrapper_schema=Z6_V2_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        mlx_triage_argv=mlx_triage_argv,
    )


__all__ = [
    "Z6_V2_MLX_ARCHIVE_BOUND_ADAPTER_ID",
    "Z6_V2_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA",
    "Z6_V2_MLX_ARCHIVE_CANDIDATE_FAMILY",
    "Z6_V2_MLX_ARCHIVE_TRANSFORM_KIND",
    "export_z6_v2_mlx_archive",
    "export_z6_v2_mlx_archive_bound_candidate_package",
    "pack_archive_from_exported_state_dict",
    "z6_v2_meta_from_config",
]
