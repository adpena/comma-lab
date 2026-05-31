# SPDX-License-Identifier: MIT
"""Byte-closed PSV4 archive export helpers for MLX/local training artifacts.

Mirrors the canonical sister at
:mod:`tac.substrates.pact_nerv_selector_v3.archive_candidate` per the 11th
INDIVIDUALLY-FRACTAL standing directive 2026-05-27 - SELECTOR-V4's OWN
archive export pass, NOT a shared-helper shortcut from SELECTOR-V3.

The substrate-distinguishing primitive (Run-Length selector coding via
varint per Robinson-Cherry 1967 + Capon 1959) is delegated to the canonical
:class:`tac.substrates.pact_nerv_selector_v4.architecture.RunLengthSelectorCoder`;
this module bridges the MLX-trained state_dict into the PSV4 byte-closed
archive used by the inflate runtime.

Note on L0 SCAFFOLD vs ULTIMATE Step 13 spec:

The ULTIMATE memo describes Variant #13 as a "per-pair-per-class arithmetic
coder via per_segnet_class_chroma_priors"; the actual L0 SCAFFOLD landed by
the G3 design memo implements RLE. Both fall within "SELECTOR-PARADIGM-
EXTENSIONS" Group 3. This L1 archive_candidate follows the EXISTING L0
SCAFFOLD (RLE) per Catalog #229 PV + Catalog #303 cargo-cult discipline.
The ULTIMATE-aligned per-pair-per-class arithmetic coder VARIANT is queued
as the operator-routable next-step OR a sister substrate (e.g.
PACT-NeRV-SELECTOR-V5).
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

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
from tac.substrates.pact_nerv_selector_v4.architecture import (
    PactNervSelectorV4Config,
    RunLengthSelectorCoder,
)
from tac.substrates.pact_nerv_selector_v4.archive import pack_archive

PACT_NERV_SELECTOR_V4_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA = (
    "pact_nerv_selector_v4_mlx_archive_bound_adapter_package.v1"
)
PACT_NERV_SELECTOR_V4_MLX_RECEIVER_PROOF_SCHEMA = (
    "pact_nerv_selector_v4_mlx_generated_receiver_proof.v1"
)
PACT_NERV_SELECTOR_V4_MLX_ARCHIVE_BOUND_ADAPTER_ID = (
    "pact_nerv_selector_v4_mlx_archive_export"
)
PACT_NERV_SELECTOR_V4_MLX_ARCHIVE_CANDIDATE_FAMILY = "pact_nerv_selector_v4_mlx"
PACT_NERV_SELECTOR_V4_MLX_ARCHIVE_TRANSFORM_KIND = (
    "pact_nerv_selector_v4_mlx_archive"
)


def selector_v4_meta_from_config(cfg: PactNervSelectorV4Config) -> dict[str, object]:
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
        "selector_palette_size": int(cfg.selector_palette_size),
    }


def pack_archive_from_exported_state_dict(
    *,
    exported_state_dict: dict[str, np.ndarray],
    cfg: PactNervSelectorV4Config,
    selectors: np.ndarray | None = None,
) -> bytes:
    """Pack a PyTorch-layout exported MLX state dict into PSV4 ``0.bin`` bytes."""

    if "latents" not in exported_state_dict:
        raise ValueError("exported_state_dict missing latents")
    if selectors is None:
        selectors = np.zeros(cfg.num_pairs, dtype=np.int64)
    selectors = np.asarray(selectors)
    if selectors.shape != (cfg.num_pairs,):
        raise ValueError(
            f"selectors shape {selectors.shape} != ({cfg.num_pairs},)"
        )
    if not np.issubdtype(selectors.dtype, np.integer):
        raise ValueError(f"selectors must be integer dtype; got {selectors.dtype}")
    selector_list = [int(value) for value in selectors.tolist()]
    coder = RunLengthSelectorCoder(
        palette_size=int(cfg.selector_palette_size),
    )
    selector_bytes = coder.encode(selector_list)
    decoder_state: dict[str, torch.Tensor] = {}
    for name, arr in exported_state_dict.items():
        tensor = torch.from_numpy(np.asarray(arr).copy())
        if name == "latents":
            latents = tensor.to(dtype=torch.float32)
        elif name != "selectors":
            decoder_state[name] = tensor.to(dtype=torch.float32)
    return pack_archive(
        decoder_state,
        latents,
        selector_bytes,
        selector_v4_meta_from_config(cfg),
        palette_size=int(cfg.selector_palette_size),
    )


def export_pact_nerv_selector_v4_mlx_archive(
    model: Any,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    emit_archive_bound_candidate_package: bool = True,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
) -> tuple[Path, str, int]:
    """Export an MLX SELECTOR-V4 model as a contest-shaped ``archive.zip``."""

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
        selectors=getattr(model, "selectors", None),
    )
    bin_path = out_dir / "0.bin"
    bin_path.write_bytes(bin_bytes)

    submission_dir = out_dir / "submission"
    write_contest_runtime(
        submission_dir,
        substrate_pkg_name="pact_nerv_selector_v4",
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
            adapter_id=PACT_NERV_SELECTOR_V4_MLX_ARCHIVE_BOUND_ADAPTER_ID,
            candidate_family=PACT_NERV_SELECTOR_V4_MLX_ARCHIVE_CANDIDATE_FAMILY,
            candidate_id_prefix="pact_nerv_selector_v4_mlx",
            transform_kind=PACT_NERV_SELECTOR_V4_MLX_ARCHIVE_TRANSFORM_KIND,
            archive_zip_path=archive_zip_path,
            archive_sha256=archive_sha256,
            archive_bytes=archive_bytes,
            submission_dir=submission_dir,
            output_dir=out_dir,
            repo_root=root,
            receiver_contract_kind=(
                "pact_nerv_selector_v4_mlx_generated_inflate_sh_decode_only_receiver"
            ),
            proof_schema=PACT_NERV_SELECTOR_V4_MLX_RECEIVER_PROOF_SCHEMA,
            proof_filename="pact_nerv_selector_v4_mlx_receiver_proof.json",
            candidate_label="pact_nerv_selector_v4",
            retain_receiver_output=retain_receiver_proof_output,
            runtime_adapter_manifest_extra={
                "schema": "pact_nerv_selector_v4_mlx_runtime_adapter_manifest.v1",
                "selector_codec": "run_length_varint_selector",
                "selector_palette_size": int(cfg.selector_palette_size),
            },
            candidate_row_schema="pact_nerv_selector_v4_mlx_archive_bound_candidate_row.v1",
            wrapper_schema=PACT_NERV_SELECTOR_V4_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
            mlx_triage_argv=mlx_triage_argv,
        )
    return (archive_zip_path, archive_sha256, archive_bytes)


def export_pact_nerv_selector_v4_mlx_archive_bound_candidate_package(
    model: Any,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Export Selector-V4 MLX bytes and emit the shared package."""

    archive_zip_path, archive_sha256, archive_bytes = (
        export_pact_nerv_selector_v4_mlx_archive(
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
        adapter_id=PACT_NERV_SELECTOR_V4_MLX_ARCHIVE_BOUND_ADAPTER_ID,
        candidate_family=PACT_NERV_SELECTOR_V4_MLX_ARCHIVE_CANDIDATE_FAMILY,
        candidate_id_prefix="pact_nerv_selector_v4_mlx",
        transform_kind=PACT_NERV_SELECTOR_V4_MLX_ARCHIVE_TRANSFORM_KIND,
        archive_zip_path=archive_zip_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        submission_dir=out_dir / "submission",
        output_dir=out_dir,
        repo_root=root,
        receiver_contract_kind=(
            "pact_nerv_selector_v4_mlx_generated_inflate_sh_decode_only_receiver"
        ),
        proof_schema=PACT_NERV_SELECTOR_V4_MLX_RECEIVER_PROOF_SCHEMA,
        proof_filename="pact_nerv_selector_v4_mlx_receiver_proof.json",
        candidate_label="pact_nerv_selector_v4",
        retain_receiver_output=retain_receiver_proof_output,
        runtime_adapter_manifest_extra={
            "schema": "pact_nerv_selector_v4_mlx_runtime_adapter_manifest.v1",
            "selector_codec": "run_length_varint_selector",
            "selector_palette_size": int(cfg.selector_palette_size),
        },
        candidate_row_schema="pact_nerv_selector_v4_mlx_archive_bound_candidate_row.v1",
        wrapper_schema=PACT_NERV_SELECTOR_V4_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        mlx_triage_argv=mlx_triage_argv,
    )


__all__ = [
    "PACT_NERV_SELECTOR_V4_MLX_ARCHIVE_BOUND_ADAPTER_ID",
    "PACT_NERV_SELECTOR_V4_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA",
    "PACT_NERV_SELECTOR_V4_MLX_ARCHIVE_CANDIDATE_FAMILY",
    "PACT_NERV_SELECTOR_V4_MLX_ARCHIVE_TRANSFORM_KIND",
    "export_pact_nerv_selector_v4_mlx_archive",
    "export_pact_nerv_selector_v4_mlx_archive_bound_candidate_package",
    "pack_archive_from_exported_state_dict",
    "selector_v4_meta_from_config",
]
