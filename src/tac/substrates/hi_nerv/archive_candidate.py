# SPDX-License-Identifier: MIT
"""Byte-closed HiNeRV archive export helpers for MLX/local training artifacts.

This module is the receiver/bundling half of the MLX HiNeRV adapter.  It
bridges the MLX renderer's PyTorch-layout ``export_state_dict()`` into the HIV1
archive grammar, writes a contest-shaped ``archive.zip``, projects the payload
into the HPRC representation spine for byte-value accounting, and emits the
shared archive-bound receiver proof/package.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from tac.framework_agnostic.helpers import (
    npz_to_numpy_primitives,
    write_npz_bridge_artifact,
)
from tac.local_acceleration.mlx_numpy_portability_contract import (
    build_mlx_numpy_portability_contract,
)
from tac.optimization.archive_bound_candidate_runtime_bridge import (
    emit_archive_bound_candidate_runtime_package,
)
from tac.repo_io import sha256_file
from tac.substrates._shared.inflate_runtime import CAMERA_HW
from tac.substrates._shared.pact_nerv_full_main import (
    build_archive_zip,
    write_contest_runtime,
)
from tac.substrates.hi_nerv.architecture import validate_decoder_state_dict
from tac.substrates.hi_nerv.archive import pack_archive
from tac.substrates.hprc.representation_spine import (
    build_hi_nerv_spine_from_archive_payload,
    write_representation_spine_projection,
)

if TYPE_CHECKING:
    from tac.substrates.hi_nerv.architecture import HinervConfig

HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA = (
    "hi_nerv_mlx_archive_bound_adapter_package.v1"
)
HI_NERV_MLX_RECEIVER_PROOF_SCHEMA = "hi_nerv_mlx_generated_receiver_proof.v1"
HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_ID = "hi_nerv_mlx_archive_export"
HI_NERV_MLX_ARCHIVE_CANDIDATE_FAMILY = "hi_nerv_mlx"
HI_NERV_MLX_ARCHIVE_TRANSFORM_KIND = "hi_nerv_mlx_archive"

_LATENT_KEYS = ("latents_coarse", "latents_mid", "latents_fine")
_STATE_NPZ_NAME = "hi_nerv_mlx_exported_state.npz"
_STATE_NPZ_MANIFEST_NAME = "hi_nerv_mlx_exported_state_npz_manifest.json"


def hi_nerv_mlx_numpy_portability_contract(
    *,
    canonical_npz_bridge_used: bool = True,
    training_backend: str = "mlx",
) -> dict[str, Any]:
    """Return the honest portability contract for the current HiNeRV receiver."""

    backend = str(training_backend)
    return build_mlx_numpy_portability_contract(
        substrate_id="hi_nerv",
        training_backend=backend,
        exported_state_kind=f"pytorch_layout_numpy_arrays_from_{backend}_model",
        archive_payload_kind="hiv1_monolithic_0_bin",
        receiver_runtime_kind="torch_decode_receiver",
        receiver_dependencies=("torch", "brotli", "python_stdlib"),
        numpy_array_export=True,
        canonical_npz_bridge_used=canonical_npz_bridge_used,
        pure_numpy_inflate=False,
        notes=(
            "HiNeRV MLX export is NumPy-array backed, but the contest receiver "
            "currently decodes with PyTorch. This is contest-compliant when "
            "dependency closure passes, but not pure NumPy inflate."
        ),
    )


def _expected_receiver_output_bytes(cfg: HinervConfig) -> int:
    return int(cfg.num_pairs) * 2 * int(CAMERA_HW[0]) * int(CAMERA_HW[1]) * 3


def hi_nerv_meta_from_config(cfg: HinervConfig) -> dict[str, object]:
    """Return the minimal receiver metadata needed to rebuild the decoder."""

    return {
        "embed_dim": int(cfg.embed_dim),
        "initial_grid_h": int(cfg.initial_grid_h),
        "initial_grid_w": int(cfg.initial_grid_w),
        "decoder_channels": [int(value) for value in cfg.decoder_channels],
        "sin_frequency": float(cfg.sin_frequency),
        "num_upsample_blocks": int(cfg.num_upsample_blocks),
        "mid_injection_block_index": int(cfg.mid_injection_block_index),
        "fine_injection_block_index": int(cfg.fine_injection_block_index),
        "output_height": int(cfg.output_height),
        "output_width": int(cfg.output_width),
    }


def _state_bridge_paths(out_dir: Path) -> tuple[Path, Path]:
    return out_dir / _STATE_NPZ_NAME, out_dir / _STATE_NPZ_MANIFEST_NAME


def _write_and_reload_exported_state_via_numpy_bridge(
    *,
    exported_state_dict: dict[str, np.ndarray],
    output_dir: Path,
    source_backend: str = "mlx",
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Persist and reload the exact NumPy bridge consumed by the packer."""

    npz_path, manifest_path = _state_bridge_paths(output_dir)
    manifest = write_npz_bridge_artifact(
        exported_state_dict,
        npz_path,
        source_backend=str(source_backend),
        bridge_kind=f"hi_nerv_{source_backend}_export_state_dict_to_npz",
        manifest_path=manifest_path,
        require_finite=True,
    )
    if manifest.get("consumption_recommended") is not True:
        raise ValueError(
            "HiNeRV MLX export NPZ bridge is not consumption-recommended: "
            f"{manifest.get('blockers')}"
        )
    return npz_to_numpy_primitives(npz_path.read_bytes()), manifest


def _require_exported_tensor(
    exported_state_dict: dict[str, np.ndarray],
    key: str,
) -> torch.Tensor:
    if key not in exported_state_dict:
        raise ValueError(f"exported_state_dict missing {key!r}")
    return torch.from_numpy(np.asarray(exported_state_dict[key]).copy()).to(
        dtype=torch.float32
    )


def pack_archive_from_exported_state_dict(
    *,
    exported_state_dict: dict[str, np.ndarray],
    cfg: HinervConfig,
    decoder_codec: str = "int8_mixed",
) -> bytes:
    """Pack PyTorch-layout exported MLX tensors into HIV1 ``0.bin`` bytes."""

    latents_coarse = _require_exported_tensor(exported_state_dict, "latents_coarse")
    latents_mid = _require_exported_tensor(exported_state_dict, "latents_mid")
    latents_fine = _require_exported_tensor(exported_state_dict, "latents_fine")
    expected_shapes = {
        "latents_coarse": (int(cfg.num_pairs), int(cfg.latent_dim_coarse)),
        "latents_mid": (int(cfg.num_pairs), int(cfg.latent_dim_mid)),
        "latents_fine": (int(cfg.num_pairs), int(cfg.latent_dim_fine)),
    }
    for key, tensor in (
        ("latents_coarse", latents_coarse),
        ("latents_mid", latents_mid),
        ("latents_fine", latents_fine),
    ):
        if tuple(int(v) for v in tensor.shape) != expected_shapes[key]:
            raise ValueError(
                f"{key} shape {tuple(tensor.shape)} != {expected_shapes[key]}"
            )

    decoder_state: dict[str, torch.Tensor] = {}
    for name, arr in exported_state_dict.items():
        if name in _LATENT_KEYS:
            continue
        decoder_state[name] = torch.from_numpy(np.asarray(arr).copy()).to(
            dtype=torch.float32
        )
    validate_decoder_state_dict(
        decoder_state,
        cfg,
        context="hi_nerv_exported_decoder_state",
    )

    return pack_archive(
        decoder_state,
        latents_coarse,
        latents_mid,
        latents_fine,
        hi_nerv_meta_from_config(cfg),
        decoder_codec=decoder_codec,
    )


def export_hi_nerv_mlx_archive(
    model: Any,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    emit_archive_bound_candidate_package: bool = True,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
    decoder_codec: str = "int8_mixed",
    source_backend: str = "mlx",
) -> tuple[Path, str, int]:
    """Export an MLX HiNeRV model as a contest-shaped ``archive.zip``."""

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
    exported_state_dict, npz_bridge_manifest = (
        _write_and_reload_exported_state_via_numpy_bridge(
            exported_state_dict=model.export_state_dict(),
            output_dir=out_dir,
            source_backend=source_backend,
        )
    )
    bin_bytes = pack_archive_from_exported_state_dict(
        exported_state_dict=exported_state_dict,
        cfg=cfg,
        decoder_codec=decoder_codec,
    )
    bin_path = out_dir / "0.bin"
    bin_path.write_bytes(bin_bytes)

    submission_dir = out_dir / "submission"
    write_contest_runtime(
        submission_dir,
        substrate_pkg_name="hi_nerv",
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
    write_representation_spine_projection(
        output_dir=out_dir,
        spine=build_hi_nerv_spine_from_archive_payload(
            bin_bytes,
            source={
                "kind": "hi_nerv_export_payload",
                "archive_zip_path": archive_zip_path.as_posix(),
                "archive_zip_sha256": archive_sha256,
                "archive_zip_bytes": int(archive_bytes),
            },
            manifest_extra={
                "emitted_by": "export_hi_nerv_mlx_archive",
                "archive_bytes_are_authority_for_rate": True,
                "decoder_codec": decoder_codec,
                "num_pairs": int(cfg.num_pairs),
                "state_npz_bridge": {
                    "artifact_path": npz_bridge_manifest["artifact_path"],
                    "artifact_sha256": npz_bridge_manifest["artifact_sha256"],
                    "manifest_path": npz_bridge_manifest["manifest_path"],
                    "tensor_count": npz_bridge_manifest["tensor_count"],
                },
                "export_source_backend": str(source_backend),
            },
        ),
        basename="hprc_representation_spine_hi_nerv",
    )
    if emit_archive_bound_candidate_package:
        emit_archive_bound_candidate_runtime_package(
            adapter_id=HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_ID,
            candidate_family=HI_NERV_MLX_ARCHIVE_CANDIDATE_FAMILY,
            candidate_id_prefix="hi_nerv_mlx",
            transform_kind=HI_NERV_MLX_ARCHIVE_TRANSFORM_KIND,
            archive_zip_path=archive_zip_path,
            archive_sha256=archive_sha256,
            archive_bytes=archive_bytes,
            submission_dir=submission_dir,
            output_dir=out_dir,
            repo_root=root,
            receiver_contract_kind="hi_nerv_mlx_generated_inflate_sh_decode_only_receiver",
            proof_schema=HI_NERV_MLX_RECEIVER_PROOF_SCHEMA,
            proof_filename="hi_nerv_mlx_receiver_proof.json",
            candidate_label="hi_nerv",
            expected_receiver_output_name="0.raw",
            expected_receiver_output_bytes=_expected_receiver_output_bytes(cfg),
            retain_receiver_output=retain_receiver_proof_output,
            runtime_adapter_manifest_extra={
                "schema": "hi_nerv_mlx_runtime_adapter_manifest.v1",
                "latent_pyramid": ["coarse", "mid", "fine"],
                "decoder_codec": decoder_codec,
                "num_pairs": int(cfg.num_pairs),
                "state_npz_bridge_manifest": npz_bridge_manifest,
                "mlx_numpy_portability_contract": (
                    hi_nerv_mlx_numpy_portability_contract(
                        training_backend=source_backend
                    )
                ),
            },
            candidate_row_schema="hi_nerv_mlx_archive_bound_candidate_row.v1",
            wrapper_schema=HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
            mlx_triage_argv=mlx_triage_argv,
        )
    return (archive_zip_path, archive_sha256, archive_bytes)


def export_hi_nerv_mlx_archive_bound_candidate_package(
    model: Any,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
    decoder_codec: str = "int8_mixed",
    source_backend: str = "mlx",
) -> dict[str, Any]:
    """Export HiNeRV MLX bytes and emit the shared candidate package."""

    archive_zip_path, archive_sha256, archive_bytes = export_hi_nerv_mlx_archive(
        model,
        output_dir,
        repo_root=repo_root,
        emit_archive_bound_candidate_package=False,
        decoder_codec=decoder_codec,
        source_backend=source_backend,
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
    _, npz_bridge_manifest_path = _state_bridge_paths(out_dir)
    npz_bridge_manifest = json.loads(
        npz_bridge_manifest_path.read_text(encoding="utf-8")
    )
    return emit_archive_bound_candidate_runtime_package(
        adapter_id=HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_ID,
        candidate_family=HI_NERV_MLX_ARCHIVE_CANDIDATE_FAMILY,
        candidate_id_prefix="hi_nerv_mlx",
        transform_kind=HI_NERV_MLX_ARCHIVE_TRANSFORM_KIND,
        archive_zip_path=archive_zip_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        submission_dir=out_dir / "submission",
        output_dir=out_dir,
        repo_root=root,
        receiver_contract_kind="hi_nerv_mlx_generated_inflate_sh_decode_only_receiver",
        proof_schema=HI_NERV_MLX_RECEIVER_PROOF_SCHEMA,
        proof_filename="hi_nerv_mlx_receiver_proof.json",
        candidate_label="hi_nerv",
        expected_receiver_output_name="0.raw",
        expected_receiver_output_bytes=_expected_receiver_output_bytes(cfg),
        retain_receiver_output=retain_receiver_proof_output,
        runtime_adapter_manifest_extra={
            "schema": "hi_nerv_mlx_runtime_adapter_manifest.v1",
            "latent_pyramid": ["coarse", "mid", "fine"],
            "decoder_codec": decoder_codec,
            "num_pairs": int(cfg.num_pairs),
            "state_npz_bridge_manifest": npz_bridge_manifest,
            "mlx_numpy_portability_contract": (
                hi_nerv_mlx_numpy_portability_contract(
                    training_backend=source_backend
                )
            ),
        },
        candidate_row_schema="hi_nerv_mlx_archive_bound_candidate_row.v1",
        wrapper_schema=HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        mlx_triage_argv=mlx_triage_argv,
    )


__all__ = [
    "HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_ID",
    "HI_NERV_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA",
    "HI_NERV_MLX_ARCHIVE_CANDIDATE_FAMILY",
    "HI_NERV_MLX_ARCHIVE_TRANSFORM_KIND",
    "export_hi_nerv_mlx_archive",
    "export_hi_nerv_mlx_archive_bound_candidate_package",
    "hi_nerv_meta_from_config",
    "hi_nerv_mlx_numpy_portability_contract",
    "pack_archive_from_exported_state_dict",
]
