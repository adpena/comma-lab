# SPDX-License-Identifier: MIT
"""Byte-closed PSV3 archive export helpers for MLX/local training artifacts.

Mirrors the canonical sister at
:mod:`tac.substrates.pact_nerv_selector_v2.archive_candidate` per the 11th
INDIVIDUALLY-FRACTAL standing directive 2026-05-27 — SELECTOR-V3's OWN
archive export pass, NOT a shared-helper shortcut from SELECTOR-V2.

The substrate-distinguishing primitive (Rice-Golomb selector coding per
Golomb 1966 + Rice 1971) is delegated to the canonical
:class:`tac.substrates.pact_nerv_selector_v3.architecture.RiceGolombSelectorCoder`;
this module bridges the MLX-trained state_dict into the PSV3 byte-closed
archive used by the inflate runtime.
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
from tac.substrates.pact_nerv_selector_v3.architecture import (
    PactNervSelectorV3Config,
    RiceGolombSelectorCoder,
)
from tac.substrates.pact_nerv_selector_v3.archive import pack_archive

PACT_NERV_SELECTOR_V3_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA = (
    "pact_nerv_selector_v3_mlx_archive_bound_adapter_package.v1"
)
PACT_NERV_SELECTOR_V3_MLX_RECEIVER_PROOF_SCHEMA = (
    "pact_nerv_selector_v3_mlx_generated_receiver_proof.v1"
)
PACT_NERV_SELECTOR_V3_MLX_ARCHIVE_BOUND_ADAPTER_ID = (
    "pact_nerv_selector_v3_mlx_archive_export"
)
PACT_NERV_SELECTOR_V3_MLX_ARCHIVE_CANDIDATE_FAMILY = "pact_nerv_selector_v3_mlx"
PACT_NERV_SELECTOR_V3_MLX_ARCHIVE_TRANSFORM_KIND = (
    "pact_nerv_selector_v3_mlx_archive"
)


def selector_v3_meta_from_config(cfg: PactNervSelectorV3Config) -> dict[str, object]:
    """Return the minimal receiver metadata needed to rebuild the decoder.

    The Rice-Golomb parameter ``k`` is included so the inflate runtime can
    re-instantiate the canonical coder for selector-stream decoding.
    """

    return {
        "embed_dim": int(cfg.embed_dim),
        "initial_grid_h": int(cfg.initial_grid_h),
        "initial_grid_w": int(cfg.initial_grid_w),
        "decoder_channels": [int(value) for value in cfg.decoder_channels],
        "sin_frequency": float(cfg.sin_frequency),
        "num_upsample_blocks": int(cfg.num_upsample_blocks),
        "output_height": int(cfg.output_height),
        "output_width": int(cfg.output_width),
        "rice_golomb_k": int(cfg.rice_golomb_k),
        "selector_palette_size": int(cfg.selector_palette_size),
    }


def pack_archive_from_exported_state_dict(
    *,
    exported_state_dict: dict[str, np.ndarray],
    cfg: PactNervSelectorV3Config,
    selectors: np.ndarray | None = None,
    decoder_quantization: str = "fp16_brotli_q9",
) -> bytes:
    """Pack a PyTorch-layout exported MLX state dict into PSV3 ``0.bin`` bytes."""

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
    coder = RiceGolombSelectorCoder(
        palette_size=int(cfg.selector_palette_size),
        k=int(cfg.rice_golomb_k),
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
        selector_v3_meta_from_config(cfg),
        palette_size=int(cfg.selector_palette_size),
        decoder_quantization=decoder_quantization,
    )


def export_pact_nerv_selector_v3_mlx_archive(
    model: Any,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    decoder_quantization: str = "fp16_brotli_q9",
    fp4_qat_epochs: int = 0,
    fp4_qat_learning_rate_scale: float = 0.1,
    base_learning_rate: float = 1e-3,
    emit_archive_bound_candidate_package: bool = True,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
) -> tuple[Path, str, int]:
    """Export an MLX SELECTOR-V3 model as a contest-shaped ``archive.zip``.

    WAVE-N+2 SLOT 1 (2026-05-28) Compound C extension: when
    ``decoder_quantization == 'heterogeneous_per_tensor'`` AND
    ``fp4_qat_epochs > 0`` the helper runs the canonical FP4-QAT
    post-training fine-tune on the top-K tensors (selected by the
    canonical sensitivity-ranking helper) BEFORE archive emit. Per
    CLAUDE.md "QAT pipeline" non-negotiable + Quantizr 0.33 canonical
    pattern: scalar-weight-only fine-tune at scaled LR (default 0.1×)
    for ``fp4_qat_epochs`` (Quantizr canonical = 200; smoke = 50).
    """

    root = (
        Path(repo_root)
        if repo_root is not None
        else Path(__file__).resolve().parents[4]
    )
    out_dir = Path(output_dir)
    if not out_dir.is_absolute():
        out_dir = root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    exported_state_dict = model.export_state_dict()

    # WAVE-N+2 SLOT 1: optional FP4-QAT fine-tune on top-K tensors
    # BEFORE archive emit. Only fires when heterogeneous quant + QAT epochs > 0.
    qat_metrics: dict[str, object] = {}
    if (
        decoder_quantization == "heterogeneous_per_tensor"
        and int(fp4_qat_epochs) > 0
    ):
        import json

        import torch as _torch

        from tac.substrates.pact_nerv_selector_v3.heterogeneous_bit_allocation import (
            apply_fp4_qat_finetune_on_top_k_tensors,
            compute_per_tensor_sensitivity_via_taylor_expansion,
            derive_heterogeneous_bit_allocation,
        )

        sd_torch: dict[str, _torch.Tensor] = {}
        for name, arr in exported_state_dict.items():
            if name == "selectors":
                continue
            sd_torch[name] = _torch.from_numpy(np.asarray(arr).copy()).to(
                dtype=_torch.float32
            )
        sensitivity = compute_per_tensor_sensitivity_via_taylor_expansion(sd_torch)
        allocation = derive_heterogeneous_bit_allocation(sd_torch, sensitivity)
        qat_result = apply_fp4_qat_finetune_on_top_k_tensors(
            sd_torch,
            allocation,
            qat_epochs=int(fp4_qat_epochs),
            qat_learning_rate_scale=float(fp4_qat_learning_rate_scale),
            base_learning_rate=float(base_learning_rate),
            seed=0,
        )
        # Replace exported state dict entries with QAT-fine-tuned floats so
        # the archive emit's heterogeneous_per_tensor quantization runs over
        # grid-snapped floats (near-zero quantization error per Quantizr).
        for name in qat_result.fp4_tensors_finetuned:
            t = qat_result.fine_tuned_state_dict[name]
            exported_state_dict[name] = t.detach().cpu().numpy().astype(
                np.float32
            ).copy()
        # Emit QAT metrics sidecar for landing-memo + observability.
        qat_metrics = {
            "fp4_tensors_finetuned": list(qat_result.fp4_tensors_finetuned),
            "qat_epochs": qat_result.qat_epochs,
            "qat_learning_rate": qat_result.qat_learning_rate,
            "final_qat_loss": qat_result.final_qat_loss,
            "per_tensor_cos_pre_qat": dict(qat_result.per_tensor_cos_pre_qat),
            "per_tensor_cos_post_qat": dict(qat_result.per_tensor_cos_post_qat),
            "allocation_rationale": allocation.rationale,
            "rationale": qat_result.rationale,
        }
        (out_dir / "qat_metrics.json").write_text(
            json.dumps(qat_metrics, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    cfg = model.cfg
    bin_bytes = pack_archive_from_exported_state_dict(
        exported_state_dict=exported_state_dict,
        cfg=cfg,
        selectors=getattr(model, "selectors", None),
        decoder_quantization=decoder_quantization,
    )
    bin_path = out_dir / "0.bin"
    bin_path.write_bytes(bin_bytes)

    submission_dir = out_dir / "submission"
    write_contest_runtime(
        submission_dir,
        substrate_pkg_name="pact_nerv_selector_v3",
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
            adapter_id=PACT_NERV_SELECTOR_V3_MLX_ARCHIVE_BOUND_ADAPTER_ID,
            candidate_family=PACT_NERV_SELECTOR_V3_MLX_ARCHIVE_CANDIDATE_FAMILY,
            candidate_id_prefix="pact_nerv_selector_v3_mlx",
            transform_kind=PACT_NERV_SELECTOR_V3_MLX_ARCHIVE_TRANSFORM_KIND,
            archive_zip_path=archive_zip_path,
            archive_sha256=archive_sha256,
            archive_bytes=archive_bytes,
            submission_dir=submission_dir,
            output_dir=out_dir,
            repo_root=root,
            receiver_contract_kind=(
                "pact_nerv_selector_v3_mlx_generated_inflate_sh_decode_only_receiver"
            ),
            proof_schema=PACT_NERV_SELECTOR_V3_MLX_RECEIVER_PROOF_SCHEMA,
            proof_filename="pact_nerv_selector_v3_mlx_receiver_proof.json",
            candidate_label="pact_nerv_selector_v3",
            retain_receiver_output=retain_receiver_proof_output,
            runtime_adapter_manifest_extra={
                "schema": "pact_nerv_selector_v3_mlx_runtime_adapter_manifest.v1",
                "selector_codec": "rice_golomb_selector",
                "selector_palette_size": int(cfg.selector_palette_size),
                "rice_golomb_k": int(cfg.rice_golomb_k),
                "decoder_quantization": decoder_quantization,
                "fp4_qat_epochs": int(fp4_qat_epochs),
            },
            candidate_row_schema="pact_nerv_selector_v3_mlx_archive_bound_candidate_row.v1",
            wrapper_schema=PACT_NERV_SELECTOR_V3_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
            mlx_triage_argv=mlx_triage_argv,
        )
    return (archive_zip_path, archive_sha256, archive_bytes)


def export_pact_nerv_selector_v3_mlx_archive_bound_candidate_package(
    model: Any,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
    decoder_quantization: str = "fp16_brotli_q9",
    fp4_qat_epochs: int = 0,
    fp4_qat_learning_rate_scale: float = 0.1,
    base_learning_rate: float = 1e-3,
    retain_receiver_proof_output: bool = False,
    mlx_triage_argv: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Export Selector-V3 MLX bytes and emit the shared package."""

    archive_zip_path, archive_sha256, archive_bytes = (
        export_pact_nerv_selector_v3_mlx_archive(
            model,
            output_dir,
            repo_root=repo_root,
            decoder_quantization=decoder_quantization,
            fp4_qat_epochs=fp4_qat_epochs,
            fp4_qat_learning_rate_scale=fp4_qat_learning_rate_scale,
            base_learning_rate=base_learning_rate,
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
        adapter_id=PACT_NERV_SELECTOR_V3_MLX_ARCHIVE_BOUND_ADAPTER_ID,
        candidate_family=PACT_NERV_SELECTOR_V3_MLX_ARCHIVE_CANDIDATE_FAMILY,
        candidate_id_prefix="pact_nerv_selector_v3_mlx",
        transform_kind=PACT_NERV_SELECTOR_V3_MLX_ARCHIVE_TRANSFORM_KIND,
        archive_zip_path=archive_zip_path,
        archive_sha256=archive_sha256,
        archive_bytes=archive_bytes,
        submission_dir=out_dir / "submission",
        output_dir=out_dir,
        repo_root=root,
        receiver_contract_kind=(
            "pact_nerv_selector_v3_mlx_generated_inflate_sh_decode_only_receiver"
        ),
        proof_schema=PACT_NERV_SELECTOR_V3_MLX_RECEIVER_PROOF_SCHEMA,
        proof_filename="pact_nerv_selector_v3_mlx_receiver_proof.json",
        candidate_label="pact_nerv_selector_v3",
        retain_receiver_output=retain_receiver_proof_output,
        runtime_adapter_manifest_extra={
            "schema": "pact_nerv_selector_v3_mlx_runtime_adapter_manifest.v1",
            "selector_codec": "rice_golomb_selector",
            "selector_palette_size": int(cfg.selector_palette_size),
            "rice_golomb_k": int(cfg.rice_golomb_k),
            "decoder_quantization": decoder_quantization,
            "fp4_qat_epochs": int(fp4_qat_epochs),
        },
        candidate_row_schema="pact_nerv_selector_v3_mlx_archive_bound_candidate_row.v1",
        wrapper_schema=PACT_NERV_SELECTOR_V3_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA,
        mlx_triage_argv=mlx_triage_argv,
    )


__all__ = [
    "PACT_NERV_SELECTOR_V3_MLX_ARCHIVE_BOUND_ADAPTER_ID",
    "PACT_NERV_SELECTOR_V3_MLX_ARCHIVE_BOUND_ADAPTER_PACKAGE_SCHEMA",
    "PACT_NERV_SELECTOR_V3_MLX_ARCHIVE_CANDIDATE_FAMILY",
    "PACT_NERV_SELECTOR_V3_MLX_ARCHIVE_TRANSFORM_KIND",
    "export_pact_nerv_selector_v3_mlx_archive",
    "export_pact_nerv_selector_v3_mlx_archive_bound_candidate_package",
    "pack_archive_from_exported_state_dict",
    "selector_v3_meta_from_config",
]
