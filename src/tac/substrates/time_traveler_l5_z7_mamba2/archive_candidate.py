# SPDX-License-Identifier: MIT
"""Byte-closed Z7-Mamba-2 archive export helpers for MLX-trained artifacts.

Mirrors the canonical sister
:mod:`tac.substrates.z6_v2_cargo_cult_unwind.archive_candidate` per the 11th
INDIVIDUALLY-FRACTAL standing directive 2026-05-27 — Z7-Mamba-2's OWN archive
export pass, NOT shared-helper shortcut from Z6-v2 sister.

Z7-Mamba-2's substrate-distinguishing primitive (Mamba-2 selective state-space
recurrence + Z6-compatible PixelShuffle decoder) is implemented at the
``architecture.py`` + ``mlx_native.py`` + ``mlx_module.py`` surfaces; this
module bridges the MLX-trained state_dict into the Z7MCM2 byte-closed archive
used by the inflate runtime.

The MLX → Z7MCM2 path:

1. ``Z7Mamba2MLXModule.export_state_dict()`` returns a numpy dict with
   PyTorch-layout keys + shapes (the existing
   ``Z7Mamba2MLXNativeRenderer.export_state_dict`` does the heavy lifting:
   transposes MLX channels-last conv weights ``(out, kH, kW, in)`` to
   PyTorch ``(out, in, kH, kW)``; keys match the canonical PyTorch sister
   ``Z7Mamba2PredictiveCodingSubstrate.state_dict()`` 1:1 per Catalog #1251).
2. THIS module splits that flat dict into ``encoder_state_dict`` (empty for
   ``context_conditioning_mode="none"``; populated for ``"latent_affine"``
   per Z7-LSTM/GRU sister convention) / ``decoder_state_dict`` /
   ``predictor_state_dict`` per the
   :func:`tac.substrates.time_traveler_l5_z7_mamba2.archive.pack_archive`
   contract.
3. Converts each section's numpy arrays to ``torch.Tensor`` (required by
   ``pack_archive`` signature) + extracts ``latent_init`` / ``residuals`` /
   ``ego_motion`` separately.
4. Delegates to ``pack_archive`` for the canonical Z7MCM2 byte stream.
5. Writes ``0.bin`` + builds the contest-shaped ``archive.zip`` via the
   canonical PR95-family helpers ``write_contest_runtime`` +
   ``build_archive_zip`` per Catalog #146 / #205 / #295 / #367.

[verified-against: Catalog #146 contest-compliant inflate runtime template]
[verified-against: Catalog #205 canonical select_inflate_device]
[verified-against: Catalog #295 PYTHONPATH self-containment]
[verified-against: tac.substrates.z6_v2_cargo_cult_unwind.archive_candidate canonical sister pattern]
[verified-against: tac.substrates.time_traveler_l5_z7_mamba2.archive.pack_archive signature]
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from tac.repo_io import sha256_file
from tac.substrates._shared.pact_nerv_full_main import (
    build_archive_zip,
    write_contest_runtime,
)
from tac.substrates.time_traveler_l5_z7_mamba2.architecture import (
    Z7Mamba2PredictiveCodingConfig,
)
from tac.substrates.time_traveler_l5_z7_mamba2.archive import pack_archive

if TYPE_CHECKING:
    from tac.substrates.time_traveler_l5_z7_mamba2.mlx_native import (
        Z7Mamba2MLXRenderConfig,
    )


def z7_mamba2_pytorch_config_from_mlx(
    mlx_cfg: Z7Mamba2MLXRenderConfig,
) -> Z7Mamba2PredictiveCodingConfig:
    """Build a canonical PyTorch config from the MLX render config.

    The MLX render config is a strict subset of the PyTorch config (context
    conditioning is DEFERRED to a sister L1 EXTENSION per the existing
    ``Z7Mamba2MLXRenderConfig`` docstring); we fill in the PyTorch-specific
    fields with their canonical defaults so ``pack_archive`` validates clean.
    """
    return Z7Mamba2PredictiveCodingConfig(
        latent_dim=mlx_cfg.latent_dim,
        ego_motion_dim=mlx_cfg.ego_motion_dim,
        d_model=mlx_cfg.d_model,
        d_state=mlx_cfg.d_state,
        expand=mlx_cfg.expand,
        d_conv=mlx_cfg.d_conv,
        stateful=mlx_cfg.stateful,
        identity_predictor=mlx_cfg.identity_predictor,
        num_pairs=mlx_cfg.num_pairs,
        decoder_embed_dim=mlx_cfg.decoder_embed_dim,
        decoder_initial_grid_h=mlx_cfg.decoder_initial_grid_h,
        decoder_initial_grid_w=mlx_cfg.decoder_initial_grid_w,
        decoder_channels=tuple(int(c) for c in mlx_cfg.decoder_channels),
        decoder_num_upsample_blocks=mlx_cfg.decoder_num_upsample_blocks,
        output_height=mlx_cfg.output_height,
        output_width=mlx_cfg.output_width,
        latent_init_std=mlx_cfg.latent_init_std,
        # MLX scaffold is "none" context mode only (per existing
        # ``Z7Mamba2MLXRenderConfig`` docstring; ``latent_affine`` is a sister
        # L1 EXTENSION). The PyTorch sister default also matches "none".
        context_conditioning_mode="none",
    )


def _split_state_dict_for_z7mcm2(
    exported_state_dict: dict[str, np.ndarray],
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Split MLX-exported state_dict into Z7MCM2 sections.

    The exported dict has PyTorch-layout keys:
    - ``predictor.*`` (Mamba-2 cell + input/output projections)
    - ``decoder.*`` (initial_proj + PixelShuffle conv stack + final conv)
    - ``latent_init`` / ``residuals`` / ``ego_motion_buffer``
    - (optional) ``context_conditioner.*`` for latent_affine mode

    Returns:
        ``(encoder_sd, decoder_sd, predictor_sd, latent_init, residuals, ego)``
        where the encoder_sd is empty when context mode is "none" and populated
        with ``context_conditioner.*`` keys when "latent_affine" is active
        (sister of Z7-LSTM/GRU convention).
    """
    encoder_sd: dict[str, torch.Tensor] = {}
    decoder_sd: dict[str, torch.Tensor] = {}
    predictor_sd: dict[str, torch.Tensor] = {}
    latent_init: torch.Tensor | None = None
    residuals: torch.Tensor | None = None
    ego_motion: torch.Tensor | None = None

    for key, arr in exported_state_dict.items():
        tensor = torch.from_numpy(np.asarray(arr).copy()).to(dtype=torch.float32)
        if key == "latent_init":
            latent_init = tensor
        elif key == "residuals":
            residuals = tensor
        elif key == "ego_motion_buffer":
            ego_motion = tensor
        elif key.startswith("predictor."):
            predictor_sd[key] = tensor
        elif key.startswith("decoder."):
            decoder_sd[key] = tensor
        elif key.startswith("context_conditioner."):
            encoder_sd[key] = tensor
        else:
            # Unknown key — surface explicitly so a future state_dict shape
            # drift fails closed instead of silently routing into the wrong
            # section (sister of Catalog #229 premise-verification discipline).
            raise ValueError(
                f"unexpected Z7-Mamba-2 exported state_dict key {key!r}; "
                "expected one of {predictor.*, decoder.*, "
                "context_conditioner.*, latent_init, residuals, "
                "ego_motion_buffer}"
            )

    if latent_init is None or residuals is None or ego_motion is None:
        raise ValueError(
            "exported_state_dict missing required {latent_init, residuals, "
            "ego_motion_buffer}; got keys: "
            f"{sorted(exported_state_dict.keys())}"
        )
    return encoder_sd, decoder_sd, predictor_sd, latent_init, residuals, ego_motion


def z7_mamba2_meta_from_config(
    cfg: Z7Mamba2MLXRenderConfig,
) -> dict[str, object]:
    """Minimal receiver metadata needed to rebuild Z7Mamba2 at inflate time."""
    return {
        "latent_dim": int(cfg.latent_dim),
        "ego_motion_dim": int(cfg.ego_motion_dim),
        "d_model": int(cfg.d_model),
        "d_state": int(cfg.d_state),
        "expand": int(cfg.expand),
        "d_conv": int(cfg.d_conv),
        "decoder_embed_dim": int(cfg.decoder_embed_dim),
        "decoder_initial_grid_h": int(cfg.decoder_initial_grid_h),
        "decoder_initial_grid_w": int(cfg.decoder_initial_grid_w),
        "decoder_channels": [int(c) for c in cfg.decoder_channels],
        "decoder_num_upsample_blocks": int(cfg.decoder_num_upsample_blocks),
        "output_height": int(cfg.output_height),
        "output_width": int(cfg.output_width),
        "stateful": bool(cfg.stateful),
        "identity_predictor": bool(cfg.identity_predictor),
        "context_conditioning_mode": "none",
        "mamba2_mlx_backend_lineage": str(cfg.mamba2_mlx_backend_lineage),
        "canonical_ssd_mlx_backend_wired": bool(cfg.canonical_ssd_mlx_backend_wired),
        "canonical_ssd_mlx_blocker": str(cfg.canonical_ssd_mlx_blocker),
    }


def pack_archive_from_exported_state_dict(
    *,
    exported_state_dict: dict[str, np.ndarray],
    mlx_cfg: Z7Mamba2MLXRenderConfig,
) -> bytes:
    """Pack a PyTorch-layout exported MLX state dict into Z7MCM2 ``0.bin`` bytes."""
    (
        encoder_sd,
        decoder_sd,
        predictor_sd,
        latent_init,
        residuals,
        ego_motion,
    ) = _split_state_dict_for_z7mcm2(exported_state_dict)
    pytorch_cfg = z7_mamba2_pytorch_config_from_mlx(mlx_cfg)
    return pack_archive(
        encoder_state_dict=encoder_sd,
        decoder_state_dict=decoder_sd,
        predictor_state_dict=predictor_sd,
        latent_init=latent_init,
        residuals=residuals,
        ego_motion=ego_motion,
        meta=z7_mamba2_meta_from_config(mlx_cfg),
        config=pytorch_cfg,
    )


def export_z7_mamba2_mlx_archive(
    model: Any,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> tuple[Path, str, int]:
    """Export an MLX Z7-Mamba-2 model as a contest-shaped ``archive.zip``.

    Per the canonical mlx_score_aware harness contract
    (``tac.substrates._shared.mlx_score_aware.bundle.RendererBundle.export_archive_fn``):
    returns ``(archive_zip_path, sha256_hex, size_bytes)``.

    Sister of :func:`tac.substrates.z6_v2_cargo_cult_unwind.archive_candidate.export_z6_v2_mlx_archive`.
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

    cfg = model.cfg
    bin_bytes = pack_archive_from_exported_state_dict(
        exported_state_dict=model.export_state_dict(),
        mlx_cfg=cfg,
    )
    bin_path = out_dir / "0.bin"
    bin_path.write_bytes(bin_bytes)

    submission_dir = out_dir / "submission"
    # Wave N+9 Slot 1 self-containment fix (Catalog #295 + #361): Z7-Mamba-2
    # `inflate.py` transitively imports (a) `tac.substrates._shared.inflate_runtime`
    # for `select_inflate_device` / `raw_output_path` / `write_rgb_pair_to_raw`,
    # (b) `tac.substrates.time_traveler_l5_z6.architecture` for `_Z6Decoder` +
    # `EVAL_HW`, and (c) `tac.substrates.time_traveler_l5_z7_lstm_predictive_coding.architecture`
    # for `LatentAffineContextConditioner` + `Z7GruPredictiveCodingConfig` +
    # `normalize_context_conditioning_mode`. Without explicit vendoring the
    # shipped submission fails at first import (empirically verified at
    # Wave N+9 Slot 1 landing). The canonical helper was extended in the
    # same commit batch to accept `vendor_extra_substrate_packages` +
    # `vendor_shared_inflate_runtime` per the sister-extinction architecture.
    write_contest_runtime(
        submission_dir,
        substrate_pkg_name="time_traveler_l5_z7_mamba2",
        repo_root=root,
        vendor_shared_inflate_runtime=True,
        vendor_extra_substrate_packages=(
            ("time_traveler_l5_z6", ("architecture.py",)),
            ("time_traveler_l5_z7_lstm_predictive_coding", ("architecture.py",)),
        ),
        vendor_extra_tac_subpackages=(
            ("optimization", ("mamba2_predictor.py",)),
        ),
    )
    (submission_dir / "0.bin").write_bytes(bin_bytes)
    archive_zip_path = out_dir / "archive.zip"
    build_archive_zip(
        archive_zip_path,
        bin_bytes=bin_bytes,
        submission_dir=submission_dir,
    )
    return (
        archive_zip_path,
        sha256_file(archive_zip_path),
        archive_zip_path.stat().st_size,
    )


__all__ = [
    "export_z7_mamba2_mlx_archive",
    "pack_archive_from_exported_state_dict",
    "z7_mamba2_meta_from_config",
    "z7_mamba2_pytorch_config_from_mlx",
]
