# SPDX-License-Identifier: MIT
"""Byte-closed PSV2 archive export helpers for MLX/local training artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from tac.repo_io import sha256_file
from tac.substrates._shared.pact_nerv_full_main import (
    build_archive_zip,
    write_contest_runtime,
)
from tac.substrates.pact_nerv_selector_v2.architecture import (
    FEC6_FIXED_K16_MODE_IDS,
    ArithmeticSelectorCoder,
    PactNervSelectorV2Config,
)
from tac.substrates.pact_nerv_selector_v2.archive import pack_archive


def selector_v2_meta_from_config(cfg: PactNervSelectorV2Config) -> dict[str, object]:
    """Return the minimal receiver metadata needed to rebuild the decoder."""

    coder = ArithmeticSelectorCoder(cfg.selector_palette_size)
    return {
        "embed_dim": int(cfg.embed_dim),
        "initial_grid_h": int(cfg.initial_grid_h),
        "initial_grid_w": int(cfg.initial_grid_w),
        "decoder_channels": [int(value) for value in cfg.decoder_channels],
        "sin_frequency": float(cfg.sin_frequency),
        "num_upsample_blocks": int(cfg.num_upsample_blocks),
        "output_height": int(cfg.output_height),
        "output_width": int(cfg.output_width),
        "selector_cum_freq": [int(value) for value in coder.cum_freq],
        "selector_precision": int(coder.precision),
        "selector_mode_ids": list(FEC6_FIXED_K16_MODE_IDS),
    }


def pack_archive_from_exported_state_dict(
    *,
    exported_state_dict: dict[str, np.ndarray],
    cfg: PactNervSelectorV2Config,
    selectors: np.ndarray | None = None,
) -> bytes:
    """Pack a PyTorch-layout exported MLX state dict into PSV2 ``0.bin`` bytes."""

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
    coder = ArithmeticSelectorCoder(cfg.selector_palette_size)
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
        selector_v2_meta_from_config(cfg),
        palette_size=int(cfg.selector_palette_size),
    )


def export_pact_nerv_selector_v2_mlx_archive(
    model: Any,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> tuple[Path, str, int]:
    """Export an MLX Selector-V2 model as a contest-shaped archive.zip."""

    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[4]
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
        substrate_pkg_name="pact_nerv_selector_v2",
        repo_root=root,
    )
    (submission_dir / "0.bin").write_bytes(bin_bytes)
    archive_zip_path = out_dir / "archive.zip"
    build_archive_zip(
        archive_zip_path,
        bin_bytes=bin_bytes,
        submission_dir=submission_dir,
    )
    return archive_zip_path, sha256_file(archive_zip_path), archive_zip_path.stat().st_size


__all__ = [
    "export_pact_nerv_selector_v2_mlx_archive",
    "pack_archive_from_exported_state_dict",
    "selector_v2_meta_from_config",
]
