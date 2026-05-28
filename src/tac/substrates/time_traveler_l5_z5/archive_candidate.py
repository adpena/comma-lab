# SPDX-License-Identifier: MIT
"""Byte-closed Z5 Rao-Ballard archive export bridge for MLX-trained models.

Sister of :mod:`tac.substrates.z6_v2_cargo_cult_unwind.archive_candidate` per
the 11th INDIVIDUALLY-FRACTAL standing directive 2026-05-27 — Z5's OWN
archive export pass, NOT shared-helper shortcut from Z6/Z7 sisters.

Z5's substrate-distinguishing primitive (2-level Rao-Ballard hierarchical
predictor with EXPLICIT z_high + ego_motion -> z_low_pred forecast) is
implemented at the ``architecture.py`` + ``mlx_renderer.py`` surfaces; this
module bridges the MLX-trained state_dict into the Z5RB1 byte-closed archive
used by the inflate runtime.

[verified-against: src/tac/substrates/time_traveler_l5_z5/architecture.py]
[verified-against: src/tac/substrates/time_traveler_l5_z5/archive.py]
[verified-against: src/tac/substrates/time_traveler_l5_z5/mlx_renderer.py]
[verified-against: src/tac/substrates/z6_v2_cargo_cult_unwind/archive_candidate.py canonical pattern]
[verified-against: Catalog #146 contest-compliant inflate runtime template]
[verified-against: Catalog #205 canonical select_inflate_device]
[verified-against: Catalog #295 PYTHONPATH self-containment]
[verified-against: Catalog #361 vendor_module_with_fresh_mtime]
[verified-against: Catalog #367 raw bytes contract 1164x874x1200x3]
"""

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
from tac.substrates.time_traveler_l5_z5.architecture import Z5RaoBallardConfig
from tac.substrates.time_traveler_l5_z5.archive import pack_archive


def z5_meta_from_config(cfg: Z5RaoBallardConfig) -> dict[str, object]:
    """Minimal receiver metadata needed to rebuild Z5RaoBallardSubstrate at inflate."""
    return {
        "embed_dim": int(cfg.embed_dim),
        "initial_grid_h": int(cfg.initial_grid_h),
        "initial_grid_w": int(cfg.initial_grid_w),
        "decoder_channels": [int(c) for c in cfg.decoder_channels],
        "num_upsample_blocks": int(cfg.num_upsample_blocks),
        "sin_frequency": float(cfg.sin_frequency),
        "film_generator_depth": int(cfg.film_generator_depth),
        "film_hidden_width": int(cfg.film_hidden_width),
        "output_height": int(cfg.output_height),
        "output_width": int(cfg.output_width),
        "predictor_hidden_dim": int(cfg.predictor_hidden_dim),
        "predictor_num_layers": int(cfg.predictor_num_layers),
        "lambda_residual": float(cfg.lambda_residual),
        "cooperative_receiver_beta": float(cfg.cooperative_receiver_beta),
    }


def pack_archive_from_exported_state_dict(
    *,
    exported_state_dict: dict[str, np.ndarray],
    cfg: Z5RaoBallardConfig,
) -> bytes:
    """Pack a PyTorch-layout exported MLX state dict into Z5RB1 ``0.bin`` bytes.

    Splits the unified state_dict into the canonical 3 weight pools (decoder /
    predictor / per-pair latents) that the Z5RB1 grammar's separate brotli
    blobs require.
    """
    required = {"low_latents", "high_latents", "ego_vecs"}
    for r in required:
        if r not in exported_state_dict:
            raise ValueError(
                f"exported_state_dict missing required key {r!r}; "
                f"got {sorted(exported_state_dict.keys())[:10]}..."
            )
    decoder_state: dict[str, torch.Tensor] = {}
    predictor_state: dict[str, torch.Tensor] = {}
    low_latents: torch.Tensor | None = None
    high_latents: torch.Tensor | None = None
    ego_vecs: torch.Tensor | None = None
    for name, arr in exported_state_dict.items():
        tensor = torch.from_numpy(np.asarray(arr).copy()).to(dtype=torch.float32)
        if name == "low_latents":
            low_latents = tensor
        elif name == "high_latents":
            high_latents = tensor
        elif name == "ego_vecs":
            ego_vecs = tensor
        elif name.startswith("predictor."):
            # Strip the "predictor." prefix to match the predictor submodule's
            # state_dict naming (the inflate runtime calls predictor.load_state_dict
            # on this dict directly).
            predictor_state[name[len("predictor."):]] = tensor
        elif name.startswith("decoder."):
            decoder_state[name[len("decoder."):]] = tensor
        else:
            raise ValueError(
                f"unexpected state_dict key {name!r}; expected prefix "
                "low_latents / high_latents / ego_vecs / predictor. / decoder."
            )
    assert low_latents is not None
    assert high_latents is not None
    assert ego_vecs is not None
    return pack_archive(
        decoder_state,
        predictor_state,
        low_latents,
        high_latents,
        ego_vecs,
        z5_meta_from_config(cfg),
    )


def export_z5_mlx_archive(
    model: Any,
    output_dir: str | Path,
    *,
    repo_root: str | Path | None = None,
) -> tuple[Path, str, int]:
    """Export an MLX Z5 model as a contest-shaped ``archive.zip``.

    Mirrors the canonical Z6-v2 sister
    :func:`tac.substrates.z6_v2_cargo_cult_unwind.archive_candidate.export_z6_v2_mlx_archive`.
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
        cfg=cfg,
    )
    bin_path = out_dir / "0.bin"
    bin_path.write_bytes(bin_bytes)

    submission_dir = out_dir / "submission"
    write_contest_runtime(
        submission_dir,
        substrate_pkg_name="time_traveler_l5_z5",
        repo_root=root,
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
    "export_z5_mlx_archive",
    "pack_archive_from_exported_state_dict",
    "z5_meta_from_config",
]
